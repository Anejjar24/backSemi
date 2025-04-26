from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import io
import re
import subprocess
import traceback
import pandas as pd
import numpy as np
import tempfile
import os

def index(request):
    """
    Render the main page of the application.
    """
    return render(request, 'pages/index.html')

@csrf_exempt
def generate_code(request):
    """
    Generate Python code from natural language prompt and optional CSV data.
    Takes a POST request with 'prompt' and optional 'csv' fields.
    Returns the generated Python code optimized for data science tasks.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=405)
    
    try:
        # Parse request data
        data = json.loads(request.body)
        prompt = data.get('prompt', '')
        csv_content = data.get('csv', '')
        
        if not prompt:
            return JsonResponse({'error': 'No prompt provided. Please describe what you want to do with your data.'}, status=400)
        
        # Process CSV data if provided
        df = None
        csv_preview = "No CSV data provided."
        csv_file_path = None
        
        if csv_content:
            df, csv_preview, csv_file_path = process_csv_data(csv_content)
            
        # Generate complete prompt with necessary context
        full_prompt = generate_full_prompt(prompt, df, csv_preview)
        
        # Get code from LLM
        generated_code = get_code_from_llm(full_prompt)
        
        # Enhance the generated code
        enhanced_code = enhance_generated_code(generated_code, df)
        
        # Validate code if CSV data was provided
        validation_result = None
        if df is not None and csv_file_path:
            validation_result = validate_code(enhanced_code, csv_file_path)
        
        # Prepare response with code and explanations
        response = {
            'code': enhanced_code,
            'explanation': generate_explanation(enhanced_code, df),
            'validation': validation_result
        }
        
        # Clean up temporary files
        if csv_file_path and os.path.exists(csv_file_path):
            os.remove(csv_file_path)
            
        return JsonResponse(response)
        
    except Exception as e:
        error_details = traceback.format_exc()
        return JsonResponse({
            'error': 'Processing failed', 
            'details': str(e),
            'traceback': error_details
        }, status=500)

def process_csv_data(csv_content):
    """
    Process CSV data using multiple methods to handle various formats.
    Returns the DataFrame, a preview of the data, and path to a temporary CSV file.
    """
    df = None
    csv_preview = "Failed to process CSV data."
    csv_file_path = None
    
    try:
        # Try multiple parsing methods
        parsing_methods = [
            # Standard CSV parsing
            lambda: pd.read_csv(io.StringIO(csv_content)),
            # Try with different separators
            lambda: pd.read_csv(io.StringIO(csv_content), sep=';'),
            lambda: pd.read_csv(io.StringIO(csv_content), sep='\t'),
            # Try with auto separator detection
            lambda: pd.read_csv(io.StringIO(csv_content), sep=None, engine='python'),
            # Try with header inference
            lambda: pd.read_csv(io.StringIO(csv_content), header='infer'),
            # Try with fixed width format
            lambda: pd.read_fwf(io.StringIO(csv_content))
        ]
        
        # Try each parsing method until one works
        for method in parsing_methods:
            try:
                df = method()
                if df is not None and not df.empty and len(df.columns) > 1:
                    break
            except:
                continue
        
        # If all methods fail, try custom parsing
        if df is None or df.empty or len(df.columns) <= 1:
            df = custom_parse_csv(csv_content)
        
        # Create a preview and save temp file for validation
        if df is not None and not df.empty:
            # Clean column names
            df.columns = [str(col).strip().replace(' ', '_').lower() for col in df.columns]
            
            # Create preview
            preview_rows = min(5, len(df))
            csv_preview = f"Data preview ({df.shape[0]} rows, {df.shape[1]} columns):\n"
            csv_preview += df.head(preview_rows).to_string(index=False)
            
            # Add data summary
            csv_preview += f"\n\nData summary:\n"
            csv_preview += df.describe(include='all').to_string()
            
            # Save to temporary file for validation
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
                csv_file_path = temp_file.name
                df.to_csv(csv_file_path, index=False)
        
    except Exception as e:
        csv_preview = f"Error processing CSV: {str(e)}"
    
    return df, csv_preview, csv_file_path

def custom_parse_csv(csv_content):
    """
    Custom CSV parser for complex or non-standard formats.
    """
    try:
        lines = csv_content.strip().split('\n')
        all_entries = []
        
        # Try to detect delimiter by checking first line
        first_line = lines[0] if lines else ""
        potential_delimiters = [',', ';', '\t', ' ']
        delimiter = max(potential_delimiters, key=lambda d: first_line.count(d))
        
        for line in lines:
            entries = line.strip().split(delimiter)
            if entries:
                all_entries.append(entries)
        
        if all_entries:
            # Detect number of columns based on most common row length
            row_lengths = [len(row) for row in all_entries]
            most_common_length = max(set(row_lengths), key=row_lengths.count)
            
            # Process header row if present (first row with enough columns)
            header_candidates = [row for row in all_entries if len(row) == most_common_length]
            header = header_candidates[0] if header_candidates else [f'col_{i}' for i in range(most_common_length)]
            
            # Filter data rows to those with the correct number of columns
            data_rows = [row for row in all_entries if len(row) == most_common_length]
            if len(data_rows) > 0 and header == data_rows[0]:
                data_rows = data_rows[1:]  # Remove header from data if it's the first row
                
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=header)
            return df
    except:
        pass
        
    return pd.DataFrame()  # Return empty DataFrame if all parsing attempts fail

def generate_full_prompt(prompt, df, csv_preview):
    """
    Generate a comprehensive prompt for the LLM with context about the data.
    """
    # Base prompt with user request
    full_prompt = f"""
Generate complete, well-documented Python code that solves the following task:
{prompt}

"""
    # Add CSV data information if available
    if df is not None and not df.empty:
        # Add data preview
        full_prompt += f"""
Here's information about the provided CSV data:
{csv_preview}

DataFrame Information:
- Rows: {df.shape[0]}
- Columns: {df.shape[1]}
- Column names: {', '.join(df.columns.tolist())}

Data types:
{df.dtypes.to_string()}

"""
        # Add missing values information
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            full_prompt += f"""
Missing values per column:
{missing_data[missing_data > 0].to_string()}

"""
        # Add categorical columns information
        cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_columns:
            full_prompt += f"""
Categorical columns: {', '.join(cat_columns)}

"""
        # Add numerical columns information
        num_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if num_columns:
            full_prompt += f"""
Numerical columns: {', '.join(num_columns)}

"""

    # Add detailed instructions for code generation
    full_prompt += """
Instructions for generating the Python code:
1. Write complete, executable code that works with the provided data.
2. Include all necessary imports at the beginning.
3. Add clear comments explaining each step of the data processing and analysis.
4. Include data cleaning steps (handling missing values, data type conversion).
5. For machine learning tasks, include data preprocessing, model training, evaluation, and visualization of results.
6. Include visualizations (plots, charts) to help understand the data and results.
7. Structure the code in logical sections: data loading, preprocessing, analysis, visualization, and results.
8. Make sure the code is optimized and follows best practices.
9. Use pandas, numpy, matplotlib, seaborn, and scikit-learn libraries as needed.
10. Include error handling for common issues.

The code should be self-contained and executable by researchers with minimal Python knowledge.
"""

    return full_prompt

def get_code_from_llm(prompt):
    """
    Get code from LLM (CodeLlama) using the provided prompt.
    """
    try:
        # Run CodeLlama with the prompt
        result = subprocess.run(
            ['ollama', 'run', 'codellama', prompt],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=600  # Increased timeout for complex tasks
        )

        if result.returncode != 0:
            return f"# Error running LLM: {result.stderr}\n\n# Please try again with a more specific prompt."

        response_text = result.stdout.strip()

        # Extract code from markdown code blocks
        code_pattern = r"```(?:python)?\s*(.*?)\s*```"
        code_matches = re.findall(code_pattern, response_text, re.DOTALL)
        
        if code_matches:
            # Combine all code blocks if multiple are found
            return "\n\n".join(code_matches)
        else:
            # If no code blocks are found, use the entire response
            return response_text
            
    except Exception as e:
        return f"# Error generating code: {str(e)}\n\n# Please try again with a more specific prompt."

def enhance_generated_code(code, df):
    """
    Enhance the generated code with additional imports, error handling, and best practices.
    """
    # Ensure standard imports are present
    standard_imports = [
        "import pandas as pd",
        "import numpy as np",
        "import matplotlib.pyplot as plt",
        "import seaborn as sns",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.preprocessing import StandardScaler, LabelEncoder",
        "from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report",
        "import warnings",
        "warnings.filterwarnings('ignore')"
    ]

    # Check which imports are already in the code
    existing_imports = []
    for imp in standard_imports:
        if imp in code:
            existing_imports.append(imp)
    
    # Add missing imports
    missing_imports = [imp for imp in standard_imports if imp not in existing_imports]
    enhanced_code = "\n".join(missing_imports) + "\n\n" if missing_imports else ""
    
    # Add main guard if not present
    if "if __name__ == '__main__'" not in code:
        enhanced_code += code + "\n\n"
        enhanced_code += """
# Execute the script when run directly
if __name__ == '__main__':
    # Set plot style for better visualization
    plt.style.use('seaborn-whitegrid')
    sns.set_context("notebook", font_scale=1.2)
    
    try:
        print("Starting data analysis...")
        # The main code logic has been executed above
        print("Analysis completed successfully!")
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
"""
    else:
        enhanced_code += code
    
    # Add pretty printing of DataFrames if pandas is used
    if "pd.read_csv" in enhanced_code or "DataFrame" in enhanced_code:
        pd_config = """
# Configure pandas to display more rows and columns
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)
"""
        # Insert after imports but before main code
        import_section_end = enhanced_code.find("import") + len("import")
        while import_section_end < len(enhanced_code) and enhanced_code.find("import", import_section_end) != -1:
            next_import = enhanced_code.find("import", import_section_end)
            if next_import - import_section_end > 10:  # If there's a gap between imports
                break
            import_section_end = next_import + len("import")
        
        # Find the next line after imports
        next_line = enhanced_code.find("\n", import_section_end) + 1
        enhanced_code = enhanced_code[:next_line] + pd_config + enhanced_code[next_line:]
    
    return enhanced_code

def validate_code(code, csv_file_path):
    """
    Validate the generated code by running it against the provided CSV file.
    Returns validation results or error messages.
    """
    # Create a temporary Python file with the code
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
        script_path = temp_file.name
        
        # Add code to use the temporary CSV file
        modified_code = code.replace("pd.read_csv(", f"pd.read_csv(r'{csv_file_path}' #")
        temp_file.write(modified_code.encode('utf-8'))
    
    try:
        # Try to run the script
        result = subprocess.run(
            ['python', script_path],
            capture_output=True,
            text=True,
            timeout=60  # Limit execution time
        )
        
        # Clean up the temporary script file
        if os.path.exists(script_path):
            os.remove(script_path)
            
        if result.returncode == 0:
            return {
                "status": "success",
                "message": "The code executed successfully.",
                "output": result.stdout[:500] + ("..." if len(result.stdout) > 500 else "")
            }
        else:
            return {
                "status": "error",
                "message": "The code failed to execute.",
                "error": result.stderr[:500] + ("..." if len(result.stderr) > 500 else "")
            }
    except Exception as e:
        # Clean up the temporary script file
        if os.path.exists(script_path):
            os.remove(script_path)
            
        return {
            "status": "error",
            "message": f"Error validating code: {str(e)}",
            "error": traceback.format_exc()[:500]
        }

def generate_explanation(code, df):
    """
    Generate a plain language explanation of what the code does.
    """
    explanation = "Voici une explication du code généré:\n\n"
    
    # Identify sections in the code
    sections = []
    if "import" in code:
        sections.append("Import des bibliothèques nécessaires")
    if "read_csv" in code:
        sections.append("Chargement des données")
    if ".dropna" in code or ".fillna" in code:
        sections.append("Nettoyage des données manquantes")
    if "describe" in code or "info" in code:
        sections.append("Analyse descriptive des données")
    if "plt." in code or "sns." in code:
        sections.append("Visualisation des données")
    if "train_test_split" in code:
        sections.append("Division des données en ensembles d'entraînement et de test")
    if "fit" in code:
        sections.append("Entraînement d'un modèle")
    if "predict" in code:
        sections.append("Prédiction et évaluation du modèle")
    
    # Create a summary of what the code does
    explanation += "Le code effectue les étapes suivantes:\n"
    for i, section in enumerate(sections, 1):
        explanation += f"{i}. {section}\n"
    
    explanation += "\nCe script peut être exécuté sans modifications supplémentaires et produira des résultats basés sur vos données."
    
    # Add tips for understanding the results
    if df is not None:
        explanation += "\n\nConseil pour l'interprétation des résultats:"
        
        if "accuracy_score" in code or "classification_report" in code:
            explanation += "\n- Les métriques comme l'accuracy et le F1-score indiquent la performance du modèle de classification."
        if "mean_squared_error" in code:
            explanation += "\n- L'erreur quadratique moyenne (MSE) indique la précision des prédictions du modèle de régression."
        if "plt.scatter" in code:
            explanation += "\n- Les graphiques de dispersion montrent les relations entre deux variables."
        if "plt.hist" in code:
            explanation += "\n- Les histogrammes montrent la distribution des données."
        if "confusion_matrix" in code:
            explanation += "\n- La matrice de confusion montre les vrais positifs, faux positifs, vrais négatifs et faux négatifs."
    
    return explanation