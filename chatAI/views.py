import json
import os
import tempfile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
import requests
import pandas as pd
import io
import traceback
import logging
from .models import Conversation, Message

# Configurez le logging
logger = logging.getLogger(__name__)

# URL de l'API Ollama
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Configuration du modèle
MODEL_NAME = "codellama"

@csrf_exempt
@login_required
def generate_code(request):
    """
    Vue pour générer du code Python basé sur une requête de l'utilisateur
    et éventuellement un fichier CSV, et sauvegarder la conversation.
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Seules les requêtes POST sont acceptées'}, status=405)
    
    try:
        # Récupération des données de la requête
        logger.debug(f"Headers: {request.headers}")
        logger.debug(f"POST data: {request.POST}")
        logger.debug(f"FILES: {request.FILES}")
        
        # Initialisation des variables
        query = ""
        body = {}
        csv_content = None
        csv_context = ""
        conversation_id = None
        
        # Extraction du prompt/query et de l'ID de conversation
        if 'data' in request.POST:
            try:
                data_json = json.loads(request.POST.get('data', '{}'))
                query = data_json.get('query', '')
                conversation_id = data_json.get('conversation_id', None)
            except json.JSONDecodeError:
                logger.error("Erreur de décodage JSON dans les données POST")
                return JsonResponse({
                    'success': False, 
                    'error': 'Format JSON invalide dans les données'
                }, status=400)
        else:
            # Tenter de lire directement du corps de la requête
            try:
                body = json.loads(request.body.decode('utf-8'))
                query = body.get('prompt', '') or body.get('query', '')
                conversation_id = body.get('conversation_id', None)
            except json.JSONDecodeError:
                logger.error("Erreur de décodage JSON dans le corps de la requête")
                return JsonResponse({
                    'success': False, 
                    'error': 'Format de requête invalide'
                }, status=400)
        
        if not query:
            return JsonResponse({
                'success': False, 
                'error': 'La requête doit contenir un prompt ou une query'
            }, status=400)
        
        # Gestion de la conversation
        if conversation_id:
            try:
                conversation = Conversation.objects.get(id=conversation_id, user=request.user)
            except Conversation.DoesNotExist:
                # Si la conversation n'existe pas ou n'appartient pas à l'utilisateur, créer une nouvelle
                conversation = Conversation.objects.create(
                    user=request.user,
                    title=f"Conversation sur {query[:30]}..." if len(query) > 30 else query
                )
        else:
            # Créer une nouvelle conversation
            conversation = Conversation.objects.create(
                user=request.user,
                title=f"Conversation sur {query[:30]}..." if len(query) > 30 else query
            )
        
        # Traitement du fichier CSV si présent
        csv_file_obj = None
        if 'csv_file' in request.FILES:
            csv_file = request.FILES['csv_file']
            csv_file_obj = csv_file  # Sauvegarde du fichier pour la base de données
            try:
                # Lire le contenu du fichier avec pandas
                csv_df = pd.read_csv(csv_file, encoding='utf-8')
                
                # Créer un aperçu du CSV pour le contexte
                csv_info = (
                    f"Informations sur le CSV:\n"
                    f"- Nom: {csv_file.name}\n"
                    f"- Dimensions: {csv_df.shape[0]} lignes x {csv_df.shape[1]} colonnes\n"
                    f"- Colonnes: {', '.join(csv_df.columns.tolist())}\n"
                    f"- Types de données: {', '.join([f'{col}: {dtype}' for col, dtype in zip(csv_df.columns, csv_df.dtypes.astype(str))])}\n\n"
                    f"Voici les 5 premières lignes du CSV:\n{csv_df.head(5).to_string()}\n\n"
                )
                
                # Stocker le contenu CSV pour la génération
                csv_content = csv_df.to_csv(index=False)
                csv_context = csv_info
                
                logger.debug(f"CSV traité avec succès: {csv_file.name}")
            except Exception as e:
                logger.error(f"Erreur de traitement du CSV: {str(e)}")
                return JsonResponse({
                    'success': False, 
                    'error': f'Erreur lors du traitement du fichier CSV: {str(e)}'
                }, status=400)
        elif body and 'csv' in body and body['csv']:  # Vérification que body existe et contient csv
            # Si le CSV est directement fourni dans le corps de la requête (comme chaîne)
            try:
                csv_content = body['csv']
                csv_df = pd.read_csv(io.StringIO(csv_content), encoding='utf-8')
                csv_info = (
                    f"Informations sur le CSV:\n"
                    f"- Dimensions: {csv_df.shape[0]} lignes x {csv_df.shape[1]} colonnes\n"
                    f"- Colonnes: {', '.join(csv_df.columns.tolist())}\n"
                    f"- Types de données: {', '.join([f'{col}: {dtype}' for col, dtype in zip(csv_df.columns, csv_df.dtypes.astype(str))])}\n\n"
                    f"Voici les 5 premières lignes du CSV:\n{csv_df.head(5).to_string()}\n\n"
                )
                csv_context = csv_info
                logger.debug("CSV fourni dans le corps de la requête traité avec succès")
            except Exception as e:
                logger.error(f"Erreur de traitement du CSV fourni dans le corps: {str(e)}")
                return JsonResponse({
                    'success': False, 
                    'error': f'Erreur lors du traitement du CSV fourni: {str(e)}'
                }, status=400)
        
        # Enregistrer le message de l'utilisateur
        user_message = Message.objects.create(
            conversation=conversation,
            role='user',
            content=query,
            csv_file=csv_file_obj
        )
        
        # Construction du prompt pour Ollama
        system_prompt = (
            "Tu es un assistant spécialisé dans la génération de code Python. "
            "Réponds uniquement avec du code Python exécutable et bien commenté. "
            "Ne fournis pas d'explications, uniquement du code Python. "
            "Assure-toi que le code soit correct, efficace et bien structuré."
        )
        
        full_prompt = f"""
{system_prompt}

{csv_context}

L'utilisateur demande:
{query}

Génère un script Python complet qui répond à cette demande.
```python
"""
        
        # Requête vers Ollama
        logger.debug(f"Envoi de la requête à Ollama avec le prompt: {full_prompt[:100]}...")
        
        try:
            ollama_payload = {
                "model": MODEL_NAME,
                "prompt": full_prompt,
                "stream": False
            }
            
            response = requests.post(
                OLLAMA_API_URL,
                json=ollama_payload
            )
            
            if response.status_code != 200:
                logger.error(f"Erreur Ollama: {response.status_code} - {response.text}")
                return JsonResponse({
                    'success': False, 
                    'error': f"Le service de génération de code a retourné une erreur: {response.status_code}"
                }, status=500)
            
            # Traitement de la réponse d'Ollama
            response_data = response.json()
            generated_text = response_data.get('response', '')
            
            # Extraction du code Python de la réponse
            # Généralement, le modèle renvoie le code entre des balises ```python et ```
            code = generated_text
            
            # Nettoyer le code (supprimer les balises de code si présentes)
            if "```python" in code and "```" in code[code.find("```python")+10:]:
                code = code[code.find("```python")+10:]
                code = code[:code.find("```")]
            elif "```" in code and "```" in code[code.find("```")+3:]:
                code = code[code.find("```")+3:]
                code = code[:code.find("```")]
            
            # Nettoyer les espaces en début et fin
            code = code.strip()
            
            # Exécution du code si un CSV est fourni (option avancée)
            result = None
            if csv_content:
                result = execute_code_with_csv(code, csv_content)
            
            # Enregistrer la réponse de l'assistant
            assistant_message = Message.objects.create(
                conversation=conversation,
                role='assistant',
                content="Code généré",
                generated_code=code,
                execution_result=result
            )
            
            # Mettre à jour la conversation
            conversation.updated_at = assistant_message.created_at
            conversation.save()
            
            return JsonResponse({
                'success': True,
                'code': code,
                'result': result,
                'conversation_id': conversation.id,
                'message_id': assistant_message.id
            })
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération du code: {str(e)}")
            logger.error(traceback.format_exc())
            return JsonResponse({
                'success': False, 
                'error': f"Erreur lors de la génération du code: {str(e)}"
            }, status=500)
    
    except Exception as e:
        logger.error(f"Erreur générale: {str(e)}")
        logger.error(traceback.format_exc())
        return JsonResponse({
            'success': False, 
            'error': f"Erreur inattendue: {str(e)}"
        }, status=500)


def execute_code_with_csv(code, csv_content):
    """
    Exécute le code Python généré en lui fournissant le contenu CSV.
    Retourne les résultats de l'exécution sous forme de dictionnaire.
    """
    # Créer un environnement d'exécution isolé
    execution_locals = {}
    execution_globals = {
        'pd': pd,
        'numpy': __import__('numpy'),
        'plt': __import__('matplotlib.pyplot'),
        'io': io,
        'StringIO': io.StringIO
    }
    
    # Préparer l'environnement avec le CSV
    setup_code = f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# Capturer la sortie standard
output_capture = io.StringIO()
error_capture = io.StringIO()

# Charger le CSV dans un DataFrame
csv_data = '''
{csv_content}
'''
try:
    df = pd.read_csv(StringIO(csv_data))
except Exception as e:
    print(f"Erreur lors du chargement du CSV: {{str(e)}}")
    df = None
"""
    
    # Créer un dossier temporaire pour les figures
    with tempfile.TemporaryDirectory() as temp_dir:
        figures_dir = os.path.join(temp_dir, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Ajouter le code pour sauvegarder les figures
        save_figures_code = """
# Sauvegarder toutes les figures générées
figure_paths = []
for i, fig in enumerate(plt.get_fignums()):
    figure = plt.figure(fig)
    path = f'figures/figure_{i}.png'
    figure.savefig(path)
    figure_paths.append(path)
"""
        
        # Combiner tous les codes
        full_code = setup_code + "\n\n"
        
        # Indenter le code de l'utilisateur pour le placer dans un try/except
        indented_code = "\n".join(["    " + line for line in code.split("\n")])
        full_code += f"""
# Début du code généré
try:
    with redirect_stdout(output_capture), redirect_stderr(error_capture):
{indented_code}

    # Sauvegarder les figures
{save_figures_code}
except Exception as e:
    error_capture.write(f"Exception lors de l'exécution: {{str(e)}}")
"""
        
        try:
            # Exécuter le code complet
            exec(full_code, execution_globals, execution_locals)
            
            # Récupérer les sorties
            output = execution_locals.get('output_capture', io.StringIO()).getvalue()
            errors = execution_locals.get('error_capture', io.StringIO()).getvalue()
            figure_paths = execution_locals.get('figure_paths', [])
            
            # TODO: traiter les figures pour les rendre accessibles via le web
            # Pour l'instant, nous retournons simplement les chemins (qui ne seront pas accessibles)
            
            result = {
                'output': output,
                'errors': errors,
                'figures': figure_paths  # Dans une implémentation réelle, ces chemins devraient être accessibles
            }
            
            return json.dumps(result)
            
        except Exception as e:
            return json.dumps({
                'output': '',
                'errors': f"Erreur d'exécution du code: {str(e)}",
                'figures': []
            })

@login_required
def get_conversations(request):
    """
    Récupère la liste des conversations de l'utilisateur.
    """
    conversations = Conversation.objects.filter(user=request.user)
    conversations_data = []
    
    for conversation in conversations:
        conversations_data.append({
            'id': conversation.id,
            'title': conversation.title,
            'created_at': conversation.created_at.isoformat(),
            'updated_at': conversation.updated_at.isoformat(),
        })
    
    return JsonResponse({
        'success': True,
        'conversations': conversations_data
    })

@login_required
def get_conversation_messages(request, conversation_id):
    """
    Récupère les messages d'une conversation spécifique.
    """
    try:
        conversation = Conversation.objects.get(id=conversation_id, user=request.user)
    except Conversation.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Conversation introuvable'
        }, status=404)
    
    messages = conversation.messages.all()
    messages_data = []
    
    for message in messages:
        message_data = {
            'id': message.id,
            'role': message.role,
            'content': message.content,
            'created_at': message.created_at.isoformat(),
        }
        
        if message.generated_code:
            message_data['generated_code'] = message.generated_code
        
        if message.execution_result:
            message_data['execution_result'] = message.execution_result
        
        if message.csv_file:
            message_data['csv_file'] = message.csv_file.url
        
        messages_data.append(message_data)
    
    return JsonResponse({
        'success': True,
        'conversation': {
            'id': conversation.id,
            'title': conversation.title,
            'created_at': conversation.created_at.isoformat(),
            'updated_at': conversation.updated_at.isoformat(),
            'messages': messages_data
        }
    })

@csrf_exempt
@login_required
def create_conversation(request):
    """
    Crée une nouvelle conversation.
    """
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Seules les requêtes POST sont acceptées'}, status=405)
    
    try:
        data = json.loads(request.body)
        title = data.get('title', 'Nouvelle conversation')
        
        conversation = Conversation.objects.create(
            user=request.user,
            title=title
        )
        
        return JsonResponse({
            'success': True,
            'conversation': {
                'id': conversation.id,
                'title': conversation.title,
                'created_at': conversation.created_at.isoformat(),
                'updated_at': conversation.updated_at.isoformat()
            }
        })
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@csrf_exempt
@login_required
def delete_conversation(request, conversation_id):
    """
    Supprime une conversation.
    """
    if request.method != 'DELETE':
        return JsonResponse({'success': False, 'error': 'Seules les requêtes DELETE sont acceptées'}, status=405)
    
    try:
        conversation = Conversation.objects.get(id=conversation_id, user=request.user)
        conversation.delete()
        
        return JsonResponse({
            'success': True,
            'message': 'Conversation supprimée avec succès'
        })
    
    except Conversation.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Conversation introuvable'
        }, status=404)
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)