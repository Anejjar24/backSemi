import json
import os
import tempfile
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import requests
import pandas as pd
import io
import traceback
import logging
import time
from django.conf import settings
from django.shortcuts import get_object_or_404
from .models import Conversation, Message
from django.contrib.auth.models import User
from django.utils import timezone
from rest_framework.views import APIView
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
import base64
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from builtins import print as builtin_print
import re

logger = logging.getLogger(__name__)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "codellama"

@csrf_exempt
def generate_code(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST allowed'}, status=405)

    try:
        query = ""
        csv_content = ""
        csv_context = ""

        if 'data' in request.POST:
            data_json = json.loads(request.POST.get('data', '{}'))
            query = data_json.get('query', '')
        else:
            body = json.loads(request.body.decode('utf-8'))
            query = body.get('prompt', '') or body.get('query', '')

        if not query:
            return JsonResponse({'success': False, 'error': 'Prompt is required'}, status=400)

        if 'csv_file' in request.FILES:
            csv_file = request.FILES['csv_file']
            csv_df = pd.read_csv(csv_file, encoding='utf-8')
            csv_content = csv_df.to_csv(index=False)
            csv_context = f"CSV Columns: {', '.join(csv_df.columns)}\nPreview:\n{csv_df.head().to_string()}"

        system_prompt = (
            "Tu es un assistant expert Python. "
            "Réponds uniquement avec du code Python exécutable, sans aucune explication avant ou après. "
            "Utilise les noms de colonnes fournis dans fichier csv, ne les invente pas. "
            "Encadre tout le code dans un bloc markdown comme ceci : ```python ... ``` "
            "Le code doit contenir une fonction main() et le bloc if __name__ == '__main__'. "
            "N’inclus aucun texte explicatif ni commentaire après le code."
           

        )

        full_prompt = f"""{system_prompt}

{csv_context}

L'utilisateur demande :
{query}

Génère un script Python complet :
```python
"""

        response = requests.post(
            OLLAMA_API_URL,
            json={"model": MODEL_NAME, "prompt": full_prompt, "stream": False},
            timeout=3600
        )
        if response.status_code != 200:
            return JsonResponse({'success': False, 'error': response.text}, status=500)

        response_data = response.json()
        generated_text = response_data.get('response', '')
        code_affichage = generated_text.strip()

        match = re.search(r"```python\s*(.*?)```", generated_text, re.DOTALL)
        if not match:
            match = re.search(r"```(?:\w*)?\s*(.*?)```", generated_text, re.DOTALL)

        code_executable = match.group(1).strip() if match else generated_text.strip()
        code_executable = re.split(
            r'\n\s*(In this script|This script|Note that|Note:|Explanation:|Ce script|Notez que|Remarquez que|Remarque :|Remarque:|Explication:|Résultat |Output:)',
            code_executable,
            flags=re.IGNORECASE
        )[0]
        code_executable = code_executable.replace("```python", "").replace("```", "").strip()
        code_executable = code_executable.replace("**name**", "__name__")

        main_index = code_executable.rfind("if __name__")
        if main_index != -1:
            code_executable = code_executable[:main_index]
            main_block = re.search(r"(if __name__.*)", generated_text, re.DOTALL)
            if main_block:
                code_executable += "\n" + main_block.group(1)

        return JsonResponse({
            'success': True,
            'code': code_affichage,
            'code_to_run': code_executable
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@csrf_exempt
def execute_code(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Only POST allowed'}, status=405)

    try:
        code = ""
        if 'data' in request.POST:
            data_json = json.loads(request.POST.get('data', '{}'))
            code = data_json.get('code', '')
        else:
            body = json.loads(request.body.decode('utf-8'))
            code = body.get('code', '')

        if not code:
            return JsonResponse({'success': False, 'error': 'No code provided'}, status=400)

        clean_code = re.sub(r'^```(?:python)?\s*', '', code.strip())
        clean_code = re.sub(r'\s*```$', '', clean_code).strip()
        clean_code = clean_code.replace("**name**", "__name__")

        csv_filename = None
        csv_content = None
        if 'csv_file' in request.FILES:
            uploaded_file = request.FILES['csv_file']
            csv_filename = uploaded_file.name  # ✅ nom réel du fichier (ex: sales_data.csv)
            csv_content = uploaded_file.read().decode('utf-8')

        result = execute_code_with_csv(clean_code, csv_content, csv_filename)
        return JsonResponse({'success': True, 'result': result})

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


def execute_code_with_csv(code, csv_content=None, csv_filename=None):
    """
    Exécute le code Python généré avec un fichier CSV optionnel et capture la sortie et les figures
    Version améliorée qui gère mieux les cas où le code contient du texte explicatif
    """
    import sys
    import matplotlib
    matplotlib.use('Agg')  # Important: définir le backend avant d'importer pyplot
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import io
    import builtins
    import base64
    import csv
    
    logger = logging.getLogger(__name__)
    
    # ========== PHASE 1: NETTOYAGE AVANCÉ DU CODE ==========
    # Nettoyer les balises markdown
    clean_code = re.sub(r"```[\w]*", "", code) 
    clean_code = re.sub(r"```", "", clean_code)
    
    # Remplacer **name** par __name__
    clean_code = clean_code.replace("**name**", "__name__")
    
    # IMPORTANT: Extraire uniquement le bloc de code Python valide
    # Méthode 1: Utiliser la première phrase explicative comme séparateur
    patterns = [
        r'\n\s*(?:Notez que|Note that|Remarque|Explanation:|Explication:|This script|Ce script|Output:)',
        r'\n\s*(?:Pour utiliser|To use)',
        r'\n\s*(?:Dans ce code|In this code)'
    ]
    
    for pattern in patterns:
        parts = re.split(pattern, clean_code, flags=re.IGNORECASE)
        if len(parts) > 1:
            clean_code = parts[0]
            break
    
    # Méthode 2: Détecter la fin du code valide (recherche de la dernière instruction Python valide)
    lines = clean_code.splitlines()
    valid_lines = []
    for line in lines:
        stripped = line.strip()
        # Si la ligne ressemble à du texte explicatif plutôt qu'à du code Python, arrêtez
        if stripped and not stripped.startswith('#') and not re.match(r'^[a-zA-Z0-9_\s\(\)\[\]\{\}:=<>+\-*/,.\'\"]+$', stripped):
            # Vérifie si ça ressemble plus à une explication qu'à du code
            if len(stripped.split()) > 5 and not any(x in stripped for x in ['(', ')', '=', '+', '-', '*', '/', '[', ']']):
                break
        valid_lines.append(line)
    
    # Nettoyage final: supprimer les lignes vides à la fin
    while valid_lines and not valid_lines[-1].strip():
        valid_lines.pop()
    
    clean_code = "\n".join(valid_lines).strip()
    
    # ========== PHASE 2: EXÉCUTION ==========
    execution_globals = {
        'pd': pd, 'np': np, 'plt': plt, 'sns': sns,
        'io': io, 'sys': sys, 'StringIO': StringIO,
        'base64': base64, 'builtin_print': builtins.print,
        '__name__': '__main__'
    }
    
    execution_locals = {}
    output_capture = StringIO()
    error_capture = StringIO()
    
    setup_code = """
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Important: configurer Matplotlib pour un environnement non-interactif
import matplotlib.pyplot as plt
import seaborn as sns
import random, math, statistics
import io, base64, sys
from io import StringIO
from builtins import print as builtin_print

plt.switch_backend('Agg')

# Fonction pour capturer toutes les figures et les convertir en base64
def get_figure_base64():
    figures = []
    for i in plt.get_fignums():
        fig = plt.figure(i)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
        figures.append(img_str)
    return figures

# Remplacer plt.show() par une fonction personnalisée qui sauvegarde la figure
original_show = plt.show
def custom_show(*args, **kwargs):
    # Ne rien faire, juste conserver la figure en mémoire
    pass
plt.show = custom_show
"""
    
    # Configurer le fichier CSV s'il est fourni
    if csv_content:
        if not csv_filename:
            csv_filename = "uploaded.csv"
        setup_code += f"""
with open('{csv_filename}', 'w', encoding='utf-8') as f:
    f.write(\"\"\"{csv_content}\"\"\")
"""
    
    # Indenter le code pour la gestion d'erreurs
    indented_code = "\n".join("    " + line for line in clean_code.splitlines())
    
    # Préparation du code final à exécuter
    final_code = f"""
def custom_print(*args, **kwargs):
    builtin_print(*args, **kwargs)
    sys.stdout.flush()

print = custom_print

try:
{indented_code}

    # Exécuter la fonction main() si elle existe
    if "main" in locals() and callable(locals()["main"]):
        try:
            locals()["main"]()
        except Exception as e:
            print(f"Erreur lors de l'exécution de main(): {{str(e)}}")
            import traceback
            print(traceback.format_exc())
except Exception as e:
    print(f"Erreur d'exécution: {{str(e)}}")
    import traceback
    print(traceback.format_exc())

print("\\n--- Exécution terminée ---")
"""
    
    try:
        # Exécuter le code de configuration
        exec(setup_code, execution_globals, execution_locals)
        
        # Exécuter le code de l'utilisateur
        with redirect_stdout(output_capture), redirect_stderr(error_capture):
            exec(final_code, execution_globals, execution_locals)
        
        output_text = output_capture.getvalue()
        error_text = error_capture.getvalue()
        
        # Capture des figures après l'exécution du code

        figures = []
        try:
            # Vérifier si des figures ont été créées
            if plt.get_fignums():
                for i in plt.get_fignums():
                    fig = plt.figure(i)
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    buf.seek(0)
                    img_str = base64.b64encode(buf.getvalue()).decode('utf-8')
                    figures.append(f"data:image/png;base64,{img_str}")
                plt.close('all')  # Fermer toutes les figures pour libérer la mémoire
        except Exception as e:
            error_text += f"\nErreur lors de la capture des graphiques: {str(e)}\n{traceback.format_exc()}"            
    

        return json.dumps({
            'output': output_text,
            'errors': error_text,
            'figures': figures
        })
    except Exception as e:
        return json.dumps({
            'output': '',
            'errors': str(e) + "\n" + traceback.format_exc(),
            'figures': []
        })











@csrf_exempt
def test_add_conversation_with_messages(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)

            # Récupère le user avec id=2 (HARAT)
            try:
                user = User.objects.get(id=3)
            except User.DoesNotExist:
                return JsonResponse({'success': False, 'error': 'Utilisateur non trouvé'}, status=404)

            title = data.get('title', 'AZOUZ CONVERSATION')
            messages = data.get('messages', [])

            # Créer une nouvelle conversation
            conv = Conversation.objects.create(user=user, title=title, created_at=timezone.now())

            # Ajouter les messages
            for msg in messages:
                Message.objects.create(
                    conversation=conv,
                    sender=msg['sender'],
                    content=msg['content'],
                    timestamp=timezone.now()
                )

            return JsonResponse({
                'success': True,
                'conversation_id': conv.id,
                'messages_count': len(messages)
            })

        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'error': 'JSON invalide'}, status=400)
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)}, status=500)

    return JsonResponse({'success': False, 'error': 'Méthode non autorisée'}, status=405)


"""
@csrf_exempt
def get_user_conversations(request):
    if request.method == 'GET':
        if not request.user.is_authenticated:
            return JsonResponse({'success': False, 'error': 'Utilisateur non authentifié'}, status=401)

        conversations = Conversation.objects.filter(user=request.user).order_by('-updated_at')
        conv_list = [
            {
                'id': conv.id,
                'title': conv.title,
                'created_at': conv.created_at.isoformat(),
                'updated_at': conv.updated_at.isoformat() if conv.updated_at else conv.created_at.isoformat(),
            }
            for conv in conversations
        ]

        return JsonResponse({'success': True, 'conversations': conv_list})
    
    return JsonResponse({'success': False, 'error': 'Méthode non autorisée'}, status=405)
"""



@api_view(['GET'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def get_user_conversations(request):
    user = request.user
    conversations = Conversation.objects.filter(user=user).order_by('-updated_at')
    conv_list = [
        {
            'id': str(conv.id),
            'title': conv.title,
            'created_at': conv.created_at.isoformat(),
            'updated_at': conv.updated_at.isoformat() if conv.updated_at else conv.created_at.isoformat(),
        }
        for conv in conversations
    ]
    return JsonResponse({'success': True, 'conversations': conv_list})



@api_view(['DELETE'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def delete_user_conversation(request, pk):
    user = request.user
    conversation = get_object_or_404(Conversation, pk=pk)

    if conversation.user != user:
        return JsonResponse({
            'success': False,
            'error': 'Vous n\'êtes pas autorisé à supprimer cette conversation.'
        }, status=403)

    try:
        conversation.delete()
        return JsonResponse({
            'success': True,
            'message': 'Conversation supprimée avec succès.'
        }, status=200)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': f'Erreur lors de la suppression : {str(e)}'
        }, status=500)



        
@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def add_user_conversation(request):
    try:
        user = request.user  # Utilisateur récupéré automatiquement via le token
        data = request.data

        title = data.get('title', 'Nouvelle Conversation')
        messages = data.get('messages', [])

        # Créer la conversation pour l'utilisateur connecté
        conv = Conversation.objects.create(user=user, title=title, created_at=timezone.now())

        # Ajouter les messages s'il y en a
        for msg in messages:
            Message.objects.create(
                conversation=conv,
                sender=msg['sender'],
                content=msg['content'],
                timestamp=timezone.now()
            )

        return JsonResponse({
            'success': True,
            'conversation': {
                'id': conv.id,
                'title': conv.title,
                'created_at': conv.created_at.isoformat(),
                'updated_at': conv.updated_at.isoformat() if conv.updated_at else conv.created_at.isoformat()
            },
            'messages_count': len(messages)
        })

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)
    


@api_view(['GET'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def open_user_conversation(request, pk):
    try:
        user = request.user
        # On vérifie que la conversation appartient à cet utilisateur
        conversation = Conversation.objects.get(pk=pk, user=user)

        # Récupération des messages liés à la conversation
        messages = Message.objects.filter(conversation=conversation).order_by('timestamp')
        messages_data = [{
            'id': msg.id,
            'sender': msg.sender,
            'content': msg.content,
            'timestamp': msg.timestamp.isoformat()
        } for msg in messages]

        return JsonResponse({
            'success': True,
            'conversation': {
                'id': conversation.id,
                'title': conversation.title,
                'created_at': conversation.created_at.isoformat(),
                'updated_at': conversation.updated_at.isoformat() if conversation.updated_at else conversation.created_at.isoformat(),
                'messages': messages_data
            }
        })

    except Conversation.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Conversation introuvable ou non autorisée'}, status=404)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def add_message_to_conversation(request):
    try:
        data = request.data
        user = request.user

        conversation_id = data.get('conversation')
        content = data.get('content')
        sender = data.get('sender', 'user')  # Par défaut 'user'

        if not conversation_id or not content:
            return JsonResponse({'success': False, 'error': 'conversation et content sont requis'}, status=400)

        # Vérifie que la conversation existe et appartient à l'utilisateur
        conversation = Conversation.objects.get(id=conversation_id, user=user)

        # Crée le message
        msg = Message.objects.create(
            conversation=conversation,
            sender=sender,
            content=content,
            timestamp=timezone.now()
        )

        return JsonResponse({
            'success': True,
            'message': {
                'id': msg.id,
                'sender': msg.sender,
                'content': msg.content,
                'timestamp': msg.timestamp.isoformat()
            }
        })

    except Conversation.DoesNotExist:
        return JsonResponse({'success': False, 'error': 'Conversation introuvable ou non autorisée'}, status=404)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)
    



@api_view(['POST'])
@permission_classes([IsAuthenticated])
def update_conversation_title(request, conversation_id):
    try:
        conversation = Conversation.objects.get(id=conversation_id, user=request.user)
    except Conversation.DoesNotExist:
        return Response({'error': 'Conversation introuvable'}, status=status.HTTP_404_NOT_FOUND)

    new_title = request.data.get('title')
    if not new_title:
        return Response({'error': 'Titre requis'}, status=status.HTTP_400_BAD_REQUEST)

    conversation.title = new_title
    conversation.save()
    return Response({'success': True, 'title': conversation.title})