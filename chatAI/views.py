"""
This is a complete fixed version of the views.py file with all f-string backslash issues resolved.
For all f-strings that might contain Windows paths or backslashes, we've replaced them with forward slashes
or used string concatenation to avoid the f-string expression backslash errors.
"""

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
##hadu tzadu
from django.shortcuts import get_object_or_404
from django.http import JsonResponse
from .models import Conversation, Message
from django.contrib.auth.models import User
from django.views.decorators.csrf import csrf_exempt
import json
from django.utils import timezone
from rest_framework.views import APIView
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .models import Conversation
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator

from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from contextlib import redirect_stdout, redirect_stderr
import re
# Configuration du logging
logger = logging.getLogger(__name__)

# URL de l'API Ollama - avec gestion d'erreur de connexion
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Configuration du modèle
MODEL_NAME = "codellama"

@csrf_exempt
def generate_code(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Seules les requêtes POST sont acceptées'}, status=405)

    try:
        query = ""
        csv_content = ""
        csv_context = ""

        if 'data' in request.POST:
            try:
                data_json = json.loads(request.POST.get('data', '{}'))
                query = data_json.get('query', '')
            except json.JSONDecodeError:
                return JsonResponse({'success': False, 'error': 'Format JSON invalide dans les données'}, status=400)
        else:
            try:
                body = json.loads(request.body.decode('utf-8'))
                query = body.get('prompt', '') or body.get('query', '')
            except json.JSONDecodeError:
                return JsonResponse({'success': False, 'error': 'Format de requête invalide'}, status=400)

        if not query:
            return JsonResponse({'success': False, 'error': 'La requête doit contenir un prompt ou une query'}, status=400)

        if 'csv_file' in request.FILES:
            try:
                csv_file = request.FILES['csv_file']
                csv_df = pd.read_csv(csv_file, encoding='utf-8')
                csv_content = csv_df.to_csv(index=False)
                csv_context = f"CSV Columns: {', '.join(csv_df.columns)}\nPreview:\n{csv_df.head().to_string()}"
            except Exception as e:
                return JsonResponse({'success': False, 'error': f'Erreur CSV: {str(e)}'}, status=400)

        system_prompt = (
            "Tu es un assistant spécialisé dans la génération de code Python. "
            "Réponds uniquement avec du code Python exécutable et bien commenté. "
            "Ne fournis pas d'explications, uniquement du code Python."
        )

        full_prompt = f"""
{system_prompt}

{csv_context}

L'utilisateur demande:
{query}

Génère un script Python complet qui répond à cette demande.
```python
"""

        # Ajout d'un timeout et gestion des erreurs de connexion
        try:
            response = requests.post(
                OLLAMA_API_URL, 
                json={"model": MODEL_NAME, "prompt": full_prompt, "stream": False},
                timeout=120  # 60 secondes de timeout
            )
            
            if response.status_code != 200:
                logger.error(f"Erreur Ollama: status={response.status_code}, response={response.text}")
                return JsonResponse({
                    'success': False, 
                    'error': f'Erreur lors de la génération du code: {response.text}'
                }, status=500)
                
            response_data = response.json()
            generated_text = response_data.get('response', '')
            
        except requests.exceptions.ConnectionError:
            logger.error("Impossible de se connecter au serveur Ollama")
            return JsonResponse({
                'success': False, 
                'error': 'Impossible de se connecter au serveur Ollama. Vérifiez que Ollama est bien démarré.'
            }, status=503)
        except requests.exceptions.Timeout:
            return JsonResponse({
                'success': False, 
                'error': 'Le serveur Ollama a mis trop de temps à répondre'
            }, status=504)
        except Exception as e:
            logger.error(f"Erreur de requête Ollama: {str(e)}")
            return JsonResponse({
                'success': False, 
                'error': f'Erreur lors de la communication avec Ollama: {str(e)}'
            }, status=500)

        # Nettoyage du code généré
        # D'abord, on essaie de trouver le bloc de code entre les délimiteurs Python
        match = re.search(r"```python\s*(.*?)```", generated_text, re.DOTALL)
        if match:
            code_executable = match.group(1).strip()
        else:
            # Si pas de délimiteur Python spécifique, on cherche n'importe quel délimiteur code
            match = re.search(r"```(?:\w*)\s*(.*?)```", generated_text, re.DOTALL)
            if match:
                code_executable = match.group(1).strip()
            else:
                # Si toujours pas de délimiteur, on prend tout le texte
                code_executable = generated_text.strip()

        # On s'assure que le code ne contient pas de délimiteurs
        code_executable = code_executable.replace("```python", "").replace("```", "")
        
        # Pour l'affichage, on garde le format avec les délimiteurs s'ils existent
        code_affichage = f"```python\n{code_executable}\n```"

        return JsonResponse({
            'success': True,
            'code': code_affichage,
            'code_to_run': code_executable
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return JsonResponse({'success': False, 'error': f'Erreur inattendue: {str(e)}'}, status=500)


@csrf_exempt
def execute_code(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'Seules les requêtes POST sont acceptées'}, status=405)

    try:
        code = ""
        if 'data' in request.POST:
            try:
                data_json = json.loads(request.POST.get('data', '{}'))
                code = data_json.get('code', '')
            except json.JSONDecodeError:
                return JsonResponse({'success': False, 'error': 'Format JSON invalide'}, status=400)
        else:
            try:
                body = json.loads(request.body.decode('utf-8'))
                code = body.get('code', '')
            except json.JSONDecodeError:
                return JsonResponse({'success': False, 'error': 'Format de requête invalide'}, status=400)

        if not code:
            return JsonResponse({'success': False, 'error': 'Aucun code fourni'}, status=400)

        # Suppression de délimiteurs code si présents
        code = re.sub(r'^```(?:python)?\s*', '', code)
        code = re.sub(r'```$', '', code)

        csv_content = None
        if 'csv_file' in request.FILES:
            try:
                csv_file = request.FILES['csv_file']
                csv_content = csv_file.read().decode('utf-8')
            except Exception as e:
                return JsonResponse({'success': False, 'error': f'Erreur CSV: {str(e)}'}, status=400)

        result = execute_code_with_csv(code, csv_content)
        return JsonResponse({'success': True, 'result': result})

    except Exception as e:
        logger.error(traceback.format_exc())
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


def execute_code_with_csv(code, csv_content=None):
    execution_locals = {}
    execution_globals = {
        'pd': pd,
        'np': __import__('numpy'),
        'plt': __import__('matplotlib.pyplot'),
        'io': io,
        'StringIO': io.StringIO
    }

    output_capture = io.StringIO()
    error_capture = io.StringIO()

    setup_code = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
import statistics
import scipy
import sklearn
import json
import datetime
import re
import os
import sys
from io import StringIO
import warnings

warnings.filterwarnings('ignore')
"""

    if csv_content:
        setup_code += f"""
csv_data = '''{csv_content}'''
df = pd.read_csv(StringIO(csv_data))
"""

    try:
        # Exécuter le code de configuration
        exec(setup_code, execution_globals, execution_locals)
        
        # Exécuter le code principal avec redirection de stdout/stderr
        with redirect_stdout(output_capture), redirect_stderr(error_capture):
            exec(code, execution_globals, execution_locals)

        return json.dumps({
            'output': output_capture.getvalue(),
            'errors': error_capture.getvalue(),
            'figures': []  # Remplacer par le code de sauvegarde de figures si nécessaire
        })
    except Exception as e:
        # Capturer l'erreur complète avec traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return json.dumps({
            'output': '',
            'errors': error_msg,
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