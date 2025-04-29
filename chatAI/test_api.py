# test_api.py
# Script pour tester la connectivité avec l'API Django et Ollama

import requests
import json

def test_django_api():
    """Teste la connectivité avec l'API Django"""
    print("Testant la connexion à l'API Django...")
    
    try:
        # Remplacez l'URL par celle de votre API Django
        url = "http://localhost:8000/api/generate-code/"
        
        # Données de test simples
        data = {"query": "Affiche 'Hello World'"}
        
        # Création d'un FormData simple
        files = {}  # Pas de fichier pour ce test
        form_data = {"data": json.dumps(data)}
        
        # Envoi de la requête
        response = requests.post(url, data=form_data, files=files)
        
        # Affichage des résultats
        print(f"Status code: {response.status_code}")
        print(f"Headers: {response.headers}")
        
        if response.status_code == 200:
            print("Réponse OK!")
            try:
                print(f"Contenu JSON: {json.dumps(response.json(), indent=2)}")
            except:
                print(f"Contenu non-JSON: {response.text[:200]}...")
        else:
            print(f"Erreur: {response.text[:200]}...")
            
    except Exception as e:
        print(f"Erreur de connexion: {str(e)}")

def test_ollama_api():
    """Teste la connectivité avec l'API Ollama"""
    print("\nTestant la connexion à Ollama...")
    
    try:
        url = "http://localhost:11434/api/tags"
        response = requests.get(url)
        
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            print("Ollama est accessible!")
            try:
                models = response.json()
                print("Modèles disponibles:")
                for model in models.get("models", []):
                    print(f" - {model.get('name')}")
            except:
                print(f"Réponse non-JSON: {response.text[:200]}...")
        else:
            print(f"Erreur: {response.text[:200]}...")
    except Exception as e:
        print(f"Erreur de connexion à Ollama: {str(e)}")
        print("Assurez-vous qu'Ollama est en cours d'exécution.")

if __name__ == "__main__":
    test_django_api()
    test_ollama_api()