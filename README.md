# Guide de démarrage du Backend PFA

Ce guide vous aide à configurer et lancer le backend du projet PFA en local.

---
## 🎬 Démonstartion video
https://github.com/user-attachments/assets/a285767f-d1a5-4b21-8899-1752601bc0c3

## 🔧 Étapes à suivre


 Avant TOUT  : Concernant la version de python, vous devez disposer au moins de la version 3.8.

### 1. Cloner le dépôt Git

Dans un dossier de votre choix, ouvrez un terminal et exécutez :

```bash
git clone https://github.com/Anejjar24/BackEndPFA.git
```

### 2. Créer et activer un environnement virtuel
# Créer l'environnement virtuel
```bash
virtualenv test
```

# Activer l'environnement (sous Windows)

```bash
test\Scripts\activate
```
# Installer django
Dans l'environnement virtuel activé, installez Django :
```bash
pip install django
```

# Créer un nouveau projet Django
Dans votre terminal :
```bash
django-admin startproject PFA
```

Cela va créer un dossier PFA contenant les fichiers de base du projet Django.
### 3. Copier les fichiers du dépôt cloné
Déplacez-vous dans le dossier du projet :
```bash
cd PFA
```

Copiez le contenu du dossier BackEndPFA (cloné à l'étape 1) dans le dossier PFA de l'application (celui créé juste avant).

Remplacez les fichiers existants si nécessaire.
### 4. Modifier les informations d'email dans settings.py
Dans le fichier settings.py, repérez la configuration email :

```bash
EMAIL_HOST_USER = 'votre email'
EMAIL_HOST_PASSWORD = 'votre google app password'
```
Remplacez ces champs par :

votre adresse email Gmail réelle,

et votre mot de passe d'application Google (pas votre mot de passe normal).

💡 Astuce : vous devez créer un mot de passe d’application depuis votre compte Google (sécurité > mots de passe d’application).
### 5 Installer les bibliothèques nécessaires
Toujours dans l'environnement virtuel, installez les dépendances suivantes :
```bash
pip install djangorestframework
pip install django-cors-headers
pip install python-dotenv
```

### 6. Lancer le serveur Django
```bash
python manage.py migrate
```
pour lancer projet:
```bash
python manage.py runserver
```
