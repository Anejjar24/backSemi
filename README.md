# Guide de démarrage du Backend PFA

Ce guide vous aide à configurer et lancer le backend du projet PFA en local.

---

## 🔧 Étapes à suivre

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
### 4. Installer les bibliothèques nécessaires
Toujours dans l'environnement virtuel, installez les dépendances suivantes :
```bash
cd BackEndPFA
pip install djangorestframework
pip install django-cors-headers
pip install python-dotenv
```

### 5. Lancer le serveur Django
```bash
cd BackEndPFA
python manage.py runserver
```
