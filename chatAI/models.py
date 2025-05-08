from django.db import models
from django.contrib.auth.models import User

# Create your models here.


class DataAnalysisRequest(models.Model):
    query = models.TextField(help_text="Requête en langage naturel")
    csv_file = models.FileField(upload_to='uploads/', null=True, blank=True)
    generated_code = models.TextField(null=True, blank=True)
    execution_result = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Analyse #{self.id}: {self.query[:50]}..."
    
#hadu li tzadu

class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)  # Référence vers l'utilisateur
    title = models.CharField(max_length=255)  # Titre de la conversation
    created_at = models.DateTimeField(auto_now_add=True)  # Date de création
    updated_at = models.DateTimeField(auto_now=True)  # Date de mise à jour automatique

    def __str__(self):
        return f"Conversation #{self.id} - {self.title} - {self.user.username}"


class Message(models.Model):
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')  # Référence vers la conversation
    sender = models.CharField(max_length=10, choices=[('user', 'User'), ('assistant', 'AI')])  # Expéditeur : user ou assistant
    content = models.TextField()  # Contenu du message
    timestamp = models.DateTimeField(auto_now_add=True)  # Horodatage

    def __str__(self):
        return f"{self.sender} - {self.content[:30]}..."  # Affichage des 30 premiers caractères du message