from django.db import models
from django.contrib.auth.models import User

class DataAnalysisRequest(models.Model):
    query = models.TextField(help_text="Requête en langage naturel")
    csv_file = models.FileField(upload_to='uploads/', null=True, blank=True)
    generated_code = models.TextField(null=True, blank=True)
    execution_result = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Analyse #{self.id}: {self.query[:50]}..."

class Conversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations')
    title = models.CharField(max_length=255, default="Nouvelle conversation")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.title} ({self.user.username})"

    class Meta:
        ordering = ['-updated_at']

class Message(models.Model):
    ROLE_CHOICES = (
        ('user', 'Utilisateur'),
        ('assistant', 'Assistant'),
    )
    
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    csv_file = models.FileField(upload_to='conversations/', null=True, blank=True)
    generated_code = models.TextField(null=True, blank=True)
    execution_result = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.role} message in {self.conversation.title}"

    class Meta:
        ordering = ['created_at']