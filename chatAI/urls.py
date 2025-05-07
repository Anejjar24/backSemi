# chatAI/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('generate/', views.generate_code, name='llama'),  # Updated to match React's fetch URL
    path('generate-code/', views.generate_code, name='generate_code'),


    # Nouvelles URLs pour la gestion des conversations
    path('conversations/', views.get_conversations, name='get_conversations'),
    path('conversations/create/', views.create_conversation, name='create_conversation'),
    path('conversations/<int:conversation_id>/', views.get_conversation_messages, name='get_conversation_messages'),
    path('conversations/<int:conversation_id>/delete/', views.delete_conversation, name='delete_conversation'),
]
