# chatAI/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('generate/', views.generate_code, name='llama'),  # Updated to match React's fetch URL
    path('generate-code/', views.generate_code, name='generate_code'),
    #path('conversations/create/', views.add_conversation_with_messages, name='test_add_conversation'), #tzadt hadiiii
    path('conversations/create/', views.add_user_conversation, name='create-user-conversation'),
    path('conversations/', views.get_user_conversations, name='get_user_conversations'),
    path('conversations/<int:pk>/delete/', views.delete_user_conversation, name='delete-user-conversation'),
    path('conversations/<int:pk>/', views.open_user_conversation, name='open-user-conversation'),
    path('execute-code/', views.execute_code, name='execute-code'),
]
