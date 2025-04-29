# chatAI/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('generate/', views.generate_code, name='llama'),  # Updated to match React's fetch URL
    path('generate-code/', views.generate_code, name='generate_code'),
]
