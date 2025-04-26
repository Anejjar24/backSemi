# chatAI/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('api/chatAPI', views.generate_code, name='generate_code'),  # Updated to match React's fetch URL
    path('generate/', views.generate_code, name='llama'),  # Updated to match React's fetch URL

]