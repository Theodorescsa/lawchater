
from django.urls import path

from . import views

urlpatterns = [
    path('chat/', views.chat),
    path('health/', views.health),
    path('ingest/', views.ingest),
    path('debug/', views.debug_vectorstore, name='debug'),
]
