from django.urls import path
from .views import web_lista_noticias
#
urlpatterns = [
    path('', web_lista_noticias, name='web_lista_noticias'),
    path('page=<int:page>/', web_lista_noticias, name='web_lista_noticias'),
]

