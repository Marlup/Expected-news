from django.db import models

# Create your models here.
class Noticia(models.Model):
    indice = models.CharField(max_length=255)
    titulo = models.CharField(max_length=255)
    fuente = models.CharField(max_length=255)
    fecha_creacion = models.DateTimeField(auto_now_add=True)
    subtitulo = models.CharField(max_length=255)