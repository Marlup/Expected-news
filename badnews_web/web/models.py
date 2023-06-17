from django.db import models

class New(models.Model):
    title = models.CharField(max_length=255, default="empty title")
    subtitle = models.CharField(max_length=255, default="empty subtitle")
    article = models.TextField(default="empty article")
    source = models.CharField(max_length=255, default="empty source")
    apiSource = models.CharField(max_length=255, default="empty api")
    country = models.CharField(max_length=255, default="empty country")
    creationDate = models.DateField(default="0000-01-01")
    updateDate = models.DateField(default="1-1-1")
    author = models.CharField(max_length=255, default="empty author")
    image_url = models.CharField(max_length=255, default="empty image")