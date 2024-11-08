from django.db import models
from django.utils import timezone

# Create your models here.

class News(models.Model):
    url = models.CharField(
        name="url", 
        primary_key=True,
        max_length=255, 
        default=""
        )
    media_url = models.CharField(
        name="mediaUrl",
        max_length=255, 
        default=""
        )
    creation_date = models.DateTimeField(
        name="creationDate", 
        max_length=32,
        default="",
        blank=True
        )
    insert_date = models.DateTimeField(
        name="insertDate", 
        max_length=32,
        default="",
        blank=True
        )
    update_date = models.DateTimeField(
        name="updateDate", 
        max_length=32,
        default="",
        blank=True
        )
    title = models.CharField(
        name="title", 
        max_length=255, 
        default="",
        blank=False
        )
    description = models.CharField(
        name="description", 
        max_length=512, 
        default="",
        blank=True
        )
    article_body = models.TextField(
        name="articleBody", 
        default="",
        blank=True
        )
    main_topic = models.TextField(
        name="mainTopic", 
        default="",
        blank=True
        )
    other_topic = models.TextField(
        name="otherTopic", 
        default="",
        blank=True
        )
    image_url = models.CharField(
        name="imageUrl", 
        max_length=255, 
        default="",
        blank=True
        )
    country = models.CharField(
        name="country", 
        max_length=32, 
        default="",
        blank=True
        )
    n_tokens = models.SmallIntegerField(
        name="nTokens",  
        default=-1,
        blank=True
        )
    score = models.FloatField(
        name="score",
        default=0.0,
        blank=False
        )
    preprocessed = models.BooleanField(
        name="preprocessed",  
        default=False,
        blank=0
        )
    class Meta:
        # Change the table name to "news"
        db_table = 'news'

    class Garbage(models.Model):
        url = models.CharField(
            name="url", 
            max_length=255, 
            default=""
            )
        media_url = models.CharField(
            name="mediaUrl",
            max_length=255, 
            default=""
            )
        insert_datetime = models.DateTimeField(
            name="insertDatetime", 
            max_length=32,
            default=timezone.now,
            blank=False
            )
        status_code = models.CharField(
            name="statusCode", 
            max_length=5,
            default="",
            blank=False
            )
        class Meta:
            # Change the table name to "garbage"
            db_table = 'garbage'
            constraints = [
                models.UniqueConstraint(
                    fields=['url', 
                            'mediaUrl'], 
                    name='unique_news_url_combination'
                    )
                    ]
            