from django.db import models

# Create your models here.

class News(models.Model):
    url = models.CharField(name="url", 
                           primary_key=True,
                           max_length=255, 
                           default="")
    media_url = models.CharField(name="mediaUrl",
                                 max_length=255, 
                                 default="")
    creation_date = models.DateField(name="creationDate", 
                                     max_length=32,
                                     default="",
                                     blank=False)
    insert_date = models.DateField(name="insertDate", 
                                   max_length=32,
                                   default="",
                                   blank=False)
    update_date = models.DateField(name="updateDate", 
                                  max_length=32,
                                  default="",
                                  blank=True)
    title = models.CharField(name="title", 
                             max_length=255, 
                             default="",
                             blank=False)
    description = models.CharField(name="description", 
                                   max_length=512, 
                                   default="",
                                   blank=True)
    article_body = models.TextField(name="articleBody", 
                                    default="",
                                    blank=True)
    main_topic = models.TextField(name="mainTopic", 
                                  default="",
                                  blank=True)
    other_topic = models.TextField(name="otherTopic", 
                                   default="",
                                     blank=True)
    image_url = models.CharField(name="imageUrl", 
                                 max_length=255, 
                                 default="",
                                     blank=True)
    country = models.CharField(name="country", 
                               max_length=32, 
                               default="",
                                     blank=True)
    n_tokens = models.SmallIntegerField(name="nTokens",  
                                        default=-1,
                                     blank=True)
    
    class Meta:
        # Change the table name to "new_table_name"
        db_table = 'news'
