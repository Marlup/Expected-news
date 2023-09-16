from django.urls import path
from .views import view_main_feed, view_main_feed_by_topic
from django.conf import settings
from django.conf.urls.static import static 

urlpatterns = [
    path('', 
         view_main_feed, 
         name='news-feed'),
    path('/<str:topic>/', 
         view_main_feed_by_topic, 
         name='news-feed-topic'),
    ] + static(settings.STATIC_URL, 
               document_root=settings.STATIC_ROOT)
