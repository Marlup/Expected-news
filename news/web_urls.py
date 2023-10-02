from django.urls import path
from .views import (
    view_feed, 
    view_feed_by_topic,
    get_more_news,
    get_more_news_by_topic
)
from django.conf import settings
from django.conf.urls.static import static 
#    path(url,
#         view,
#         name,
#         pattern_name
#         )
urlpatterns = [
     path('', 
          view_feed, 
          name='news-feed'),
     path('<str:topic>/', 
          view_feed_by_topic, 
          name='news-feed-topic'),
     path('get_more_news/<int:offset>/', 
          get_more_news, 
          name='get-more-news'),
     path('get_more_news/<str:topic>&<int:offset>/', 
          get_more_news_by_topic, 
          name='get-more-news-by-topic'),
    ] + static(settings.STATIC_URL, 
               document_root=settings.STATIC_ROOT)
