from django.urls import path
from .views import view_main_feed
from django.conf import settings
from django.conf.urls.static import static 

urlpatterns = [
    path('', 
         view_main_feed, 
         name='news_list'),
    path('feed', 
         view_main_feed, 
         name='news_list'),
    path('page=<int:page>/', 
         view_main_feed, 
         name='news_list'),
    ] + static(settings.STATIC_URL, 
               document_root=settings.STATIC_ROOT)
