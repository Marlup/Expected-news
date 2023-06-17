from django.urls import path
from .views import web_news_list
from django.conf import settings
from django.conf.urls.static import static 

urlpatterns = [
    path('', web_news_list, name='web_news_list'),
    path('page=<int:page>/', web_news_list, name='web_news_list'),
    ] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

