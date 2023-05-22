from django.urls import path
from .views import web_news_list, web_extract_fields
#
urlpatterns = [
    path('', web_news_list, name='web_news_list'),
    path('page=<int:page>/', web_news_list, name='web_news_list'),
    path('between dates/from=<str:from_date>-to=<str:to_date>/page=<int:page>/', 
         web_extract_fields, 
         name='web_extract_fields'),
]

