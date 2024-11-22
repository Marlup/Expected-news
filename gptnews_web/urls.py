"""
URL configuration for gptnews_web project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from news.views import (
    view_feed,
    view_feed_by_topic,
    get_more_news,
    get_more_news_by_topic
)

urlpatterns = [
    path('admin/', 
         admin.site.urls),
    path('', 
         view_feed), # Default page 
    path('<str:topic>/',
         view_feed_by_topic), # Defined list displaying,
    path('get_more_news/<int:offset>/', 
         get_more_news, 
         name='get-more-news'),
    path('get_more_news/<str:topic>&<int:offset>/', 
         get_more_news_by_topic, 
         name='get-more-news-by-topic')
]