from django.shortcuts import render, get_object_or_404
from .models import New
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage

# Create your views here.
def web_news_list(request):
    news = New.objects.all()
    return render(request, 'web/fastnews django.html', {"news": news})

