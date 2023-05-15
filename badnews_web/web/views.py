from django.shortcuts import render, get_object_or_404
from .models import Noticia
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage

# Create your views here.
def web_lista_noticias(request, page_num=0):
    print("\nPage number:", request)
    print("Page number:", page_num, "\n")
    noticias = Noticia.objects.all().order_by('-fecha_creacion')
    paginator = Paginator(noticias, 5)  # Display 4 news articles per page
    
    pagina = request.GET.get('page')
    #print("Has previous:", pagina.has_previous(), "\n")
    #print("Has next:", pagina.has_next(), "\n")
    try:
        noticias = paginator.page(pagina)
    except PageNotAnInteger:
    # If page is not an integer, deliver first page.
        noticias = paginator.page(1)
    except EmptyPage:
    # If page is out of range, deliver last page of results.
        noticias = paginator.page(paginator.num_pages)
        
    context = {'noticias': noticias, 
               'pagina': noticias,
              }
    return render(request, 'web/web_lista_noticias.html', context)                                                        