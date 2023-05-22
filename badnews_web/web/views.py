from django.shortcuts import render, get_object_or_404
from .models import New
from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage

# Create your views here.
def web_news_list(request, page_num=0, on_filter=False, extracted_news=None):
    print("\nPage number:", request)
    print("Page number:", page_num, "\n")
    if not on_filter:
        news = New.objects.all().order_by('creationDate')
    else:
        news = extracted_news
    paginator = Paginator(news, 5)  # Display 4 news articles per page
    
    pagina = request.GET.get('page')
    try:
        news = paginator.page(pagina)
    except PageNotAnInteger:
    # If page is not an integer, deliver first page.
        news = paginator.page(1)
    except EmptyPage:
    # If page is out of range, deliver last page of results.
        news = paginator.page(paginator.num_pages)
        
    context = {'news': news, 
               'pagina': news,
              }
    return render(request, 'web/web_news_list.html', context)

def web_extract_fields(request):
    # Get the input "from date" and "to date" from the request
    from_date = request.GET.get('from_date')
    to_date = request.GET.get('to_date')

    # Perform the query using the date range
    records = New.objects.filter(creationDate__range=[from_date, to_date])

    # Pass the records to the template or process them as needed
    context = {'records': records}
    web_news_list(request, on_filter=True, extract_news=records)