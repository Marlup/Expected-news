from django.shortcuts import render#, get_object_or_404
from .models import News
from .server_variables import *
#from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage

# Create your views here.
def view_main_feed(request, 
                   n_rows=N_SHOWN_FEED_ROWS):
    query_set = News.objects.values("url",
                                    "creationDate",
                                    "insertDate", 
                                    "title",
                                    "description", 
                                    "articleBody", 
                                    "imageUrl",
                                    "score",
                                    "preprocessed"
                                    )
    
    query_set = query_set.filter(preprocessed=1)
    # "-column", means "order by column descending order"
    rows = query_set.order_by("-creationDate", "-score").all()[:n_rows]
    data_context = {
        "rows": rows
    }
    return render(request, 
                  './web/main-feed.html', 
                  data_context)
