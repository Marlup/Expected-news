from django.shortcuts import render#, get_object_or_404
from .models import News
from .server_variables import *
from django.http import JsonResponse
#from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage

# Auxiliar queries
def _base_query():
     return News.objects.values("url",
                                "creationDate",
                                "insertDate", 
                                "title",
                                "description", 
                                "articleBody",
                                "imageUrl",
                                "score",
                                "preprocessed"
                                )
def _end_query(query, 
               offset=0,
               n_rows=N_SHOWN_FEED_ROWS
               ):
     # "-column", means "order by column descending order"
     rows = query.filter(preprocessed=True) \
                 .order_by("-creationDate", 
                           "-score") \
                 .all()[offset:offset + n_rows]
     return rows

# filter() note: https://chat.openai.com/c/946fa933-8cd3-4327-bb93-95db6a31fc13
def _view_feed(request,
               path,
               query):
        data_context = {"rows": _end_query(query)}
        return render(request, 
                      path, 
                      data_context)

## Views ##
# Main feed view
def view_feed(request):
    return _view_feed(request, 
                      PATH_HTML_FEED,
                      _base_query())

# Feed view by topic selected by user
def view_feed_by_topic(request, 
                       topic):
    return _view_feed(request, 
                      PATH_HTML_FEED,
                      _base_query().filter(mainTopic__icontains=topic))

## Data loading ##
# Get more data when the bottom of page is reached
def get_more_news(request,
                  offset
                  ):
        #offset = int(request.GET.get("offset", 0))
    query = _base_query()
    rows = _end_query(query, 
                      offset)
    print(rows, offset)
    return JsonResponse({"rows": list(rows)})
def get_more_news_by_topic(request,
                           topic,
                           offset
                           ):
    query = _base_query().filter(mainTopic__icontains=topic)
    rows = _end_query(query, 
                      offset)
    
    return JsonResponse({"rows": list(rows)}, 
                        content_type="application/json"
                        )
