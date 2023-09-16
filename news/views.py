from django.shortcuts import render#, get_object_or_404
from .models import News
from .server_variables import *
#from django.core.paginator import Paginator, PageNotAnInteger, EmptyPage

PATH_HTML_MAIN_FEED = './web/main-feed.html'
PATH_HTML_FEED_TOPIC = './web/feed-by-topic.html'

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
    return _view_main_feed(request, 
                           PATH_HTML_MAIN_FEED,
                           query_set, 
                           n_rows)

def view_main_feed_by_topic(request, 
                            topic,
                            n_rows=N_SHOWN_FEED_ROWS):
    query_set = News.objects.values("url",
                                    "creationDate",
                                    "insertDate", 
                                    "title",
                                    "description", 
                                    "articleBody", 
                                    "mainTopic",
                                    "imageUrl",
                                    "score",
                                    "preprocessed"
                                    ) \
                            .filter(mainTopic__icontains=topic)
    return _view_main_feed(request, 
                           PATH_HTML_FEED_TOPIC,
                           query_set, 
                           n_rows)

def _view_main_feed(request,
                    path,
                    pre_query, 
                    n_rows):
        # "-column", means "order by column descending order"
        rows = pre_query.filter(preprocessed=True) \
                        .order_by("-creationDate", 
                                  "-score") \
                        .all()[:n_rows]
        data_context = {
             "rows": rows
             }
        return render(request, 
                      path, 
                      data_context)


def view_main_feed_by_topic2(request, 
                            topic,
                            n_rows=N_SHOWN_FEED_ROWS):
    rows = News.objects.filter(preprocessed=True, 
                               mainTopic__icontains=topic) \
                       .values("url",
                               "creationDate",
                               "insertDate", 
                               "title",
                               "description", 
                               "articleBody", 
                               "mainTopic",
                               "imageUrl",
                               "score",
                               "preprocessed"
                               ) \
                       .order_by("-creationDate", 
                                 "-score") \
                       .all()[:n_rows]
    data_context = {
          "rows": rows
          }
    return render(request, 
                    PATH_HTML_FEED_TOPIC, 
                    data_context)