import json
import os
from datetime import datetime, timedelta, timezone
import requests
import openai
from bs4 import BeautifulSoup
import re
import pandas as pd
import sqlite3
import json
import os
import glob
import multiprocessing as mp
import os
from hashlib import sha256
import utilities as ut
from constants import *

# Classes
class ErrorReporter(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)
        with open("Errors log.txt", "w") as errors_file:
            errors_file.write(message)
# Database names
# Managers instantiation
file_manager = ut.FileManager()
#file_manager.add_files([
#    FILE_NAME_EXTRACTION_ERRORS
#    ])
# Dates and times
TODAY_LOCAL_DATETIME = datetime.now().replace(tzinfo=timezone.utc)
CURRENT_DATE, CURRENT_TIME = str(datetime.today()).split(" ")
# Data structures
ORDER_KEYS = ("url",
              "media_url",
              "creation_datetime",
              "update_datetime",
              "title",
              "description",
              "body", 
              "main_topic",
              "other_topic",
              "image_url",
              "country",
              "n_tokens",
              "score"
              )
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
}
## Dynamical global variables
if not BLOCK_API_CALL:
    openai.api_key = os.getenv("OPENAI_API_KEY")
#with open(FILE_NAME_PROMPT_ROLE_KEYS, "r") as file:
#    prompt_role = file.read()
with open(FILE_PATH_PROMPT_ROLE_SUMMARY, "r") as file:
    prompt_role_summary = file.read()
regex_url_has_query = re.compile("[?=&+%@#]{1}")
regex_too_short_url_end = re.compile(r"^[.a-zA-Z0-9]+(-[.a-zA-Z0-9]+){,2}$")
regex_url_startswith_https = re.compile("https?:[\/]{2}")
regex_title = re.compile("title|headline|titular|titulo")
regex_body = re.compile("articleBody")
regex_date_creation = re.compile("date.*[pP]ub.*")
regex_date_modified = re.compile("date.*[mM]od.*")
regex_tags = re.compile("tag[s]?|topic[s]?|tema[s]?|etiqueta[s]?|keyword[s]?")
regex_n_tokens = re.compile("Tokens: (\d+).*", flags=re.DOTALL)
regex_headline = re.compile("Headline: (.*).*", flags=re.DOTALL)
regex_topics = re.compile("Topics: (.*).*", flags=re.DOTALL)
regex_creation_datetime = re.compile("Creation DateTime: (.*).*", flags=re.DOTALL)
regex_update_datetime = re.compile("Update DateTime: (.*).*Body Summary", flags=re.DOTALL)
regex_only_summary = re.compile("Body Summary:(.*)", flags=re.DOTALL)
regex_published_time = re.compile(r"publish(?:ed)?_?(?:time|date)")
node_ignore = ["europa", 
               "africa",
               "asia",
               "oceania",
               "prensadigital"
              ]

## Decorators
# Decorator function to handle inputs and error logging
def error_logger(manager: ut.FileManager,
                 file_name: str
                 ):
    def decorator(target_func):
        def wrapper(*args, **kwargs):
            data, cache = target_func(*args, **kwargs)
            # If list of several status_codes'
            status_code_value = cache.get("status_code", "0")
            if isinstance(status_code_value, (list, tuple)) and status_code_value:
                file_name = str(os.getpid()) + "_" + file_name
                manager.write_on_file(file_name,
                                      status_code_value
                                      )
                return data
            # If dict of 1 status_code
            if isinstance(status_code_value, dict) and status_code_value != "0":
                file_name = str(os.getpid()) + "_" + file_name
                manager.write_on_file(file_name,
                                      [cache]
                                      )
            return data
        return wrapper
    return decorator

def garbage_logger():
    def decorator(target_func):
        def wrapper(*args, **kwargs):
            others, new_garbage = target_func(*args, **kwargs)
            if not new_garbage.get("has_garbage", True):
                lock = mp.Lock()
                with lock:
                    ut.insert_garbage(new_garbage["data"])
                return others
            return others
        return wrapper
    return decorator

## Functions
def find_body_from_json(html: BeautifulSoup, 
                        data: dict, 
                        url: str,
                        media: str
                        ) -> dict:
    data["body"] = ""
    data["n_tokens"] = 0
    jsons = html.find_all("script", 
                          attrs={"type": re.compile("application[/]{1}ld[+]{1}json")})
    for json_data_str in jsons:
        json_data = json.loads(json_data_str.get_text(), strict=False)
        if isinstance(json_data, list):
            if json_data:
                json_data = json_data[0]
            else:
                continue
        if "articleBody" in json_data:
            body, n_tokens = get_body_summary(remove_body_tags(json_data["articleBody"]), 
                                              url,
                                              media
                                              )
            data["body"] = body
            data["n_tokens"] = n_tokens
            break
    return data

def extract_data_from_jsons(html: BeautifulSoup, 
                            url: str,
                            media: str,
                            only_article_body: bool=True 
                            ) -> tuple[dict, dict]:
    data_extracted = {}
    title_found = False
    description_found = False
    body_found = False
    main_topic_found = False
    other_tag_found = False
    type_found = False
    creation_datetime_found = False
    modified_datetime_found = False
    extraction_completed = False
    type_value = None
    invalid_web = False
    jsons = html.find_all("script", 
                          attrs={"type": re.compile("application[/]{1}ld[+]{1}json")})
    for json_data_str in jsons:
        if invalid_web or extraction_completed:
            break
        json_text = json_data_str.get_text()
        if json_data_str is None or not isinstance(json_text, str):
            continue
        try:
            json_data = json.loads(json_text, 
                                   strict=False)
        except Exception as e:
            continue
        if isinstance(json_data, (list, tuple)):
            if json_data:
                json_data = json_data[0]
        for k, json_values in json_data.items():
            if not body_found and "articleBody" in k:
                body, n_tokens = get_body_summary(remove_body_tags(json_data["articleBody"]), 
                                                  url,
                                                  media
                                                  )
                data_extracted["body"] = body
                data_extracted["n_tokens"] = n_tokens
                body_found = True

            if not type_found and ("@type" in k or "type" in k):
                if isinstance(json_values, list):
                    for value in json_data["@type"]:
                        type_value = value.lower()
                        if "media" in type_value:
                            invalid_web = True
                            break
                        if (type_value.startswith("news") or type_value.endswith("article")):
                            type_found = True
                            break
                elif isinstance(json_values, str):
                    type_value = json_values.lower()
                    if (type_value.startswith("news") or type_value.endswith("article")) and "media" not in type_value:
                        type_found = True
            if invalid_web:
                # "Invalid web"
                break
            if only_article_body:
                if body_found:
                    extraction_completed = True
                continue
            if not description_found and "description" in k:
                data_extracted["description"] = json_values
                description_found = True
            if not creation_datetime_found and "datePublished" in k:
                data_extracted["creation_datetime"] = json_values
                creation_datetime_found = True
            if not modified_datetime_found and "dateModified" in k:
                data_extracted["modified_datetime"] = json_values
                modified_datetime_found = True
            if not title_found and "headline" in k:
                data_extracted["title"] = json_values
                title_found = True
            if not main_topic_found and "articleSection" in k:
                if isinstance(json_values, (dict, )): 
                    data_extracted["main_topic"] = json_values["@list"][0].lower()
                else:
                    data_extracted["main_topic"] = json_values.lower()
                main_topic_found = True
            if not other_tag_found and ("keywords" in k or "tags" in k):
                if not json_values:
                    continue
                if isinstance(json_values, (tuple, list)):
                    data_extracted["other_topic"] = ";".join(json_values).lower()
                elif isinstance(json_values, str):
                    if ", " in json_values:
                        data_extracted["other_topic"] = json_values.replace(", ", ",").lower()
                    else:
                        data_extracted["other_topic"] = json_values.lower()
                else:
                    continue
                other_tag_found = True
            if all((title_found, 
                    description_found,
                    body_found,
                    creation_datetime_found,
                    modified_datetime_found,
                    main_topic_found,
                    other_tag_found,
                    )):
                extraction_completed = True
                break
    if not type_found:
        return {}
    return data_extracted

def extract_data_from_metadata(parsed_html: BeautifulSoup, 
                               data_input: dict) -> tuple[dict, bool]:
    data_extacted = {}
    image_found = False
    if data_input.get("title", False):
        title_found = True
    else:
        title_found = False
    if data_input.get("description", False):
        description_found = True
    else:
        description_found = False
    if data_input.get("creation_datetime", False):
        creation_datetime_found = True
    else:
        creation_datetime_found = False
    if data_input.get("modified_datetime", False):
        modified_datetime_found = True
    else:
        modified_datetime_found = False
    if data_input.get("main_topic", False):
        main_topic_found = True
    else:
        main_topic_found = False
    if data_input.get("other_topic", False):
        other_topic_found = True
    else:
        other_topic_found = False

    extraction_completed = False
    meta_tags = parsed_html.select("html head meta[property],[name]")
    for meta_tag in meta_tags:
        if extraction_completed:
            break
        # Possible attributes:
        # property
        attribute_val = meta_tag.attrs.get("property", "")
        # name
        if not attribute_val:
            attribute_val = meta_tag.attrs.get("name", "")
            if not attribute_val:
                continue
        attribute_val = attribute_val.lower()
        meta_content = meta_tag.attrs.get("content", "")
        if not meta_content:
            continue
        if not title_found and "title" in attribute_val:
            data_extacted["title"] = meta_content
            title_found = True
        if not description_found and "description" in attribute_val:
            data_extacted["description"] = meta_content
            description_found = True
        #if not creation_datetime_found and ("publish" in attribute_val and "time" in attribute_val):
        if not creation_datetime_found and re.search(r"publish(?:ed)?_?(?:time|date)", attribute_val):
            data_extacted["creation_datetime"] = meta_content
            creation_datetime_found = True
        if not main_topic_found and "section" in attribute_val:
            data_extacted["main_topic"] = meta_content.lower()
            main_topic_found = True
        if not other_topic_found and "keyword" in attribute_val:
            if ", " in meta_content:
                data_extacted["other_topic"] = meta_content.replace(", ", ",").lower()
            else:
                data_extacted["other_topic"] = meta_content.lower()
            other_topic_found = True
        #if not modified_datetime_found and ("modif" in attribute_val and "time" in attribute_val):
        if not modified_datetime_found and re.search(r"modif(?:ied)?_?(?:time|date)", attribute_val):
            data_extacted["modified_datetime"] = meta_content
            modified_datetime_found = True
        if not image_found and attribute_val.endswith("image"):
            data_extacted["image_url"] = meta_content
            image_found = True
        if all((title_found, 
                description_found,
                creation_datetime_found,
                modified_datetime_found,
                main_topic_found,
                other_topic_found,
                image_found
                )):
            extraction_completed = True
    return data_extacted

def extract_keys_with_gpt(parsed_code: BeautifulSoup) -> dict:
    #print("..Keys through gpt..")
    tags_with_text = parsed_code.find_all(lambda tag: tag.name in ("p", "h1", "h2"))
    text_clean_from_tags = "".join([re.sub("\n+", "\n", tag.get_text()) for tag in tags_with_text])
    try:
        openai_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", 
                 "content": prompt_role_summary},
                {"role": "user", 
                 "content": text_clean_from_tags},
            ]
        )
        message_content = openai_response["choices"][0].message["content"]
        try:
            n_tokens = regex_n_tokens.search(message_content).groups()[0]
        except:
            n_tokens = -1
        try:
            title = regex_headline.search(message_content).groups()[0]
        except:
            title = ""
        try:
            tags = regex_topics.search(message_content).groups()[0]
        except:
            tags = ""
        try:
            creation_datetime = regex_creation_datetime.search(message_content).groups()[0]
        except:
            creation_datetime = ""
        try:
            modified_datetime = regex_update_datetime.search(message_content).groups()[0]
        except:
            modified_datetime = ""
        try:
            body = regex_only_summary.search(message_content).groups()[0]
        except:
            body = ""
        data = {
            "n_tokens": n_tokens,
            "title": title,
            "other_topic": tags,
            "creation_datetime": creation_datetime,
            "modified_datetime": modified_datetime,
            "body": body
            }
        data["image_url"] = ""

        return data
    except Exception as e:
        return {}
    
def remove_body_tags(text: str) -> str:
    return re.sub("<.*?>", "", text)

def find_body_with_gpt(parsed_html: BeautifulSoup, 
                       url: str,
                       media: str
                       ) -> dict:
    tags_with_text = parsed_html.find_all(lambda tag: tag.name in ("p", ))
    text_clean_from_tags = "".join([re.sub("\n+", "\n", tag.get_text()) for tag in tags_with_text])
    #clean_paragraphs = clean_paragraphs.split("\n")[0]
    #parag_texts = str({i: x if x else "\n" for i, x in enumerate(clean_paragraphs)})[1:-1]
    body, n_tokens = get_body_summary(text_clean_from_tags, 
                                      url,
                                      media
                                      )
    data = {
        "n_tokens": n_tokens,
        "body": body
    }
    return data

#@error_logger(file_manager, FILE_NAME_EXTRACTION_ERRORS)
@garbage_logger()
def get_body_summary(text: str, 
                     url: str,
                     media: str
                     ) -> tuple[str, str]:
    if BLOCK_API_CALL:
        #return (text, -1), {"status_code": STATUS_0, "id": ""}
        return (text, -1), {"has_garbage": True, "data": []}
    try:
        openai_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt_role_summary},
                {"role": "user", "content": text},
            ]
        )
    except Exception as e:
        #return ("", 0), {"status_code": STATUS_4, "id": url}
        return ("", 0), {"has_garbage": False, "data": [(url, media, STATUS_4)]}
    
    message_content = openai_response["choices"][0].message["content"]
    try:
        n_tokens = regex_n_tokens.search(message_content).groups()[0]
    except:
        n_tokens = -1
    try:
        body_summary = regex_only_summary.search(message_content).groups()[0]
    except:
        body_summary = ""
    if "BodyNotFound" in body_summary:
        body_summary = text
        n_tokens = -1
    #return (body_summary, n_tokens), {"status_code": STATUS_0, "id": ""}
    return (body_summary, n_tokens), {"has_garbage": True, "data": []}

def treat_raw_urls(raw_urls: list, 
                   media_url: str,
                   score: float):
    
    media_stats_manager = ut.StatisticsManager(True)
    urls = find_valid_urls(raw_urls,
                           media_url)
    if urls:
        lock = mp.Lock()
        with lock:
            read_urls = pd.DataFrame(read_stored_news(media_url), 
                                     columns=["in_store"])
        extracted_urls = pd.DataFrame(urls, 
                                      columns=["in_extraction"]
                                      ).drop_duplicates()
        novel_news = read_urls.merge(extracted_urls, 
                                     left_on="in_store", 
                                     right_on="in_extraction", 
                                     how="right")
        # Merge extracted and loaded news in order to process the new ones
        novel_news_to_process = novel_news \
            .loc[novel_news.in_store.isnull(), "in_extraction"] \
            .tolist()
        
        # Process new news
        news_data, n_no_body = find_urls_data(novel_news_to_process, 
                                              media_url=media_url,
                                              score=score,
                                              order_keys=True) # ordered
        #print("Data:", news_data)
        if news_data:
            lock = mp.Lock()
            with lock:
                insert_news(news_data)
                print("Rows inserted:", len(news_data))
        # Statistics
        n_processed = len(news_data)
        print(f"PID {str(os.getpid())} {media_url}; read {len(read_urls)}; extracted {len(extracted_urls)}; to process {len(novel_news_to_process)}; processed: {n_processed}\n")
    else:
        n_processed = 0
        n_no_body = 0
    media_stats_manager.write_stats((media_url, 
                                     n_processed, 
                                     n_no_body), 
                                     input_iso_datetime=TODAY_LOCAL_DATETIME,
                                     pid=os.getpid()
                                     )
    return n_processed, n_no_body

#@error_logger(file_manager, FILE_NAME_EXTRACTION_ERRORS)
@garbage_logger()
def find_valid_urls(input_urls: list, 
                         media_url: str
                         ) -> (list, bool):
    # Open file of garbage urls to avoid
    lock = mp.Lock()
    with lock:
        #garbage_urls = ut.read_json_file(PATH_GARBAGE_URLS, 
        #                                 FILE_NAME_GARBAGE_URLS).get(media_url, [])
        read_urls = pd.DataFrame(ut.read_garbage((media_url, )), 
                                 columns=["in_store", "media_store"]
                                 )
    # Urls from garbage table query
    novel_urls = read_urls.merge(pd.DataFrame({"in_extraction": input_urls, 
                                               "media_extraction": [media_url] * len(input_urls)}) \
                                                .drop_duplicates(), 
                                 left_on=["in_store", "media_store"], 
                                 right_on=["in_extraction", "media_extraction"], 
                                 how="right")
    # Merge extracted and loaded news in order to process the new ones
    urls_to_process = novel_urls.loc[novel_urls[["in_store", "media_store"]].isnull().all(axis=1), "in_extraction"] 
    #print("garbage_urls:", len(read_urls), "input urls:", len(input_urls), "urls_to_process:", len(urls_to_process))
    new_garbage = []
    valid_urls = []
    #status_and_id = []
    for url in urls_to_process.tolist():
        # Filter out urls with query symbols
        if url == media_url:
            #status_and_id.append({"status_code": STATUS_3_1, "id": url})
            new_garbage.append((url, media_url, STATUS_3_1))
            continue
        if  url.endswith(".xml") \
            or url.endswith(".pdf") \
            or url.endswith(".lxml") \
            or url.endswith(".jpg") \
            or url.endswith(".png") \
            or url.endswith(".gif"):
            #status_and_id.append({"status_code": STATUS_3_2, "id": url})
            new_garbage.append((url, media_url, STATUS_3_2))
            continue
        if "pagina-1.html" in url \
            or "/firmas" in url \
            or "/humor/" in url \
            or "/autor" in url \
            or "/autores/" in url \
            or "/foto/" in url \
            or "/fotos/" in url \
            or "/video/" in url \
            or "/videos/" in url \
            or "/opinión/" in url \
            or "/opinion/" in url:
            #status_and_id.append({"status_code": STATUS_3_3, "id": url})
            new_garbage.append((url, media_url, STATUS_3_3))
            continue
        url_splits = [x for x in url.split("/") if x][2:]
        if len(url_splits) <= 1:
            #status_and_id.append({"status_code": STATUS_3_4, "id": url})
            new_garbage.append((url, media_url, STATUS_3_4))
            continue
        elif len(url_splits) >= 2:
            if regex_too_short_url_end.search(url_splits[-1]):
                #status_and_id.append({"status_code": STATUS_3_5, "id": url})
                new_garbage.append((url, media_url, STATUS_3_5))
                continue
        if media_url[:-1] not in url:
            #status_and_id.append({"status_code": STATUS_3_6, "id": url})
            new_garbage.append((url, media_url, STATUS_3_6))
            continue
        if regex_url_startswith_https.search(url):
            valid_urls.append(url)
        
    #print(f"\t{media_url}; new garbage urls {len(new_garbage)};", f"Treat {len(valid_urls)} urls\n")
    #cache = {"status_code": status_and_id}
    if new_garbage:
        return list(set(valid_urls)), {"has_garbage": False, "data": new_garbage}
    else:
        return list(set(valid_urls)), {"has_garbage": True, "data": new_garbage}
    
@garbage_logger()
def find_urls_data(news_urls: list, 
                   media_url: str,
                   score: float,
                   order_keys=False
                   ):
    news_data = []
    garbage_urls = []
    n_no_articlebody_in_article = 0
    for news_url in news_urls:
        data, code = _extract_data_from_news(news_url, 
                                             media_url,
                                             score)
        if not data:
            garbage_urls.append((news_url, media_url, code))
            if code == STATUS_5_6:
                n_no_articlebody_in_article += 1
            continue
        if order_keys:
            news_data.append(order_dict_keys(data))
        else:
            news_data.append(data)
    print("Garbage from data extraction:", len(garbage_urls))
    if garbage_urls:
        return (news_data, n_no_articlebody_in_article), {"has_garbage": False, "data": garbage_urls}
    else:
        return (news_data, n_no_articlebody_in_article), {"has_garbage": True, "data": garbage_urls}
    
def _extract_data_from_news(news_url: list, 
                            media_url: str,
                            score: float):
    data = {}
    #file_name = str(os.getpid()) + "_" + FILE_NAME_EXTRACTION_ERRORS
    try:
        resp_url_news = requests.get(news_url, 
                                     headers=HEADERS, 
                                     timeout=MEDIA_GET_REQ_TIMEOUT,
                                     )
    except requests.exceptions.TooManyRedirects as e1:
        #print("An error 1 occurred; news request:", news_url, e1)
        #file_manager.write_on_file(file_name, 
        #                           [{"status_code": STATUS_1_1, "id": news_url}])
        return data, STATUS_1_1
    except requests.exceptions.RequestException as e2:
        #print("An error 2 occurred; news request:", news_url, e2)
        #file_manager.write_on_file(file_name, 
        #                           [{"status_code": STATUS_1_2, "id": news_url}])
        return data, STATUS_1_2
    except UnicodeDecodeError as e3:
        #print("An error 3 occurred; news request:", news_url, e3)
        #file_manager.write_on_file(file_name, 
        #                           [{"status_code": STATUS_1_3, "id": news_url}])
        return data, STATUS_1_3
    except Exception as e4:
        #print("An error 4 occurred; news request:", news_url, e4)
        #file_manager.write_on_file(file_name, 
        #                           [{"status_code": STATUS_1_2, "id": news_url}])
        return data, STATUS_1_2
    
    parsed_news_hmtl = BeautifulSoup(resp_url_news.content, 
                                     "html.parser")
    # Accept or reject url if news date is more than N_MAX_DAYS_OLD days older
    try:
        meta_tag_published_time = parsed_news_hmtl.html.head.find("meta", 
                                                                  attrs={"property": regex_published_time})
        if meta_tag_published_time is None:
            return data, STATUS_5_2
        publ_tsm = meta_tag_published_time.attrs["content"]
        if not publ_tsm:
            return data, STATUS_5_3
        dtime_diff = (TODAY_LOCAL_DATETIME - datetime.fromisoformat(publ_tsm)).total_seconds() / SECONDS_IN_DAY
        if dtime_diff > N_MAX_DAYS_OLD:
            return data, STATUS_5_4
    except:
        return data, STATUS_5_1
    
    extracted_data = extract_data_from_jsons(parsed_news_hmtl, 
                                             news_url,
                                             media_url
                                             )
    if extracted_data:
        data.update(extracted_data)
    extracted_data = extract_data_from_metadata(parsed_news_hmtl, 
                                                data
                                                )
    if extracted_data:
        data.update(extracted_data)
    if not data.get("title", False) or not data.get("creation_datetime", False):
        return {}, STATUS_5_5
    if not data.get("body", False):
        extracted_data = find_body_with_gpt(parsed_news_hmtl, 
                                            news_url,
                                            media_url
                                            )
        if not extracted_data.get("body", False):
            return {}, STATUS_5_6
        data.update(extracted_data)

    data["country"] = parsed_news_hmtl.html.attrs.get("lang", "")
    # TODO complete this
    data["media_url"] = media_url
    data["url"] = news_url
    data["score"] = score
    return data, STATUS_0

def order_dict_keys(keys_values: list[dict], 
                    only_values: bool=True):
    if only_values:
        return tuple([keys_values.get(target_k, "") for target_k in ORDER_KEYS])
    else:
        return {target_k: keys_values.get(target_k, "")  for target_k in ORDER_KEYS}
    
def read_stored_news(where_params):
    with sqlite3.connect(DB_NAME_NEWS, 
                         timeout=DB_TIMEOUT) as conn:
        cursor = conn.cursor()
        #create_news_table(conn, 
        #                cursor)
        if not isinstance(where_params, (tuple, list)):
            where_params = (where_params, )
        query_str = """
            SELECT 
                url
            FROM 
                news
            WHERE
                mediaUrl = ?;
            """
    return cursor.execute(query_str, where_params).fetchall()

def create_news_table(conn, 
                      cursor):
    query_str = """
        CREATE TABLE IF NOT EXISTS news (
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            articleBody TEXT NOT NULL,
            mediaUrl TEXT,
            country TEXT,
            creationDate TEXT NOT NULL,
            updateDate TEXT,
            url TEXT PRIMARY KEY NOT NULL,
            imageUrl TEXT,
            mainTopic TEXT,
            otherTopic TEXT,
            insertDate Text NOT NULL,
            nTokens Integer
            )
            ;
        """
    cursor.execute(query_str)
    conn.commit()

def insert_news(data: tuple[tuple]):
    with sqlite3.connect(DB_NAME_NEWS, 
                         timeout=DB_TIMEOUT) as conn:
        cursor = conn.cursor()
        query_str = f"""
            INSERT INTO news
                (
                url,
                mediaUrl,
                creationDate,
                insertDate,
                updateDate,    
                title,
                description,
                articleBody,
                mainTopic,
                otherTopic,
                imageUrl,
                country,
                nTokens,
                score,
                preprocessed
                )
                    VALUES
                (
                    ?,
                    ?,
                    ?,
                    DATETIME('now', 'localtime', 'utc'),
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    False
                )
                ;
            """
        if not isinstance(data, tuple):
            data = tuple(data)
        cursor.executemany(query_str, 
                           data)
        conn.commit()

def run_multi_process(file_path: str, 
                      num_processes: int, 
                      process_function):
    
    with open(file_path, "r") as file:
        data = json.load(file)

    with mp.Manager() as manager:
        sections_data = manager.dict(data)
        # Create a process for each chunk and run in parallel
        processes = []
        for _ in range(num_processes):
            process = mp.Process(target=process_function, 
                                 args=(sections_data, )
                                 )
            processes.append(process)
            process.start()
        # Wait for all processes to finish
        for process in processes:
            process.join()

def main_multi_threading_process(queued_media_data: dict):
    pid = str(os.getpid())
    file_manager.add_files([
        pid + "_" + FILE_NAME_EXTRACTION_ERRORS
        ])
    # 1 Initialize reboot if needed
    media_checkpoint = ut.read_news_checkpoint(pid, 
                                               "w+")
    if not media_checkpoint:
        checkpoint_started = False
    else:
        checkpoint_started = True
    # 1 End
    media_stats_manager = ut.StatisticsManager().restart_time()
    media_news_urls = []
    n_processed = 0
    n_no_body = 0
    while len(queued_media_data) > 0:
        # The Dict-proxy can safely remove items, as the multiprocessing.Manager()  
        # takes care of the necessary internal locking mechanisms to ensure 
        # safe concurrent access.
        media_url, sections_and_scores = queued_media_data.popitem()
        for i, (section_url, score) in enumerate(sections_and_scores["data"]):
            score = float(score)
            section_url = section_url.strip()
            if not media_url.startswith("https://"):
                media_url = "https://" + media_url
            if checkpoint_started:
                if media_url == media_checkpoint:
                    checkpoint_started = False
                else:
                    continue
            try:
                response = requests.get(section_url, 
                                        headers=HEADERS, 
                                        timeout=NEWS_ARTICLE_GET_REQ_TIMEOUT)
            except requests.exceptions.Timeout:
                # Handle the timeout exception
                print("The request timed out.")
                ErrorReporter(f"Get request timeout exception at {section_url}")
                continue
            except requests.exceptions.RequestException as e:
                # Handle other request exceptions
                print(f"An error occurred: {str(e)}")
                print("Retrying request without 'www'")
                section_url = section_url.replace("www.", "")
                response = requests.get(section_url, 
                                        headers=HEADERS, 
                                        timeout=NEWS_ARTICLE_GET_REQ_TIMEOUT)
                #ErrorReporter(f"Get request general exception, {e}, at {section_url}")
                media_url = media_url.replace("www.", "")
                print("Request success without 'www'")
            
            parsed_hmtl = BeautifulSoup(response.content, 
                                        "html.parser")
            tags_with_url = parsed_hmtl.html.body \
                                       .find_all("a", 
                                                 href=re.compile(r"^(?:https:\/\/)?|^\/{1}[^\/].*|^www[.].*"))
                                                # href=re.compile("https?:.*"))
            raw_urls = [x.attrs["href"] for x in tags_with_url if x.attrs.get("href", False)]
            for raw_url in raw_urls:
                if re.search("[?=&+%@#]{1}", raw_url):
                    search_spec_char = regex_url_has_query.search(raw_url)
                    query_start_pos = search_spec_char.span()[0]
                    raw_url = raw_url[:query_start_pos]
                if raw_url.startswith("//"):
                    raw_url = "https:" + raw_url
                elif raw_url.startswith("/"):
                    raw_url = media_url[:-1] + raw_url
                elif raw_url.startswith("www.") or not raw_url.startswith("https://"):
                    raw_url = "https://" + raw_url
                media_news_urls.append(raw_url)
        n1, n2 = treat_raw_urls(list(set(media_news_urls)), 
                                     media_url,
                                     score
                                     )
        n_processed += n1
        n_no_body += n2
        if not checkpoint_started and (i % N_SAVE_CHECKPOINT == 0):
            ut.save_checkpoint(pid, 
                               media_url)
    media_stats_manager.write_stats(("Process summary", 
                                    n_processed, 
                                    n_no_body), 
                                    input_iso_datetime=TODAY_LOCAL_DATETIME,
                                    pid="summary"
                                    )
if __name__ == "__main__":
    print(f"\n...Datetime of process: {CURRENT_DATE} {CURRENT_TIME}...\n")
    if FULL_START:
        for i in range(NUM_CORES):
            ut.save_checkpoint(str(i), "")

    # Extract the last version
    version_n = max(int(x.split("_")[-1][1:-5]) for x in glob.glob("../data/final_url_sections_v*.json"))
    sections_file_path = f"final_url_sections_v{version_n}.json"
    file_path = os.path.join(PATH_DATA, sections_file_path)

    run_multi_process(file_path, 
                      NUM_CORES, 
                      main_multi_threading_process)
    file_manager.close_all_files()
    for f in glob.glob(os.path.join(PATH_DATA, "checkpoint_*.json")):
        os.remove(os.path.join(PATH_DATA, f))
    print("\n...The process ended...")
