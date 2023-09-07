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
import multiprocessing
from tqdm import tqdm

from utilities import *
# Classes
class ErrorReporter(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)
        with open("Errors log.txt", "w") as errors_file:
            errors_file.write(message)
## Constants
FULL_START = True
BLOCK_API_CALL = True
DB_TIMEOUT = 10.0
MEDIA_GET_REQ_TIMEOUT = 8.0
NEWS_ARTICLE_GET_REQ_TIMEOUT = 6.0
SECONDS_IN_DAY = 86_400
N_MAX_DAYS_OLD = 1
NUM_CORES = 10
MAX_N_MEDIAS = 50
N_SAVE_CHECKPOINT = 5
# Status codes
STATUS_0 = "0"
STATUS_1_1 = "1_1"
STATUS_1_2 = "1_2"
STATUS_1_3 = "1_3"
STATUS_2 = "2"
STATUS_3_1 = "3_1"
STATUS_3_2 = "3_2"
STATUS_3_3 = "3_3"
STATUS_3_4 = "3_4"
STATUS_3_5 = "3_5"
STATUS_3_6 = "3_6"
STATUS_4 = "4"
# File names
PATH_DATA = os.path.join("..", "data")
FILE_NAME_EXTRACTION_ERRORS = os.path.join(PATH_DATA, "extraction-errors.txt")
FILE_NAME_PROMPT_ROLE_KEYS = os.path.join(PATH_DATA, "role-prompt-keys-extraction.txt")
FILE_NAME_PROMPT_ROLE_SUMMARY = os.path.join(PATH_DATA, "role-prompt-body-extraction.txt")
# Database names
# Managers instantiation
file_manager = FileManager()
file_manager.add_files([
    FILE_NAME_EXTRACTION_ERRORS
    ])
DB_NAME_NEWS = os.path.join(PATH_DATA, "news_db.sqlite3")
# Dates and times
TODAY_LOCAL_DATETIME = datetime.now().replace(tzinfo=timezone.utc)
CURRENT_DATE, CURRENT_TIME = str(datetime.today()).split(" ")
# Data structures
ORDER_KEYS = ("title",
              "body", 
              "source",
              "country",
              "creation_datetime",
              "modified_datetime",
              "url",
              "image",
              "tags",
              "n_tokens"
              )
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
}
## Dynamical global variables
if not BLOCK_API_CALL:
    openai.api_key = os.getenv("OPENAI_API_KEY")
#with open(FILE_NAME_PROMPT_ROLE_KEYS, "r") as file:
#    prompt_role = file.read()
with open(FILE_NAME_PROMPT_ROLE_SUMMARY, "r") as file:
    prompt_role_summary = file.read()

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
node_ignore = ["europa", 
               "africa",
               "asia",
               "oceania",
               "prensadigital"
              ]

## Decorators
# Decorator function to handle inputs and error logging
def error_logger(manager: FileManager,
                 file_name: str
                 ):
    def decorator(target_func):
        def wrapper(*args, **kwargs):
            data, cache = target_func(*args, **kwargs)
            status_code = cache.get("status_code", False)
            if isinstance(cache, (list, tuple)) and status_code:
                manager.write_on_file(file_name,
                                      status_code
                                      )
                return data
            status_code = cache.get("status_code", "0")
            if isinstance(status_code, dict) and status_code != "0":
                manager.write_on_file(file_name,
                                      [cache]
                                      )
            return data
        return wrapper
    return decorator

## Functions
def find_news_body_from_json(html: BeautifulSoup, 
                             data: dict, 
                             url: str
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
                                              url
                                              )
            data["body"] = body
            data["n_tokens"] = n_tokens
            break
    return data

def extract_data_from_jsons(html: BeautifulSoup, 
                            url: str,
                            only_article_body: bool=True 
                            ) -> tuple[dict, dict]:
    data_extracted = {}
    title_found = False
    body_found = False
    tags_found = False
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
                                                  url
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
            if not creation_datetime_found and "datePublished" in k:
                data_extracted["creation_datetime"] = json_values
                creation_datetime_found = True
            if not modified_datetime_found and "dateModified" in k:
                data_extracted["modified_datetime"] = json_values
                modified_datetime_found = True
            if not title_found == "headline" in k:
                data_extracted["title"] = json_values
                title_found = True
            if not tags_found and ("keywords" in k or "tags" in k):
                if not json_values:
                    continue
                if isinstance(json_values, (tuple, list)):
                    data_extracted["tags"] = ";".join(json_values)
                elif isinstance(json_values, str):
                    if ", " in json_values:
                        data_extracted["tags"] = json_values.replace(", ", ",")
                    else:
                        data_extracted["tags"] = json_values
                else:
                    continue
                tags_found = True
            if all((title_found, 
                    body_found,
                    creation_datetime_found,
                    modified_datetime_found,
                    tags_found,
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
    if data_input.get("creation_datetime", False):
        creation_datetime_found = True
    else:
        creation_datetime_found = False
    if data_input.get("modified_datetime", False):
        modified_datetime_found = True
    else:
        modified_datetime_found = False
    if data_input.get("tags", False):
        tags_found = True
    else:
        tags_found = False

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
        #if not creation_datetime_found and ("publish" in attribute_val and "time" in attribute_val):
        if not creation_datetime_found and re.search(r"publish(?:ed)?_?(?:time|date)", attribute_val):
            data_extacted["creation_datetime"] = meta_content
            creation_datetime_found = True
        if not tags_found and "keyword" in attribute_val:
            if ", " in meta_content:
                data_extacted["tags"] = meta_content.replace(", ", ",")
            else:
                data_extacted["tags"] = meta_content
            tags_found = True
        #if not modified_datetime_found and ("modif" in attribute_val and "time" in attribute_val):
        if not modified_datetime_found and re.search(r"modif(?:ied)?_?(?:time|date)", attribute_val):
            data_extacted["modified_datetime"] = meta_content
            modified_datetime_found = True
        if not image_found and attribute_val.endswith("image"):
            data_extacted["image"] = meta_content
            image_found = True
        if all((title_found, 
                creation_datetime_found,
                modified_datetime_found,
                tags_found,
                image_found
                )):
            extraction_completed = True
    return data_extacted

def extract_keys_with_gpt(parsed_code: BeautifulSoup) -> dict:
    #print("..Keys through gpt..")
    tags_with_text = parsed_code.find_all(lambda tag: tag.name in ("p", "h1", "h2"))
    text_clean_from_tags = "".join([re.sub("\n+", "\n", tag.get_text()) for tag in tags_with_text])
    text_clean_from_tags
    ok = False
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
        ok = True
    except Exception as e:
        #print("Keys GPT error:", e)
        pass
        
    if ok:
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
            "tags": tags,
            "creation_datetime": creation_datetime,
            "modified_datetime": modified_datetime,
            "body": body
            }
        data["image"] = ""

        return data
    else:
        return {}
    
def remove_body_tags(text: str) -> str:
    return re.sub("<.*?>", "", text)

def find_news_body_with_gpt(parsed_html: BeautifulSoup, 
                            url: str
                            ) -> dict:
    tags_with_text = parsed_html.find_all(lambda tag: tag.name in ("p", ))
    text_clean_from_tags = "".join([re.sub("\n+", "\n", tag.get_text()) for tag in tags_with_text])
    #clean_paragraphs = clean_paragraphs.split("\n")[0]
    #parag_texts = str({i: x if x else "\n" for i, x in enumerate(clean_paragraphs)})[1:-1]
    body, n_tokens = get_body_summary(text_clean_from_tags, 
                                      url
                                      )
    data = {
        "n_tokens": n_tokens,
        "body": body
    }
    return data

@error_logger(file_manager, FILE_NAME_EXTRACTION_ERRORS)
def get_body_summary(text: str, 
                     url: str
                     ) -> tuple[str, str]:
    if BLOCK_API_CALL:
        #return ("noarticlebody", -1), {"status_code": STATUS_0, "id": ""}
        return (text, -1), {"status_code": STATUS_0, "id": ""}
    try:
        openai_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt_role_summary},
                {"role": "user", "content": text},
            ]
        )
    except Exception as e:
        return ("", 0), {"status_code": STATUS_4, "id": url}
    
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
    return (body_summary, n_tokens), {"status_code": STATUS_0, "id": ""}

def treat_raw_news_urls(news_urls: list, 
                        media_url: str,
                        pid: str
                        ):
    
    media_stats_manager = StatisticsManager().restart_time()
    urls = find_valid_news_urls(news_urls,
                                media_url)
    lock = multiprocessing.Lock()
    with lock:
        select_urls = pd.DataFrame(read_stored_news(media_url), 
                           columns=["in_store"]
                           )
    extracted_urls = pd.DataFrame(urls, 
                                  columns=["in_extraction"]
                                  ).drop_duplicates()
    df_novels = select_urls.merge(extracted_urls, 
                                  left_on="in_store", 
                                  right_on="in_extraction", 
                                  how="right")
    # Merge extracted and loaded news in order to process the new ones
    novel_news_to_process = df_novels \
        .loc[df_novels.in_store.isnull(), "in_extraction"]\
            .tolist()
    print(f"Read; extracted;to_process: {len(select_urls)}, {len(extracted_urls)}, {len(novel_news_to_process)}, {len(extracted_urls) >= len(novel_news_to_process)}")
    # Process new news
    news_data, n_no_body = find_news_data(novel_news_to_process, 
                                          author=media_url,
                                          pid=pid,
                                          order_keys=True) # ordered
    if news_data:
        with lock:
            insert_news(news_data)
    # Statistics
    n_processed = len(news_data)
    media_stats_manager.write_extraction_stats((media_url, 
                                                n_processed, 
                                                n_no_body), 
                                                TODAY_LOCAL_DATETIME,
                                                pid=pid
                                                )
    if n_processed > 0:
        print(f"\tmedia name: {media_url} processed counts: {str(n_processed)}")
    return n_processed, n_no_body

@error_logger(file_manager, FILE_NAME_EXTRACTION_ERRORS)
def find_valid_news_urls(news_urls: list, 
                         media_url: str
                         ) -> (list, bool):
    valid_news_urls = []
    status_cache = []
    for news_url in news_urls:
        # Filter out urls with query symbols
        if re.search("[?=&+%@#]{1}", news_url):
            search_spec_char = re.search("[?=&+%@#]{1}", news_url)
            query_start = search_spec_char.span()[0]
            news_url = news_url[:query_start]
        if news_url == media_url:
            status_cache.append({"status_code": STATUS_3_1, "id": news_url})
            continue
        if  news_url.endswith(".xml") \
            or news_url.endswith(".pdf") \
            or news_url.endswith(".lxml") \
            or news_url.endswith(".jpg") \
            or news_url.endswith(".png") \
            or news_url.endswith(".gif"):
            status_cache.append({"status_code": STATUS_3_2, "id": news_url})
            continue
        if "pagina-1.html" in news_url \
            or "/firmas" in news_url \
            or "/humor/" in news_url \
            or "/autor" in news_url \
            or "/autores/" in news_url \
            or "/foto/" in news_url \
            or "/fotos/" in news_url \
            or "/video/" in news_url \
            or "/videos/" in news_url \
            or "/opinión/" in news_url \
            or "/opinion/" in news_url:
            status_cache.append({"status_code": STATUS_3_3, "id": news_url})
            continue
        news_url_splits = [x for x in news_url.split("/") if x][2:]
        if len(news_url_splits) <= 1:
            status_cache.append({"status_code": STATUS_3_4, "id": news_url})
            continue
        elif len(news_url_splits) >= 2:
            if re.search(r"^[.a-zA-Z0-9]+(-[.a-zA-Z0-9]+){,2}$", news_url_splits[-1]):
                status_cache.append({"status_code": STATUS_3_5, "id": news_url})
                continue
        if media_url[:-1] not in news_url:
            status_cache.append({"status_code": STATUS_3_6, "id": news_url})
            continue
        if re.search("https?:[\/]{2}", news_url):
            valid_news_urls.append(news_url)
    cache = {"status_code": status_cache}
    return list(set(valid_news_urls)), cache

def find_news_data(news_urls: list, 
                   author: str,
                   pid: str,
                   order_keys=False
                   ):
    news_media_data = []
    n_no_articlebody_in_article = 0
    for news_url in news_urls:
        try:
            resp_url_news = requests.get(news_url, 
                                         headers=HEADERS, 
                                         timeout=MEDIA_GET_REQ_TIMEOUT,
                                         )
        except requests.exceptions.TooManyRedirects as e1:
            print(news_url, "An error 1 occurred:", e1)
            file_manager.write_on_file(FILE_NAME_EXTRACTION_ERRORS, 
                                       [{"status_code": STATUS_1_1, "id": news_url}])
            continue
        except requests.exceptions.RequestException as e2:
            print(news_url, "An error 2 occurred:", e2)
            file_manager.write_on_file(FILE_NAME_EXTRACTION_ERRORS, 
                                       [{"status_code": STATUS_1_2, "id": news_url}])
            continue
        except UnicodeDecodeError as e3:
            print(news_url, "An error 3 occurred:", e3)
            file_manager.write_on_file(FILE_NAME_EXTRACTION_ERRORS, 
                                       [{"status_code": STATUS_1_3, "id": news_url}])
            continue
        except Exception as e4:
            print(news_url, "An error 4 occurred:", e4)
            file_manager.write_on_file(FILE_NAME_EXTRACTION_ERRORS, 
                                       [{"status_code": STATUS_1_2, "id": news_url}])
            continue
        
        parsed_news_hmtl = BeautifulSoup(resp_url_news.content, 
                                        "html.parser")
        # Accept or reject url if news date is more than N_MAX_DAYS_OLD days older
        try:
            meta_tag_published_time = parsed_news_hmtl.html.head.find("meta", 
                                                                      attrs={"property": re.compile(r"publish(?:ed)?_?(?:time|date)")})
            if meta_tag_published_time is None:
                continue
            publ_tsm = meta_tag_published_time.attrs["content"]
            if not publ_tsm:
                continue
            dtime_diff = (TODAY_LOCAL_DATETIME - datetime.fromisoformat(publ_tsm)).total_seconds() / SECONDS_IN_DAY
            if dtime_diff > N_MAX_DAYS_OLD:
                continue
        except:
            #print("Time difference not calculated", meta_tag_published_time, ";", news_url)
            continue
        data = {}
        extracted_data = extract_data_from_jsons(parsed_news_hmtl, 
                                                 news_url
                                                 )
        if extracted_data:
            data.update(extracted_data)
        extracted_data = extract_data_from_metadata(parsed_news_hmtl, 
                                                    data
                                                    )
        if extracted_data:
            data.update(extracted_data)
        if not data.get("title", False) or not data.get("creation_datetime", False):
            continue
        if not data.get("body", False):
            n_no_articlebody_in_article += 1
            extracted_data = find_news_body_with_gpt(parsed_news_hmtl, 
                                                     news_url)
            if not extracted_data.get("body", False):
                continue
            data.update(extracted_data)

        data["country"] = parsed_news_hmtl.html.attrs.get("lang", "")
        # TODO complete this
        data["source"] = author
        data["url"] = news_url
        if order_keys:
            news_media_data.append(order_dict_keys(data))
        else:
            news_media_data.append(data)
    return news_media_data, n_no_articlebody_in_article

def order_dict_keys(keys_values: list[dict], 
                    only_values: bool=True):
    if only_values:
        return tuple([keys_values.get(target_k, "") for target_k in ORDER_KEYS])
    else:
        return {target_k: keys_values.get(target_k, "")  for target_k in ORDER_KEYS}
    
def read_stored_news(where_params):
    with sqlite3.connect(DB_NAME_NEWS, timeout=DB_TIMEOUT) as conn:
        cursor = conn.cursor()
        create_news_table(conn, 
                        cursor)
        if not isinstance(where_params, (tuple, list)):
            where_params = (where_params, )
        query_str = """
            SELECT 
                url
            FROM 
                News
            WHERE
                source = ?;
            """
        output = cursor.execute(query_str, where_params)
        conn.commit()
    return output

def create_news_table(conn, 
                      cursor):
    query_str = """
        CREATE TABLE IF NOT EXISTS News (
            title TEXT NOT NULL,
            article TEXT NOT NULL,
            source TEXT,
            country TEXT,
            creationDate TEXT NOT NULL,
            updateDate TEXT,
            url TEXT PRIMARY KEY NOT NULL,
            image_url TEXT,
            tags TEXT,
            insertDate Text NOT NULL,
            changeDate Text,
            number_tokens Integer
        )
            ;
        """
    cursor.execute(query_str)
    conn.commit()

def insert_news(data: tuple[tuple]):
    with sqlite3.connect(DB_NAME_NEWS, 
                         timeout=DB_TIMEOUT) as conn:
        cursor = conn.cursor()
        create_news_table(conn, 
                          cursor)
        #datetime('now','localtime'),
        query_str = f"""
            INSERT INTO News
                (title,
                article,
                source,
                country,
                creationDate,
                updateDate,
                url,
                image_url,
                tags,
                insertDate,
                changeDate,
                number_tokens
                )
                    VALUES
                (
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    ?,
                    DATETIME('now', 'localtime'),
                    '',
                    ?
                )
                ;
            """
        if not isinstance(data, tuple):
            data = tuple(data)
        cursor.executemany(query_str, 
                           data)
        conn.commit()

def split_file_and_process(sections_file_path: str, 
                           num_splits: int, 
                           process_function):
    
    chunks = _split_files(sections_file_path, 
                          num_splits)

    # Create a process for each chunk and run in parallel
    processes = []
    for i, chunk in enumerate(chunks):
        split_file_path = os.path.join(PATH_DATA, f"sections_split_{i}.txt")  
        with open(split_file_path, 'w') as split_file:
            split_file.write("\n".join(chunk))

        process = multiprocessing.Process(target=process_function, 
                                          args=(chunk, 
                                                str(i),
                                                ))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    # Clean up the split files (optional)
    for i in range(num_splits):
        save_news_checkpoint(str(i), "")
        split_file_path = os.path.join(PATH_DATA, f"sections_split_{i}.txt")
        os.remove(split_file_path)

def _split_files(file_path: str, 
                 n_splits: int
                 ):
    # Split the input file into chunks
    sections = pd.read_csv(file_path, 
                           sep=";")
    medias = sections["media"].tolist()
    unique_medias =  list(set(medias))
    chunk_size = len(unique_medias) // n_splits
    end = len(unique_medias)
    remainder = len(unique_medias) % n_splits
    if remainder > 0:
        end -= remainder
    chunks = []
    for i in range(0, end, chunk_size):
        if (i + chunk_size) < end:
            submedias = unique_medias[i:i + chunk_size]
        else:
            submedias = unique_medias[i:i + chunk_size + remainder]
        subchanks = []
        for submedia in submedias:
            temp = sections.loc[sections["media"] == submedia]
            sections_chunk = temp["section"] + ";" + temp["media"]
            subchanks.extend(sections_chunk.tolist())
        chunks.append(subchanks)
    return chunks

def main_multi_threading_process(sections_chunk: list, 
                                 pid: str
                                 ):
    # 1 Initialize reboot or not
    media_checkpoint = read_news_checkpoint(pid)
    if not media_checkpoint:
        checkpoint_started = False
    else:
        checkpoint_started = True
    # 1 End
    media_stats_manager = StatisticsManager().restart_time()
    last_same_media = ""
    same_media_news_urls = []
    n_processed = 0
    n_no_body = 0
    for i, row_section_and_media in enumerate(sections_chunk):
        #media = re.search("(https?://[^/]+/)", section_url).groups()[0]
        section_url, input_media = row_section_and_media.split(";")
        section_url = section_url.strip()
        if not input_media.startswith("https://"):
            input_media = "https://" + input_media
        if checkpoint_started:
            if input_media == media_checkpoint:
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
            input_media = input_media.replace("www.", "")
            print("Request success without 'www'")
        
        parsed_hmtl = BeautifulSoup(response.content, 
                                    "html.parser")
        tags_with_url = parsed_hmtl.find_all("a", 
                                             href=re.compile(r"^(?:https:\/\/)?|^\/{1}[^\/].*|^www[.].*"))
                                             # href=re.compile("https?:.*"))
        news_urls = [x.attrs["href"] for x in tags_with_url if x.attrs.get("href", False)]
        clean_news_url = []
        for url in news_urls:
            if url.startswith("//"):
                url = "https:" + url
            elif url.startswith("/"):
                url = input_media[:-1] + url
            elif url.startswith("www.") or not url.startswith("https://"):
                url = "https://" + url
            clean_news_url.append(url)
        if input_media != last_same_media and last_same_media:
            unique_same_media_urls = list(set(same_media_news_urls))
            n1, n2 = treat_raw_news_urls(unique_same_media_urls, 
                                         last_same_media,
                                         str(pid)
                                         )
            n_processed += n1
            n_no_body += n2
            same_media_news_urls = []
        same_media_news_urls.extend(clean_news_url)
        if not checkpoint_started and (i % N_SAVE_CHECKPOINT == 0):
            save_news_checkpoint(pid, last_same_media)
        last_same_media = input_media
    unique_same_media_urls = list(set(same_media_news_urls))
    n1, n2 = treat_raw_news_urls(unique_same_media_urls,
                                 last_same_media,
                                 str(pid)
                                 )
    n_processed += n1
    n_no_body += n2
    media_stats_manager.write_extraction_stats(("Process summary", 
                                                 n_processed, 
                                                 n_no_body), 
                                                TODAY_LOCAL_DATETIME,
                                                pid="summary"
                                                )
if __name__ == "__main__":
    print(f"\n...Datetime of process: {CURRENT_DATE} {CURRENT_TIME}...\n")
    if FULL_START:
        for i in range(NUM_CORES):
            save_news_checkpoint(str(i), "")

    # Extract the last version
    version_n = max(int(x.split("_")[-1][1:-4]) for x in glob.glob("../data/final_url_sections_v*.csv"))
    sections_file_path = f"final_url_sections_v{version_n}.csv"
    file_path = os.path.join(PATH_DATA, sections_file_path)

    split_file_and_process(file_path, 
                           NUM_CORES, 
                           main_multi_threading_process)
    file_manager.close_all_files()
    print("\n...The process ended...")


