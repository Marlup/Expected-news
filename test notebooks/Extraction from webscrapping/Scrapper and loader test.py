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

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.by import By

# Configure Chrome to run in headless mode
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration, which is necessary for headless mode on some platforms
chrome_options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.1234.567 Safari/537.36"
)
caps = DesiredCapabilities().CHROME
caps["pageLoadStrategy"] = "none"  # Do not wait for full page load
driver = webdriver.Chrome(options=chrome_options)
# Classes
class ErrorReporter(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)
        with open("Errors log.txt", "w") as errors_file:
            errors_file.write(message)
## Constants
SECONDS_IN_DAY = 86_400
N_MAX_DAYS_OLD = 1
FULL_START = True
NUM_CORES = 10
MAX_N_MEDIAS = 50
N_SAVE_CHECKPOINT = 5
# File names
FILE_NAME_INVALID_URLS = "./data/No valid urls"
FILE_NAME_NOT_ARTICLE_URLS = "./data/Not articles urls"
FILE_NAME_TOO_MANY_RETRIES_URLS = "./data/Too many retries urls"
FILE_NAME_PROMPT_ROLE_KEYS = "./data/role prompt_keys extraction.txt"
file_manager = FileManager()
file_manager.add_files([
    FILE_NAME_INVALID_URLS,
    FILE_NAME_NOT_ARTICLE_URLS,
    FILE_NAME_TOO_MANY_RETRIES_URLS
    ])
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
openai.api_key = os.getenv("OPENAI_API_KEY")
with open(FILE_NAME_PROMPT_ROLE_KEYS, "r") as file:
    prompt_role = file.read()
prompt_role_summary_only = """
You are a journalist who follows these guidelines:

1. Write the summary in the third-person point of view.
2. Use the language from the provided text to write the Body Summary.
3. Include important details.

Extract and output the following information from the provided text:

Tokens: [tokens]
Body Summary: [body summary]

Replace [tokens] with the number of tokens of the incoming text.
Extract the information for [body summary] from the provided text.
If the information could not be found then write "N/A" without quotes.
Please stand by for me to prompt the text.
"""
conn = sqlite3.connect("./data/news_db.sqlite3")
cursor = conn.cursor()
scheduled_process = True
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

## Functions
def find_news_body_from_json(html: BeautifulSoup, 
                             data: dict) -> dict:
    data["body"] = None
    data["n_tokens"] = 0
    jsons = html.find_all("script", attrs={"type": re.compile("application[/]{1}ld[+]{1}json")})
    for json_data_str in jsons:
        json_data = json.loads(json_data_str.get_text(), strict=False)
        if isinstance(json_data, list):
            if json_data:
                json_data = json_data[0]
            else:
                continue
        if "articleBody" in json_data:
            body, n_tokens = get_body_summary(remove_body_tags(json_data["articleBody"]))
            data["body"] = body
            data["n_tokens"] = n_tokens
            break
    return data

def extract_data_from_jsons(html: BeautifulSoup, 
                            url: str, 
                            file: str="") -> tuple[dict, dict]:
    data = {}
    keys_found = {}
    keys_found["title"] = False
    keys_found["body"] = False
    keys_found["tags"] = False
    keys_found["type"] = False
    keys_found["creation_datetime"] = False
    keys_found["modified_datetime"] = False
    keys_found["image"] = False
    extraction_completed = False
    data["body"] = None
    data["n_tokens"] = 0
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
            print("JSON error:", e, "\n", url)
        if isinstance(json_data, (list, tuple)):
            if json_data:
                json_data = json_data[0]
        for k, json_values in json_data.items():
            if not keys_found["body"] and "articleBody" in k:
                body, n_tokens = get_body_summary(remove_body_tags(json_data["articleBody"]))
                data["body"] = body
                data["n_tokens"] = n_tokens
                keys_found["body"] = True
            if not keys_found["type"] and ("@type" in k or "type" in k):
                if isinstance(json_values, list):
                    for value in json_data["@type"]:
                        type_value = value.lower()
                        if "media" in type_value:
                            invalid_web = True
                            break
                        if (type_value.startswith("news") or type_value.endswith("article")):
                            keys_found["type"] = True
                            break
                elif isinstance(json_values, str):
                    type_value = json_values.lower()
                    if (type_value.startswith("news") or type_value.endswith("article")) and "media" not in type_value:
                        #print("Value without media?", type_value)
                        keys_found["type"] = True
            if invalid_web:
                print("Invalid web")
                break
            if not keys_found["creation_datetime"] and "datePublished" in k:
                data["creation_datetime"] = json_values
                keys_found["creation_datetime"] = True
            if not keys_found["modified_datetime"] and "dateModified" in k:
                data["modified_datetime"] = json_values
                keys_found["modified_datetime"] = True
            if not keys_found["title"] and "headline" in k:
                data["title"] = json_values
                keys_found["title"] = True
            if not keys_found["tags"] and ("keywords" in k or "tags" in k):
                if json_values:
                    data["tags"] = ";".join(json_values)
                keys_found["tags"] = True
            if all(keys_found.values()):
                extraction_completed = True
                break
    # news_url is not a valid Article
    if not keys_found["type"]:
        #print(f"Bad value from json in url:", url, end="", file=file)
        #print(url, file=file)
        return {"key_value": None}, {}
    return data, keys_found

def extract_data_from_metadata(parsed_html: BeautifulSoup, 
                               data: dict, 
                               keys_found) -> tuple[dict, bool]:
    temp_keys_found = keys_found.copy()
    meta_tags = parsed_html.select("html head meta[property],[name]")
    for meta_tag in meta_tags:
        # Possible attributes:
        # property
        attribute_val = meta_tag.attrs.get("property", False)
        # name
        if not attribute_val:
            attribute_val = meta_tag.attrs["name"]
        attribute_val = attribute_val.lower()
        meta_content = meta_tag.attrs.get("content", None)
        if not temp_keys_found["tags"] and "keyword" in attribute_val:
            data["tags"] = meta_content.replace(", ", ";")
            temp_keys_found["tags"] = True
        if not temp_keys_found["creation_datetime"] and ("publish" in attribute_val and "time" in attribute_val):
            data["creation_datetime"] = meta_content
            temp_keys_found["creation_datetime"] = True
        if not temp_keys_found["modified_datetime"] and ("modif" in attribute_val and "time" in attribute_val):
            data["modified_datetime"] = meta_content
            temp_keys_found["modified_datetime"] = True
        if not temp_keys_found["title"] and "title" in attribute_val:
            data["title"] = meta_content
            temp_keys_found["title"] = True
        if temp_keys_found["image"] and attribute_val.endswith("image"):
            data["image"] = meta_content
            temp_keys_found["image"] = True
        if all([v for k, v in temp_keys_found.items() if "body" not in k]):
            break
    return data, temp_keys_found["title"]

def extract_keys_with_gpt(parsed_code: BeautifulSoup) -> dict:
    print("..Keys through gpt..")
    tags_with_text = parsed_code.find_all(lambda tag: (tag.name == "p" and not tag.attrs) or ("h" in tag.name and not tag.attrs))
    text_clean_from_tags = "".join([re.sub("\n+", "\n", tag.get_text()) for tag in tags_with_text])
    text_clean_from_tags = text_clean_from_tags.split("\n")[0]
    openai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_role},
            {"role": "user", "content": text_clean_from_tags},
        ]
    )
    message_content = openai_response["choices"][0].message["content"]
    try:
        n_tokens = regex_n_tokens.search(message_content).groups()[0]
    except:
        n_tokens = 0
    try:
        title = regex_headline.search(message_content).groups()[0]
    except:
        title = None
    try:
        tags = regex_topics.search(message_content).groups()[0]
    except:
        tags = None
    try:
        creation_datetime = regex_creation_datetime.search(message_content).groups()[0]
    except:
        creation_datetime = None
    try:
        modified_datetime = regex_update_datetime.search(message_content).groups()[0]
    except:
        modified_datetime = None
    try:
        body = regex_only_summary.search(message_content).groups()[0]
    except:
        body = None
    data = {
        "n_tokens": n_tokens,
        "title": title,
        "tags": tags,
        "creation_datetime": creation_datetime,
        "modified_datetime": modified_datetime,
        "body": body
        }
    data["image"] = None
    return data

def remove_body_tags(text: str) -> str:
    return re.sub("<.*?>", "", text)

def find_news_body_with_gpt(url: str) -> dict:
    print("..Body through gpt..")
    driver.get(url)
    driver.find_elements(By.XPATH, "html/body//p|h1|h2")
    paragraphs = driver.find_elements(By.XPATH, "html/body//div/p|h1|h2")
    parag_texts = str({i: x.text if x.text else "\n" for i, x in enumerate(paragraphs)})[1:-1]
    body, n_tokens = get_body_summary(parag_texts)

    data = {
        "n_tokens": n_tokens,
        "body": body
    }
    return data

def get_body_summary(text: str) -> tuple[str, str]:
    openai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_role_summary_only},
            {"role": "user", "content": text},
        ]
    )
    message_content = openai_response["choices"][0].message["content"]
    try:
        n_tokens = regex_n_tokens.search(message_content).groups()[0]
    except:
        n_tokens = 0
    try:
        body_summary = regex_only_summary.search(message_content).groups()[0]
    except:
        body_summary = None
    return body_summary, n_tokens

def treat_raw_news_urls(news_urls: list, 
                        media_url: str,
                        pid: str,
                        lock: multiprocessing.Lock
                        ):
    
    media_stats_reporter = StatisticsReporter()
    media_stats_reporter.restart_time()
    urls = find_valid_news_urls(news_urls,
                                media_url,
                                pid,
                                lock)
    select_urls = pd.DataFrame(read_stored_news(media_url), 
                               columns=["url2"])
    extracted_urls = pd.DataFrame(urls, 
                                  columns=["url1"]
                                  )
    
    are_new_news = extracted_urls.merge(select_urls, 
                                        left_on="url1", 
                                        right_on="url2", 
                                        how="left").isnull().any(axis=1)
    # Merge extracted and loaded news in order to process the new ones
    news_to_process = extracted_urls.loc[are_new_news, "url1"].tolist()
    # Process new news
    processed_news_data, n_no_body = find_news_data(news_to_process, 
                                                    author=media_url,
                                                    pid=pid,
                                                    lock=lock
                                                    ) # unordered
    n_processed = len(processed_news_data)
    processed_news_data = order_dict_keys(processed_news_data, 
                                          ("title", 
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
                                           ) # ordered
    insert_news(processed_news_data)
    # Statistics
    media_stats_reporter.write_extraction_stats((media_url, 
                                                 n_processed, 
                                                 n_no_body), 
                                                pid=pid,
                                                input_date=CURRENT_DATE,
                                                input_time=CURRENT_TIME
                                                )
    if n_processed > 0:
        print(f"\tmedia name: {media_url} processed counts: {str(n_processed)}")
    return n_processed, n_no_body

def find_valid_news_urls(news_urls: list, 
                         media_url: str, 
                         pid: str,
                         lock: multiprocessing.Lock
                         ) -> (list, bool):
    valid_news_urls = []
    #with open(FILE_NAME_INVALID_URLS + f"_{pid}" + ".txt", "w") as file_invalid:
    for news_url in news_urls:
        # Filter out urls with query symbols
        if re.search("[?=&+%@#]{1}", news_url):
            search_spec_char = re.search("[?=&+%@#]{1}", news_url)
            query_start = search_spec_char.span()[0]
            news_url = news_url[:query_start]
        if news_url == media_url:
            #print(f"{news_url};find_valid_news_urls();continue_1", 
            #        file=file_invalid)
            file_manager.write_on_file(FILE_NAME_INVALID_URLS, 
                                       [f"{news_url};find_valid_news_urls();continue_1"], 
                                       lock, 
                                       pid)
            continue
        if  news_url.endswith(".xml") \
            or news_url.endswith(".pdf") \
            or news_url.endswith(".lxml") \
            or news_url.endswith(".jpg") \
            or news_url.endswith(".png") \
            or news_url.endswith(".gif"):
            #print(f"{news_url};find_valid_news_urls();continue_2", 
            #        file=file_invalid)
            file_manager.write_on_file(FILE_NAME_INVALID_URLS, 
                                       [f"{news_url};find_valid_news_urls();continue_2"], 
                                       lock, 
                                       pid)
            continue
        if "pagina-1.html" in news_url or "/firmas" in news_url or "/humor/" in news_url or "/autor" in news_url or "/autores/" in news_url:
            #print(f"{news_url};find_valid_news_urls();continue_3", 
            #        file=file_invalid)
            file_manager.write_on_file(FILE_NAME_INVALID_URLS, 
                                       [f"{news_url};find_valid_news_urls();continue_3"], 
                                       lock, 
                                       pid)
            continue
        news_url_splits = [x for x in news_url.split("/") if x][2:]
        if len(news_url_splits) <= 1:
            #print(f"{news_url};find_valid_news_urls();continue_4", 
            #        file=file_invalid)
            file_manager.write_on_file(FILE_NAME_INVALID_URLS, 
                                       [f"{news_url};find_valid_news_urls();continue_4"], 
                                       lock, 
                                       pid)
            continue
        elif len(news_url_splits) >= 2:
            if re.search(r"^[.a-zA-Z0-9]+(-[.a-zA-Z0-9]+){,2}$", news_url_splits[-1]):
                #print(f"{news_url};find_valid_news_urls();continue_5", 
                #        file=file_invalid)
                file_manager.write_on_file(FILE_NAME_INVALID_URLS, 
                                       [f"{news_url};find_valid_news_urls();continue_5"], 
                                       lock, 
                                       pid)
                continue
        if media_url[:-1] not in news_url:
            #print(f"{news_url};find_valid_news_urls();continue_6", 
            #        file=file_invalid)
            file_manager.write_on_file(FILE_NAME_INVALID_URLS, 
                                       [f"{news_url};find_valid_news_urls();continue_6"], 
                                       lock, 
                                       pid)
            continue
        if re.search("https?:[\/]{2}", news_url):
            #if news_url.endswith(".ht"):
            #    news_url = news_url + "ml"
            #elif news_url.endswith(".htm"):
            #    news_url = news_url + "l"
            valid_news_urls.append(news_url)
    return list(set(valid_news_urls))

def find_news_data(news_urls: list, 
                   author: str,
                   pid: str,
                   lock: multiprocessing.Lock,
                   order_keys: bool=True
                   ):
    news_media_data = []
    n_no_articlebody_in_article = 0
    #with open(FILE_NAME_NOT_ARTICLE_URLS + f"_{pid}" + ".txt", "w") as file_no_articles, \
    #     open(FILE_NAME_TOO_MANY_RETRIES_URLS + f"_{pid}" + ".txt", "w") as file_too_many_retries:
    for news_url in news_urls:
        #print("Finding data for url:", f"(author: {author}) ", news_url)
        #continue
        try:
            resp_url_news = requests.get(news_url, 
                                         headers=HEADERS)
        except requests.exceptions.TooManyRedirects:
            #file_too_many_retries.write(news_url)
            file_manager.write_on_file(FILE_NAME_TOO_MANY_RETRIES_URLS, 
                                       [news_url], 
                                       lock, 
                                       pid)
        except requests.exceptions.RequestException as e:
            print("An error occurred:", e)
            #file_too_many_retries.write(news_url)
            file_manager.write_on_file(FILE_NAME_TOO_MANY_RETRIES_URLS, 
                                       [news_url], 
                                       lock, 
                                       pid)
        
        parsed_news_hmtl = BeautifulSoup(resp_url_news.content, 
                                        "html.parser")
        # Accept or reject url if news date is more than N_MAX_DAYS_OLD days older
        try:
            meta_tag_published_time = parsed_news_hmtl.html.head.find("meta", 
                                                                      attrs={"property": re.compile(r"publish(?:ed)?_?(?:time|date)")})
            if meta_tag_published_time is None:
                print("Time difference not calculated", meta_tag_published_time, ";", news_url)
                continue
            publ_tsm = meta_tag_published_time.attrs["content"]
            dtime_diff = (TODAY_LOCAL_DATETIME - datetime.fromisoformat(publ_tsm)).total_seconds() / SECONDS_IN_DAY
            if dtime_diff > N_MAX_DAYS_OLD:
                continue
        except:
            print("Time difference not calculated", meta_tag_published_time, ";", news_url)
            pass
        data = {}
        data["creation_datetime"] = ""
        data["modified_datetime"] = ""
        
        #try:
        extracted_data, keys_found = extract_data_from_jsons(parsed_news_hmtl, 
                                                             news_url
                                                             #file_no_articles
                                                            )
        #print("from jsons:", extracted_data)
        # news_url is not a valid Article
        if not keys_found:
            continue
        data.update(extracted_data)
        extracted_data, title_found = extract_data_from_metadata(parsed_news_hmtl, 
                                                                 data, 
                                                                 keys_found)
        #print("from metadata:", extracted_data)
        # @AUTHOR Update is not done on existing keys. The dict of filters, 'keys_found'
        # avoids extraction of already found keys
        data.update(extracted_data)
        if not title_found:
            print("No title", news_url)
            continue
        if not keys_found["body"]:
            print("No body", news_url)
            n_no_articlebody_in_article += 1
            #continue
            data.update(find_news_body_with_gpt(news_url))
        #except:
            #pass
            #print("what happened", news_url)
            #keys_found = {}
            #keys_found["title"] = False
            #keys_found["body"] = False
                #data = extract_keys_with_gpt(parsed_news_hmtl)
        # Skip data element if neither title nor article were found, i.e. None value on both.
        if data is None or (not title_found or not keys_found["body"]):
            print("nothing")
            continue
        #print("Ok", news_url)
        #if not keys_found["tags"]:
        #    elements_with_tags = parsed_news_hmtl.find_all("meta", 
        #                                                   attrs={"name": "news_keywords"})
        #    data["tags"] = [x.attrs["content"] for x in elements_with_tags]
        #    data["tags"] = ";".join(data["tags"])
        data["country"] = parsed_news_hmtl.html.attrs.get("lang", "")
        # TODO complete this
        data["source"] = author
        data["url"] = news_url
        news_media_data.append(data)
    if order_keys:
        news_media_data = order_dict_keys(news_media_data, 
                                          ORDER_KEYS
                                          ) 
    return news_media_data, n_no_articlebody_in_article

def order_dict_keys(data_container: list, 
                    ord_keys: tuple, 
                    only_values: bool=True):
    if only_values:
        return [tuple(data.get(k, "") for k in ord_keys) for data in data_container]
    else:
        return [{k: data.get(k, "") for k in ord_keys} for data in data_container]
    
def read_stored_news(where_params):
    create_news_table()
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
    return output

def create_news_table():
    query_str = """
        CREATE TABLE IF NOT EXISTS News (
            title TEXT NOT NULL,
            article TEXT NOT NULL,
            source TEXT NOT NULL,
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

def insert_news(data):
    create_news_table()
    query_str = """
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
                datetime('now','localtime'),
                '',
                ?
            )
            ;
        """
    #print("data:\n", data)
    cursor.executemany(query_str, data)

def split_file_and_process(sections_file_path: str, 
                           num_splits: int, 
                           process_function):
    
    lock = multiprocessing.Lock()
    chunks = _split_files(sections_file_path, 
                          num_splits)

    # Create a process for each chunk and run in parallel
    processes = []
    for i, chunk in enumerate(chunks):
        split_file_path = os.path.join("data", f"sections_split_{i}.txt")  
        with open(split_file_path, 'w') as split_file:
            split_file.write("\n".join(chunk))

        process = multiprocessing.Process(target=process_function, 
                                          args=(chunk, 
                                                str(i), 
                                                lock
                                                ))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    # Clean up the split files (optional)
    for i in range(num_splits):
        save_news_checkpoint(str(i), "")
        split_file_path = os.path.join("data", f"sections_split_{i}.txt")
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
                                 pid: str,
                                 lock: multiprocessing.Lock
                                 ):
    # 1 Initialize reboot or not
    media_checkpoint = read_news_checkpoint(pid)
    if not media_checkpoint:
        checkpoint_started = False
    else:
        checkpoint_started = True
    # 1 End
    media_stats_reporter = StatisticsReporter()
    media_stats_reporter.restart_time()
    last_same_media = ""
    same_media_news_urls = []
    n_processed = 0
    n_no_body = 0
    for i, row_section_and_media in tqdm(enumerate(sections_chunk)):
        #media = re.search("(https?://[^/]+/)", section_url).groups()[0]
        section_url, media = row_section_and_media.split(";")
        section_url = section_url.strip()
        if not media.startswith("https://"):
            media = "https://" + media
        if checkpoint_started:
            if media == media_checkpoint:
                checkpoint_started = False
            else:
                continue

        try:
            response = requests.get(section_url, 
                                    headers=HEADERS, 
                                    timeout=10)
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
                                    timeout=10)
            #ErrorReporter(f"Get request general exception, {e}, at {section_url}")
            media = media.replace("www.", "")
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
                url = media[:-1] + url
            elif url.startswith("www.") or not url.startswith("https://"):
                url = "https://" + url
            clean_news_url.append(url)
        if media != last_same_media and last_same_media:
            unique_same_media_urls = list(set(same_media_news_urls))
            n1, n2 = treat_raw_news_urls(unique_same_media_urls, 
                                         last_same_media,
                                         str(pid),
                                         lock
                                         )
            n_processed += n1
            n_no_body += n2
            #same_media_urls = []
        same_media_news_urls.extend(clean_news_url)
        if not checkpoint_started and (i % N_SAVE_CHECKPOINT == 0):
            save_news_checkpoint(pid, last_same_media)
        last_same_media = media
    unique_same_media_urls = list(set(same_media_news_urls))
    n1, n2 = treat_raw_news_urls(unique_same_media_urls,
                                 last_same_media,
                                 str(pid),
                                 lock
                                 )
    n_processed += n1
    n_no_body += n2
    media_stats_reporter.write_extraction_stats(("Process summary", 
                                                 n_processed, 
                                                 n_no_body), 
                                                pid="summary",
                                                input_date=CURRENT_DATE,
                                                input_time=CURRENT_TIME
                                                )

if __name__ == "__main__":
    print(f"\n...Datetime of process: {CURRENT_DATE} {CURRENT_TIME}...\n")
    if FULL_START:
        for i in range(NUM_CORES):
            save_news_checkpoint(str(i), "")

    # Extract the last version
    version_n = max(int(x.split("_")[-1][1:-4]) for x in glob.glob("./data/final_url_sections_v*.csv"))
    sections_file_path = f"final_url_sections_v{version_n}.csv"
    file_path = os.path.join(".", "data", sections_file_path)

    split_file_and_process(file_path, 
                           NUM_CORES, 
                           main_multi_threading_process)
    conn.commit()
    file_manager.close_all_files()
    print("\n...The process ended...")