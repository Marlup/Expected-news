import logging
import os
import re
import json
import time
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
import sqlite3
import multiprocessing as mp

import openai
import pandas as pd

from main_utils import (
    direct_recursive_destructure,
    remove_body_tags,
    clean_topics,
    StatisticsManager,
    read_garbage,
    find_invalid_files,
    find_invalid_sections,
    get_last_sections_file_num,
)

from constants import *

# Dates and times
TODAY_LOCAL_DATETIME = datetime.now().replace(tzinfo=timezone.utc)
dt = datetime.today() \
             .strftime(f"%Y %m %d_%H %M %S")
CURRENT_DATE, CURRENT_TIME = str(dt).split("_")

# Logging
#logger_bad_url = logging.getLogger(LOGGER_NAME_BAD_URL)
logging.basicConfig(filename=f"data/logs/bad_urls/bad_urls_{CURRENT_DATE}_{CURRENT_TIME[:3]}.log", 
                    filemode="w",
                    level=logging.INFO, 
                    format="%(asctime)s,%(funcName)s,%(lineno)d,%(message)s",
                    datefmt="%H:%M:%S",
                    force=True,
                    )
# Get API_KEY
if not BLOCK_API_CALL:
    openai.api_key = os.getenv("OPENAI_API_KEY")
# Get LLM role
with open(FILE_PATH_PROMPT_ROLE_SUMMARY, "r") as file:
    PROMPT_ROLE_SUMMARY = file.read()

# Compiled regular expressions
regex_url_has_query = re.compile(r"[?!)(=&+%@#]{1}")
regex_valid_url_format = re.compile(r"^(?:https:\/\/)?|^\/{1}[^\/].*|^www[.].*")
regex_too_short_url_end = re.compile(r"^[.a-zA-Z0-9]+(-[.a-zA-Z0-9]+){,2}$")
regex_url_startswith_https = re.compile(r"https?:[\/]{2}")
#regex_article_content = re.compile(r"(?_)publish(?:ed)?_?(?:time|date)")
regex_publication_ts = re.compile(r"publish(?:ed)?_?(?:time|date)")
regex_modification_ts = re.compile(r"modif(?:ied)?_?(?:time|date)")
regex_application_jsons = re.compile("application[/]{1}ld[+]{1}json")
regex_title = re.compile(r"title|headline|titular|titulo")
regex_body = re.compile(r"articleBody")
regex_date_creation = re.compile(r"date.*[pP]ub.*")
regex_date_modified = re.compile(r"date.*[mM]od.*")
regex_tags = re.compile(r"tag[s]?|topic[s]?|tema[s]?|etiqueta[s]?|keyword[s]?")
regex_n_tokens = re.compile(r"Tokens: (\d+).*", flags=re.DOTALL)
regex_headline = re.compile(r"Headline: (.*).*", flags=re.DOTALL)
regex_topics = re.compile(r"Topics: (.*).*", flags=re.DOTALL)
regex_creation_datetime = re.compile(r"Creation DateTime: (.*).*", 
                                     flags=re.DOTALL
                                     )
regex_update_datetime = re.compile(r"Update DateTime: (.*).*Body Summary", 
                                   flags=re.DOTALL
                                   )
regex_only_summary = re.compile(r"Body Summary:(.*)", flags=re.DOTALL)

# OpenAI client variable
assistant_client = None
assistant = None

## Decorators
def garbage_logger():
    def decorator(target_func):
        def wrapper(*args, **kwargs):
            output_data, garbage = target_func(*args, **kwargs)
            if len(garbage) > 0:
                lock = mp.Lock()
                with lock:
                    insert_data(garbage, INSERT_GARBAGE_QUERY)
            return output_data
        return wrapper
    return decorator

## Functions
def add_new_log_line(message):
    logging.info(message)

def search_news_keys_from_jsons(
    html: BeautifulSoup, 
    url: str,
    media: str,
    only_return_body: bool=True 
    ) -> dict:
    """
    Extracts and returns structured data from JSON scripts within an HTML 
    document.

    Args:
        html (BeautifulSoup): Object representing the parsed HTML content.
        url (str): URL of the web page from which data is being extracted.
        media (str): Type of media (e.g., "news," "article") for contextual 
        validation.
        only_return_body (bool, optional): If True, the function will return 
        only the body content. Default is True.

    Returns:
        dict: Extracted data fields are as follows:
            - The first dictionary includes the following keys (if found):
                - "title": The title of the web page.
                - "description": A brief description of the web page.
                - "body": The main content of the web page.
                - "creation_datetime": The date when the content was published.
                - "modified_datetime": The date when the content was last 
                  modified.
                - "main_topic": The main topic or category of the content.
                - "other_topic": Additional topics or tags associated with the
                  content.
                - "n_tokens": The number of tokens in the extracted body 
                  content.

        If the "@type" or "type" field is not found in the JSON data, an empty 
        dictionary is returned.
    """
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
    
    jsons = html.find_all("script", attrs={"type": regex_application_jsons})
    
    for json_data_str in jsons:
        if invalid_web or extraction_completed:
            break
        json_text = json_data_str.get_text()
        
        if json_data_str is None or not isinstance(json_text, str):
            continue
        
        try:
            json_data = json.loads(json_text, 
                                   strict=False)
            if isinstance(json_data, (dict, )):
                json_data = [json_data]
        except Exception as e:
            continue
        
        for sub_json_data in json_data:
            _, destructured_keys_and_values, _, _ = direct_recursive_destructure(sub_json_data)
    
            for (key, value) in destructured_keys_and_values:
                if not type_found and ("@type" in key or "type" in key):
                    type_value = value.lower()
                    if "media" in type_value:
                        invalid_web = True
                        break
                    elif (type_value.startswith("news") or type_value.endswith("article")):
                        type_found = True
                        continue
                if not body_found and "articleBody" in key:
                    body_found = True
                    body, n_tokens = generate_body_summary_from_completion_api(
                        remove_body_tags(value), 
                        url,
                        media
                        )
                    data_extracted["body"] = body
                    data_extracted["n_tokens"] = n_tokens
                
                if only_return_body and type_found and body_found:
                    extraction_completed = True
                    continue
                if only_return_body:
                    continue
                if not description_found and "description" in key:
                    description_found = True
                    data_extracted["description"] = value
                if not creation_datetime_found and "datePublished" in key:
                    creation_datetime_found = True
                    data_extracted["creation_datetime"] = value
                if not modified_datetime_found and "dateModified" in key:
                    modified_datetime_found = True
                    data_extracted["modified_datetime"] = value
                if not title_found and "headline" in key:
                    title_found = True
                    data_extracted["title"] = value
                if not main_topic_found:
                    if ("articleSection" in key and "list" not in value) or ("list" in key):
                        main_topic_found = True
                        if "list" in key:
                            print("keywords in a list")
                            if isinstance(value, (list, tuple)):
                                data_extracted["main_topic"] = ",".join(value).lower()
                            else:
                                data_extracted["main_topic"] = value.lower()
                        else: # from 'articleSection' in key
                            data_extracted["main_topic"] = value.lower()
                if not other_tag_found and ("keywords" in key or "tags" in key):
                    if not value:
                        continue
                    topics = clean_topics(value)
                    if topics:
                        other_tag_found = True
                        data_extracted["other_topic"] = topics
                    else:
                        continue
                    
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

def search_news_keys_from_metadata(
    parsed_html: BeautifulSoup, 
    data_input: dict
    ) -> dict:
    """
    Extracts metadata from HTML meta tags and in case data_input does not contain any.

    Args:
        parsed_html (BeautifulSoup): Object representing the parsed HTML content.
        data_input (dict): Contains data input, which can influence the metadata extraction process.

    Returns:
        dict: Extracted data fields are as follows:
            - The first dictionary includes the following keys (if found):
                - "title": The title of the web page.
                - "description": A brief description of the web page.
                - "creation_datetime": The date when the content was published.
                - "modified_datetime": The date when the content was last modified.
                - "main_topic": The main topic or category of the content.
                - "other_topic": Additional topics or tags associated with the content.
                - "image_url": URL of an associated image.

    Notes:
    - The metadata is extracted based on specific attribute values ("property," "name") and keywords.
    - If a piece of metadata is not found, it is not included in the output dictionary.
    - The function checks for the presence of data in the 'data_input' dictionary and includes it in the output if available.
    """
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
        if not creation_datetime_found and regex_publication_ts.search(attribute_val):
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
        if not modified_datetime_found and regex_modification_ts.search(attribute_val):
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


def call_completion_api(text: str, prompt: str) -> str:
    """
    Generates a chat response using OpenAI's GPT-3.5 Turbo model by using a 
    conversation context consisting of a system message and a user message.

    Args:
        prompt (str): A system-level instruction or initial message for the 
        chat conversation.
        text (str): User's input or continuation of the conversation.

    Returns:
        str: The generated chat response provided by the GPT-3.5 Turbo model.

    """

    openai_response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", 
            "content": prompt},
        {"role": "user", 
            "content": text},
            ]
    )
    return openai_response["choices"][0].message["content"]

def set_or_create_assistant() -> bool:
    def _try_set_assistant_role():
        try:   
            assistant_client = openai.OpenAI()
            
            assistant = assistant_client.beta.assistants.create(
                instructions=PROMPT_ROLE_SUMMARY,
                model="gpt-4-1106-preview",
                tools=[{"type": "code_interpreter"}],
                file_ids=[]
                )
            return assistant
        except:
            return None
    
    return _try_set_assistant_role()

def run_assistant(text: str) -> str:
    thread = assistant_client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": text,
                "file_ids": []
            }
                ]
        )
    
    run = assistant_client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID
        )

def search_news_keys_from_completion_api(parsed_code: BeautifulSoup) -> dict:
    """
    Extracts structured data from parsed HTML content using BeautifulSoup.

    Args:
        parsed_code (BeautifulSoup): A BeautifulSoup object representing the 
        parsed HTML content.

    Returns:
        dict: A dictionary containing the extracted structured data, including:
            - "n_tokens": The number of tokens in the extracted content.
            - "title": The title of the content.
            - "other_topic": Additional topics or tags associated with the 
            content.
            - "creation_datetime": The date when the content was created or 
            published.
            - "modified_datetime": The date when the content was last updated.
            - "body": The main content or summary of the HTML content.
            - "image_url": URL of an associated image (initialized as an empty
            string).

    Notes:
    - This function extracts structured data from specific HTML tags 
    (e.g., <p>, <h1>, <h2>) within the parsed HTML content.
    - It cleans and processes the text content of these tags.
    - The function then utilizes regular expressions and an external function 
    (call_completion_api) to extract specific data elements.
    - If any of the data elements are not found, they are set to default or 
    empty values.
    - The extracted data is returned as a dictionary.

    """
    tags_with_text = parsed_code.find_all(lambda tag: tag.name in ("p", "h1", "h2"))
    text_clean_from_tags = "".join([re.sub("\n+", "\n", tag.get_text()) for tag in tags_with_text])
    try:
        message_content = call_completion_api(text_clean_from_tags, 
                                              PROMPT_ROLE_SUMMARY)
        try:
            n_tokens = regex_n_tokens.search(message_content) \
                                     .groups()[0]
        except:
            n_tokens = -1
        try:
            title = regex_headline.search(message_content) \
                                  .groups()[0]
        except:
            title = ""
        try:
            tags = regex_topics.search(message_content) \
                               .groups()[0]
        except:
            tags = ""
        try:
            creation_datetime = regex_creation_datetime.search(message_content) \
                                                       .groups()[0]
        except:
            creation_datetime = ""
        try:
            modified_datetime = regex_update_datetime.search(message_content) \
                                                     .groups()[0]
        except:
            modified_datetime = ""
        try:
            body = regex_only_summary.search(message_content) \
                                     .groups()[0]
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

def get_body_summary(parsed_html: BeautifulSoup, 
                             url: str,
                             media: str
                             ) -> dict:
    """
    Extracts article body summarization directly from within parsed HTML.

    Args:
        parsed_html (BeautifulSoup): Object representing the parsed HTML 
        content.
        url (str): URL of the web page associated with the text content.
        media (str): The type of media (e.g., "news," "article") for contextual
        validation.

    Returns:
        dict: A dictionary containing the extracted structured data, including:
            - "n_tokens": The number of tokens in the extracted text content.
            - "body": The main content or summary of the text content.

    Notes:
    - This function extracts structured data from specific HTML tags 
      (e.g., <p>) within the parsed HTML content.
    - It cleans and processes the text content of these tags.
    """

    tags_with_text = parsed_html.find_all(lambda tag: tag.name in ("p", ))
    text_clean_from_tags = "".join([re.sub("\n+", "\n", tag.get_text()) for tag in tags_with_text])
    #clean_paragraphs = clean_paragraphs.split("\n")[0]
    #parag_texts = str({i: x if x else "\n" for i, x in enumerate(clean_paragraphs)})[1:-1]
    body, n_tokens = generate_body_summary_from_completion_api(
        text_clean_from_tags, 
        url,
        media
        )
    data = {
        "n_tokens": n_tokens,
        "body": body
    }
    return data

@garbage_logger()
def generate_body_summary_from_completion_api(
    text: str, 
    url: str,
    media: str
    ) -> tuple[tuple[str, str | int] | dict[str, bool | list]]:
    """
    Generates a summary for text content using an external API.

    Args:
        text (str): The text content to be summarized.
        url (str): The URL of the source associated with the text content.
        media (str): The type of media (e.g., "news," "article") for contextual
        validation.

    Returns:
        tuple[str, str | int] | dict[str, bool | list]: A tuple containing two
        elements:
            - The first element is the generated summary for the text content.
            - The second element is the number of tokens in the generated
              summary. 
              a summary is not found or an error occurs -1 tokens are returned

    """
    if BLOCK_API_CALL:
        #return (text, -1), {"status_code": STATUS_0, "id": ""}
        return (text, -1), []
    try:
        message_content = call_completion_api(text, PROMPT_ROLE_SUMMARY)
    except Exception as e:
        #return ("", 0), {"status_code": STATUS_4, "id": url}
        return ("", 0), [(url, media, STATUS_4)]
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
    return (body_summary, n_tokens), []

def process_raw_urls(
    raw_urls: list, 
    media_url: str,
    score: float
    ) -> tuple[int, int]:
    """
    Processes a list of raw URLs, extracts and stores news data, and provides 
    statistics.

    Args:
        raw_urls (list): A list of raw URLs to be processed.
        media_url (str): The URL associated with the media source.
        score (float): A score or rating for processing URLs.

    Returns:
        tuple[int, int]: A tuple containing two integers:
            - The first integer represents the number of processed news 
              articles.
            - The second integer represents the number of news articles with 
              missing body content.

    """
    # Function constants
    warning_message = ""
    writer_pid = os.getpid()

    media_stats_manager = StatisticsManager(start_time=True, log_dir=PATH_STATS)
    novel_urls = filter_gargabe_urls(raw_urls, media_url)
    valid_scraped_urls = validate_raw_urls(novel_urls, media_url)
    
    if len(valid_scraped_urls) > 0:
        lock = mp.Lock()
        with lock:
            db_fetched_urls = pd.DataFrame(
                read_news_for_media(media_url), 
                columns=["from_db"]
                                     )
        
        scraped_urls = pd.DataFrame(
            valid_scraped_urls, 
            columns=["from_scraping"]
            ).drop_duplicates()
        
        novel_news = db_fetched_urls.merge(
            scraped_urls, 
            left_on="from_db", 
            right_on="from_scraping", 
            how="right"
            )
        
        # Merge extracted and loaded news to obtain new ones
        news_to_process = novel_news.loc[novel_news.from_db.isnull(), 
                                         "from_scraping"] \
                                    .tolist()
        
        # Process new news
        news_data, n_no_body = search_news_keys_from_valid_urls(
            news_to_process, 
            media_url=media_url,
            score=score,
            order_keys=True
            )
        
        #print("Data:", news_data)
        if news_data:
            lock = mp.Lock()
            with lock:
                insert_data(news_data, query=INSERT_NEWS_QUERY)
                #print("Rows inserted:", len(news_data), "\n")
                
        # Statistics
        n_processed = len(news_data)
        n_fetch_from_db = len(db_fetched_urls)
        n_scrap_from_web = len(scraped_urls)
        n_to_processing = len(news_to_process)
        n_insertions_to_db = len(news_data)

        # Set a warning if there are news to process but no new news to insert
        if n_insertions_to_db == 0:
            warning_message = "0 insertions to db"
        
        print(f"""
        Summary of PID {writer_pid} at media {media_url}:
            · URLs fetched from db:        {n_fetch_from_db}
            · URLs scraped from web:       {n_scrap_from_web}
            · URLs to processing:          {n_to_processing}
            · URLs sent to db:             {n_insertions_to_db}
        """)
    else:
        n_processed = 0
        n_no_body = 0
    
    stats_for_log = (media_url, n_processed, n_no_body, warning_message)
    
    media_stats_manager.write_stats(stats_for_log, writer_id=writer_pid)
    
    return n_processed, n_no_body

def filter_gargabe_urls(scraping_urls: list, media_url: str) -> tuple[list, dict]:
    """
    Filters list of input URLs, avoiding known garbage URLs.

    Args:
        scraping_urls (list): URLs to be filtered.
        media_url (str): URL associated with the media source.

    Returns:
        list: Curated URLS.
    """

    n_urls = len(scraping_urls)
    columns_db = ["url_from_db", "media_url_from_db"]
    columns_scraping = ["url_from_scraping", "media_url_from_scraping"]

    # Open file of garbage urls to avoid
    lock = mp.Lock()
    with lock:
        db_data = pd.DataFrame(
            read_garbage((media_url, )), 
            columns=columns_db
            )
    
    # Create a DataFrame of URLs to compare against
    scraping_data = pd.DataFrame(
        dict(
            url_from_scraping=scraping_urls, 
            media_url_from_scraping=n_urls * [media_url]
            )
        ).drop_duplicates()
    
    # Urls from garbage table query
    novel_urls_series = scraping_data.merge(
        db_data,
        left_on=columns_scraping,
        right_on=columns_db,
        how="left",
        indicator=True
        ).query('_merge == "left_only"')["url_from_scraping"] \
         .tolist()
    
    # Merge extracted and loaded news in order to process the new ones
    #urls_to_process = novel_urls.loc[novel_urls[columns_db].isnull().all(axis=1), "url_from_scraping"]
    #print("garbage_urls:", len(db_data), "input urls:", len(input_urls), "urls_to_process:", len(urls_to_process))

    return novel_urls_series

@garbage_logger()
def validate_raw_urls(urls: list, media_url: str) -> tuple[list, dict]:
    """
    Filters and validates a list of input URLs, avoiding known garbage URLs.

    Args:
        urls (list): URLs to be filtered and validated.
        media_url (str): URL associated with the media source.

    Returns:
        tuple[list, dict]: A tuple containing two elements:
            - The first element is a list of validated and filtered URLs.
            - The second element is a dictionary providing information about 
              garbage URLs.
    """

    garbage = []
    valid_urls = []
    
    for url in urls:
        # Filter out urls with query symbols
        if url == media_url:
            #status_and_id.append({"status_code": STATUS_3_1, "id": url})
            garbage.append((url, media_url, STATUS_3_1))
            continue
        if find_invalid_files(url):
            #status_and_id.append({"status_code": STATUS_3_2, "id": url})
            garbage.append((url, media_url, STATUS_3_2))
            continue
        if find_invalid_sections(url):
            #status_and_id.append({"status_code": STATUS_3_3, "id": url})
            garbage.append((url, media_url, STATUS_3_3))
            continue
        
        url_splits = [x for x in url.split("/") if x][2:]
        if len(url_splits) <= 1:
            #status_and_id.append({"status_code": STATUS_3_4, "id": url})
            garbage.append((url, media_url, STATUS_3_4))
            continue
        elif len(url_splits) >= 2:
            if regex_too_short_url_end.search(url_splits[-1]):
                #status_and_id.append({"status_code": STATUS_3_5, "id": url})
                garbage.append((url, media_url, STATUS_3_5))
                continue
        if media_url[:-1] not in url:
            #status_and_id.append({"status_code": STATUS_3_6, "id": url})
            garbage.append((url, media_url, STATUS_3_6))
            continue
        if regex_url_startswith_https.search(url):
            valid_urls.append(url)
        
    #print(f"\t{media_url}; new garbage urls {len(garbage)};", f"Treat {len(valid_urls)} urls\n")
    #cache = {"status_code": status_and_id}
    unique_urls = list(set(valid_urls))
    if garbage:
        return unique_urls, garbage
    else:
        return unique_urls, garbage

@garbage_logger()
def search_news_keys_from_valid_urls(
    urls: list, 
    media_url: str,
    score: float,
    order_keys=True
    ) -> tuple[tuple[list, int], dict]:
    """
    Extracts news data from a list of URLs associated with a media source.

    Args:
        urls (list): A list of potential news URLs to be processed.
        media_url (str): The URL associated with the media source.
        score (float): A score or rating for processing URLs.
        order_keys (bool, optional): If True, the keys in the extracted data are ordered.
            Default is True.

    Returns:
        tuple[tuple[list, int], dict]: A tuple containing two elements:
            - The first element is a tuple with two components:
                - A list of extracted news data.
                - An integer representing the number of articles with missing body content.
            - The second element is a dictionary providing information about garbage URLs.

    """
    news_data = []
    garbage_urls = []
    n_no_articlebody_in_article = 0
    for url in urls:
        data, code = _search_keys_from_one_news_url(
            url, 
            media_url,
            score,
            order_keys
            )
        
        if not data:
            garbage_urls.append(
                (
                    url, 
                    media_url, 
                    code
                    )
                )
            if code == STATUS_5_6:
                n_no_articlebody_in_article += 1
            continue
        news_data.append(data)

    print(f"Number of garbage urls found at {media_url}:", len(garbage_urls))
    
    if garbage_urls:
        return (news_data, n_no_articlebody_in_article), garbage_urls
    else:
        return (news_data, n_no_articlebody_in_article), garbage_urls

def url_has_article_attribute(parsed_html: BeautifulSoup) -> bool:
    """
    Checks if a URL has an article attribute
    """

    if parsed_html.html is None:
        return STATUS_5_8

    html_head = parsed_html.html.head

    if html_head is None:
        return STATUS_5_9

    meta_tag = html_head.find(
        "meta", attrs=dict(content="article")
        )
    
    if meta_tag is None:
        return STATUS_5_7
    return STATUS_0

def _search_keys_from_one_news_url(
    url: list, 
    media_url: str,
    score: float,
    order_keys: bool=True
    ) -> tuple[dict[str, str | int], str]:
    """
    Extracts data from one potential news URL, including JSON, metadata, and
    summary information.

    Args:
        url (str): The URL of the potential news article to be processed.
        media_url (str): The URL associated with the media source.
        score (float): A score or rating for processing the URL.
        order_keys (bool, optional): If True, the keys in the extracted data 
        are ordered. Default is True.

    Returns:
        tuple[dict, int]: A tuple containing two elements:
            - The first element is a dictionary containing extracted data from 
              the news URL.
            - The second element is an integer representing the status code of 
              the data extraction process.

    """
    
    response, code = _request_content_from_url(url)
    
    if code != STATUS_0:
        return {}, code
    parsed_news_html = BeautifulSoup(response.content, "html.parser")
    
    # Check whether the url is an article
    code = url_has_article_attribute(parsed_news_html)
    if code != STATUS_0:
        return {}, code
    
    # Check if the publication is valid 
    code = url_has_valid_pub_date(parsed_news_html)
    if code != STATUS_0:
        return {}, code
    
    data = {}
    
    # Tries to search for data from the html metadata
    extracted_data = search_news_keys_from_metadata(parsed_news_html, data)
    if extracted_data:
        data.update(extracted_data)

    # Tries to search for data from the application/json
    extracted_data = search_news_keys_from_jsons(
        parsed_news_html, 
        url, 
        media_url
        )
    if extracted_data:
        data.update(extracted_data)
    
    # Extracts the title of the news article
    title_found = data.get("title", False)

    # Extracts the creation datetime of the news article
    creation_datetime_found = data.get("creation_datetime", False)
    if not title_found or not creation_datetime_found:
        return {}, STATUS_5_5
    
    # Extracts the body of the news article
    body_found = data.get("body", False)
    if not body_found:
        extracted_data = get_body_summary(
            parsed_news_html, 
            url,
            media_url
            )
        
        body_summary_found = extracted_data.get("body", False)
        if not body_summary_found:
            return {}, STATUS_5_6
        
        data.update(extracted_data)

    data["country"] = parsed_news_html.html.attrs.get("lang", "")
    # TODO complete this
    data["media_url"] = media_url
    data["url"] = url
    data["score"] = score

    if order_keys:
        return order_dict_keys(data), STATUS_0
    
    return data, STATUS_0

def _request_content_from_url(url: str) -> tuple[requests.Request, str]:
    """
    Sends a GET request to one potential news URL to retrieve content.

    Args:
        url (str): The URL of the potential news article to send a GET request to.

    Returns:
        tuple[requests.Response, int]: A tuple containing two elements:
            - The first element is a requests.Response object containing the response data.
            - The second element is an integer representing the status code of the request.

    """
    try:
        response = requests.get(
            url, 
            headers=HEADERS_REQUEST, 
            timeout=MEDIA_GET_REQ_TIMEOUT
            )
    except requests.exceptions.TooManyRedirects as e1:
        return None, STATUS_1_1
    except requests.exceptions.RequestException as e2:
        return None, STATUS_1_2
    except UnicodeDecodeError as e3:
        return None, STATUS_1_3
    except Exception as e4:
        return None, STATUS_1_2
    return response, STATUS_0

def url_has_valid_pub_date(parsed_html):
    """
    Accepts or rejects the url if the publication date is less than 
    N_MAX_DAYS_OLD days old. The date is discarded either it is:
        1) missing
        2) too old, given specifications, 
        3) format unexpected or invalid.
    Args:
        parsed_html (BeautifulSoup): A BeautifulSoup object representing the 
        parsed HTML content.

    Returns:
        int: An integer representing the status code for URL filtering.

    """

    try:
        head = parsed_html.html.head
        meta_tag_published_time = head.find("meta", 
                                            attrs=dict(
                                                property=regex_publication_ts
                                                )
                                            )
        if meta_tag_published_time is None:
            return STATUS_5_2
        
        publ_tsm = meta_tag_published_time.attrs["content"]
        
        if not publ_tsm:
            return STATUS_5_3
        
        datetime_diff = TODAY_LOCAL_DATETIME - datetime.fromisoformat(publ_tsm)
        days_old = datetime_diff.total_seconds() / SECONDS_IN_DAY
        
        if days_old > N_MAX_DAYS_OLD:
            return STATUS_5_4
    except:
        return STATUS_5_1
    return STATUS_0

def order_dict_keys(keys_values: dict, only_values: bool=True):
    if only_values:
        return tuple(
            [keys_values.get(target_k, "") for target_k in ORDER_KEYS]
            )
    else:
        return {
            target_k: keys_values.get(target_k, "") for target_k in ORDER_KEYS
            }
    
def read_news_for_media(where_params):
    with sqlite3.connect(DB_NAME_NEWS, timeout=DB_TIMEOUT) as conn:
        cursor = conn.cursor()
        
        if not isinstance(where_params, (tuple, list)):
            where_params = (where_params, )
    
    return cursor.execute(SELECT_ALL_URLS_IN_MEDIA_QRY_STR, where_params) \
                 .fetchall()

def insert_data(data: tuple[tuple], query: str, retries=5, initial_delay=0.1):
    """
    Inserts data into the SQLite database with a retry mechanism in case of a locked database.
    
    Parameters:
    - data: The data to be inserted.
    - query: The SQL query command to execute.
    - retries: Number of retries if the database is locked (default is 5).
    - initial_delay: Initial delay between retries in seconds (default is 0.1 seconds).
    """
    delay = initial_delay
    
    if not isinstance(data, tuple):
        data = tuple(data)

    while retries > 0:
        try:
            with sqlite3.connect(DB_NAME_NEWS, timeout=DB_TIMEOUT) as conn:
                # Enable Write-Ahead Logging (WAL) mode for better concurrency
                conn.execute('PRAGMA journal_mode=WAL;')
                cursor = conn.cursor()

                # Insert data
                cursor.executemany(query, data)
                conn.commit()  # Commit the transaction
            break  # Exit loop if the insertion is successful
        
        except sqlite3.OperationalError as e:
            if "locked" in str(e):
                print(f"Database is locked. Retrying in {delay} seconds...")
                time.sleep(delay)  # Wait before retrying
                delay *= 2  # Exponential backoff: double the delay at each retry
                retries -= 1  # Decrease the retry count
            else:
                raise # If it's not a locking issue, re-raise the exception

    if retries == 0:
        print("Failed at db insert after multiple retries.")

def run_multi_process(sections_file_path: str, num_processes: int, process_function):
    """
    Runs a specified process function in parallel using multiple processes.

    Args:
        sections_file_path (str): The path to the input file containing data for 
        parallel processing.
        num_processes (int): The number of parallel processes to create and 
        run.
        process_function (function): The function to be executed in parallel 
        processes.

    Notes:
    - This function reads data from the input file, creates a manager to share 
      data between processes, and then spawns
      multiple processes to execute the specified process function in parallel.
    - The function provides a way to distribute the workload across multiple 
      processes for efficient parallel execution.

    """

    # Load the sections urls and other data, as a dictionary
    with  open(sections_file_path, 'r') as f:
        sections_data = json.load(f)

    with mp.Manager() as manager:
        queue_sections_data = manager.dict(sections_data)
        # Create a process for each chunk and run in parallel
        processes = []
        for _ in range(num_processes):
            process = mp.Process(
                target=process_function, 
                args=(queue_sections_data, )
                )
            processes.append(process)
            process.start()
        # Wait for all processes to finish
        for process in processes:
            process.join()

def main_multi_procesing_process(queue_data: dict):
    """
    Process news articles in parallel from queued media data using 
    multi-procesing.

    Args:
        queue_data (dict): A container of media URLs, associated sections
          and scores.

    Notes:
    - This function processes news articles in parallel from queued media data
      using multi-procesing.
    - It iterates through media URLs and their associated sections and scores, 
      making HTTP requests to extract news article URLs.
    - Extracted URLs are processed, and statistics are recorded for the number 
      of processed articles and those missing body content.
    - The function is designed for parallel execution to improve processing 
      efficiency.

    """
    media_stats_manager = StatisticsManager(start_time=True, log_dir=PATH_STATS)
    media_news_urls = []
    n_processed = 0
    n_no_body = 0
    
    while len(queue_data) > 0:
        # The List-proxy can safely remove items, as the multiprocessing.Manager()  
        # takes care of the necessary internal locking mechanisms to ensure 
        # safe concurrent access.
        domain, sections_data = queue_data.popitem()
        domain_url = "https://" + domain

        for section_url, _, score in sections_data["records"]:
            score = float(score)
            section_url = section_url.strip()

            try:
                response = requests.get(
                    section_url, 
                    headers=HEADERS_REQUEST, 
                    timeout=NEWS_ARTICLE_GET_REQ_TIMEOUT
                    )
            except requests.exceptions.Timeout:
                # Report the timeout exception at the log
                add_new_log_line("Request timeout at {section_url}")
                continue
            except requests.exceptions.RequestException as e:
                # Handle other request exceptions
                print(f"An error occurred: {str(e)}\n \
                      Retrying request without 'www'")
                section_url = section_url.replace("www.", "")
                try:
                    response = requests.get(
                        section_url, 
                        headers=HEADERS_REQUEST, 
                        timeout=NEWS_ARTICLE_GET_REQ_TIMEOUT
                        )
                except Exception as e:
                    # Report the exception at the log
                    add_new_log_line(f"Unexpected exception at second request to {section_url}:\n{e}")
                    continue
                # Report the exception at the log
                add_new_log_line(f"Get request general exception, {e}, at {section_url}")
                
                domain_url = domain_url.replace("www.", "")
                print("Request success without 'www'")
            
            try:
                parsed_hmtl = BeautifulSoup(response.content, "html.parser")
                
                tags_with_url = parsed_hmtl.html.body \
                                           .find_all(
                                               "a", 
                                               href=regex_valid_url_format)
                                               # href=re.compile("https?:.*"))
            except Exception as e:
                add_new_log_line(f"Unexpected exception at fetching <a href=...> \
                                  tags: 'body' not found at section {section_url}:\n{e}")
                continue
            
            raw_urls = [x.attrs["href"] for x in tags_with_url if x.attrs.get("href", False)]
            
            for raw_url in raw_urls:
                if regex_url_has_query.search(raw_url):
                    search_spec_char = regex_url_has_query.search(raw_url)
                    query_start_pos = search_spec_char.span()[0]
                    raw_url = raw_url[:query_start_pos]
                
                if raw_url.startswith("//"):
                    raw_url = "https:" + raw_url
                elif raw_url.startswith("/"):
                    raw_url = domain_url[:-1] + raw_url
                elif raw_url.startswith("www.") \
                  or not raw_url.startswith("https://"):
                    raw_url = "https://" + raw_url
                
                media_news_urls.append(raw_url)
        
        # Get unique elements and converto to list.
        media_news_urls = list(set(media_news_urls))
        n1, n2 = process_raw_urls(media_news_urls, domain_url, score)
        
        # Update counters
        n_processed += n1
        n_no_body += n2
    
    stats_for_log = ("Process summary", n_processed, n_no_body, "")
    
    media_stats_manager.write_stats(
        stats_for_log, 
        writer_id="summary"
        )

if __name__ == "__main__":
    print(f"\nScraping started at: {CURRENT_DATE} {CURRENT_TIME}")

    # Create assistant
    assistant = set_or_create_assistant()

    # Extract the last version
    num_last_version = get_last_sections_file_num(PATH_FILE_MEDIA_SECTIONS)
    run_multi_process(
        f"data/sources/source_urls_v{num_last_version}.json", 
        NUM_PROCESSING_WORKERS, 
        main_multi_procesing_process
        )
    #for f in glob.glob(os.path.join(PATH_DATA, "checkpoint_*.json")):
    #    os.remove(os.path.join(PATH_DATA, f))
    end_date, end_time = str(datetime.today()).split(" ")
    print(f"\nScraping ended at: {end_date} {end_time}")