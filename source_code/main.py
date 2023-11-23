from bs4 import BeautifulSoup
from datetime import datetime, timedelta, timezone
from process_variables import *
import multiprocessing as mp
import pandas as pd
import utilities as ut
import requests
import sqlite3
import re
import openai
import json
import os

# Classes
class ErrorReporter(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(message)
        with open("Errors log.txt", "w") as errors_file:
            errors_file.write(message)
# Dates and times
TODAY_LOCAL_DATETIME = datetime.now().replace(tzinfo=timezone.utc)
CURRENT_DATE, CURRENT_TIME = str(datetime.today()).split(" ")
if not BLOCK_API_CALL:
    openai.api_key = os.getenv("OPENAI_API_KEY")
with open(FILE_PATH_PROMPT_ROLE_SUMMARY, "r") as file:
    PROMPT_ROLE_SUMMARY = file.read()
regex_url_has_query = re.compile(r"[?!)(=&+%@#]{1}")
regex_valid_url_format = re.compile(r"^(?:https:\/\/)?|^\/{1}[^\/].*|^www[.].*")
regex_too_short_url_end = re.compile(r"^[.a-zA-Z0-9]+(-[.a-zA-Z0-9]+){,2}$")
regex_url_startswith_https = re.compile(r"https?:[\/]{2}")
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
regex_creation_datetime = re.compile(r"Creation DateTime: (.*).*", flags=re.DOTALL)
regex_update_datetime = re.compile(r"Update DateTime: (.*).*Body Summary", flags=re.DOTALL)
regex_only_summary = re.compile(r"Body Summary:(.*)", flags=re.DOTALL)

## Decorators
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
def extract_data_from_jsons(html: BeautifulSoup, 
                            url: str,
                            media: str,
                            only_return_body: bool=True 
                            ) -> dict:
    """
    Extracts and returns structured data from JSON scripts within an HTML document.

    Args:
        html (BeautifulSoup): Object representing the parsed HTML content.
        url (str): URL of the web page from which data is being extracted.
        media (str): Type of media (e.g., "news," "article") for contextual validation.
        only_return_body (bool, optional): If True, the function will return only the body content.
            Default is True.

    Returns:
        dict: Extracted data fields are as follows:
            - The first dictionary includes the following keys (if found):
                - "title": The title of the web page.
                - "description": A brief description of the web page.
                - "body": The main content of the web page.
                - "creation_datetime": The date when the content was published.
                - "modified_datetime": The date when the content was last modified.
                - "main_topic": The main topic or category of the content.
                - "other_topic": Additional topics or tags associated with the content.
                - "n_tokens": The number of tokens in the extracted body content.

        If the "@type" or "type" field is not found in the JSON data, an empty dictionary is returned.
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
    jsons = html.find_all("script", 
                          attrs={"type": regex_application_jsons})
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
            _, destructured_keys_and_values, _, _ = utils.direct_recursive_destructure(sub_json_data)
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
                    body, n_tokens = get_body_summary(ut.remove_body_tags(value), 
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
                    topics = ut.clean_topics(value)
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

def extract_data_from_metadata(parsed_html: BeautifulSoup, 
                               data_input: dict) -> dict:
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
        prompt (str): A system-level instruction or initial message for the chat conversation.
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

def extract_keys_with_completion(parsed_code: BeautifulSoup) -> dict:
    """
    Extracts structured data from parsed HTML content using BeautifulSoup.

    Args:
        parsed_code (BeautifulSoup): A BeautifulSoup object representing the parsed HTML content.

    Returns:
        dict: A dictionary containing the extracted structured data, including:
            - "n_tokens": The number of tokens in the extracted content.
            - "title": The title of the content.
            - "other_topic": Additional topics or tags associated with the content.
            - "creation_datetime": The date when the content was created or published.
            - "modified_datetime": The date when the content was last updated.
            - "body": The main content or summary of the HTML content.
            - "image_url": URL of an associated image (initialized as an empty string).

    Notes:
    - This function extracts structured data from specific HTML tags (e.g., <p>, <h1>, <h2>) within the parsed HTML content.
    - It cleans and processes the text content of these tags.
    - The function then utilizes regular expressions and an external function (call_completion_api) to extract specific data elements.
    - If any of the data elements are not found, they are set to default or empty values.
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

def extract_summary_with_gpt(parsed_html: BeautifulSoup, 
                             url: str,
                             media: str
                             ) -> dict:
    """
    Extracts article body summarization directly from within parsed HTML.

    Args:
        parsed_html (BeautifulSoup): Object representing the parsed HTML content.
        url (str): URL of the web page associated with the text content.
        media (str): The type of media (e.g., "news," "article") for contextual validation.

    Returns:
        dict: A dictionary containing the extracted structured data, including:
            - "n_tokens": The number of tokens in the extracted text content.
            - "body": The main content or summary of the text content.

    Notes:
    - This function extracts structured data from specific HTML tags (e.g., <p>) within the parsed HTML content.
    - It cleans and processes the text content of these tags.
    """

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

@garbage_logger()
def get_body_summary(text: str, 
                     url: str,
                     media: str
                     ) -> tuple[tuple[str, str | int] | dict[str, bool | list]]:
    """
    Generates a summary for text content using an external API.

    Args:
        text (str): The text content to be summarized.
        url (str): The URL of the source associated with the text content.
        media (str): The type of media (e.g., "news," "article") for contextual validation.

    Returns:
        tuple[str, str | int] | dict[str, bool | list]: A tuple containing two elements:
            - The first element is the generated summary for the text content.
            - The second element is the number of tokens in the generated summary. 
              a summary is not found or an error occurs -1 tokens are returned

    """
    if BLOCK_API_CALL:
        #return (text, -1), {"status_code": STATUS_0, "id": ""}
        return (text, -1), {"has_garbage": True, "data": []}
    try:
        message_content = call_completion_api(text, PROMPT_ROLE_SUMMARY)
    except Exception as e:
        #return ("", 0), {"status_code": STATUS_4, "id": url}
        return ("", 0), {"has_garbage": False, "data": [(url, media, STATUS_4)]}
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
                   score: float) -> tuple[int, int]:
    """
    Processes a list of raw URLs, extracts and stores news data, and provides statistics.

    Args:
        raw_urls (list): A list of raw URLs to be processed.
        media_url (str): The URL associated with the media source.
        score (float): A score or rating for processing URLs.

    Returns:
        tuple[int, int]: A tuple containing two integers:
            - The first integer represents the number of processed news articles.
            - The second integer represents the number of news articles with missing body content.

    """

    media_stats_manager = ut.StatisticsManager(True)
    urls = find_valid_urls(raw_urls,
                           media_url)
    if urls:
        lock = mp.Lock()
        with lock:
            read_urls = pd.DataFrame(read_news_for_media(media_url), 
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
                                              order_keys=True) # keys are ordered
        #print("Data:", news_data)
        if news_data:
            lock = mp.Lock()
            with lock:
                insert_news(news_data)
                print("Rows inserted:", len(news_data), "\n")
        # Statistics
        n_processed = len(news_data)
        print(f"\nPID {str(os.getpid())} {media_url}; read {len(read_urls)}; extracted {len(extracted_urls)}; to process {len(novel_news_to_process)}; processed: {n_processed}")
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

@garbage_logger()
def find_valid_urls(input_urls: list, 
                    media_url: str
                    ) -> tuple[list, dict]:
    """
    Filters and validates a list of input URLs, avoiding known garbage URLs.

    Args:
        input_urls (list): URLs to be filtered and validated.
        media_url (str): URL associated with the media source.

    Returns:
        tuple[list, dict]: A tuple containing two elements:
            - The first element is a list of validated and filtered URLs.
            - The second element is a dictionary providing information about garbage URLs.

    """
    # Open file of garbage urls to avoid
    lock = mp.Lock()
    with lock:
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
        if ut.find_invalid_files(url):
            #status_and_id.append({"status_code": STATUS_3_2, "id": url})
            new_garbage.append((url, media_url, STATUS_3_2))
            continue
        if ut.find_invalid_sections(url):
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
                   order_keys=True
                   ) -> tuple[tuple[list, int], dict]:
    """
    Extracts news data from a list of URLs associated with a media source.

    Args:
        news_urls (list): A list of news URLs to be processed.
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
    for news_url in news_urls:
        data, code = _extract_data_from_news(news_url, 
                                             media_url,
                                             score,
                                             order_keys)
        if not data:
            garbage_urls.append((news_url, media_url, code))
            if code == STATUS_5_6:
                n_no_articlebody_in_article += 1
            continue
        news_data.append(data)

    print("Garbage from data extraction:", len(garbage_urls))
    if garbage_urls:
        return (news_data, n_no_articlebody_in_article), {"has_garbage": False, "data": garbage_urls}
    else:
        return (news_data, n_no_articlebody_in_article), {"has_garbage": True, "data": garbage_urls}

def _extract_data_from_news(news_url: list, 
                            media_url: str,
                            score: float,
                            order_keys: bool=True) -> tuple[dict[str, str | int], str]:
    
    """
    Extracts data from a news URL, including JSON, metadata, and summary information.

    Args:
        news_url (str): The URL of the news article to be processed.
        media_url (str): The URL associated with the media source.
        score (float): A score or rating for processing the URL.
        order_keys (bool, optional): If True, the keys in the extracted data are ordered.
            Default is True.

    Returns:
        tuple[dict, int]: A tuple containing two elements:
            - The first element is a dictionary containing extracted data from the news URL.
            - The second element is an integer representing the status code of the data extraction process.

    """

    response, code = _send_get_request(news_url)
    if code != STATUS_0:
        return {}, code
    parsed_news_hmtl = BeautifulSoup(response.content, 
                                     "html.parser")
    # Accept or reject url if news date is more than N_MAX_DAYS_OLD days older
    code = filter_old_url(parsed_news_hmtl)
    if code != STATUS_0:
        return {}, code
    data = {}
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
        extracted_data = extract_summary_with_gpt(parsed_news_hmtl, 
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

    if order_keys:
        return order_dict_keys(data), STATUS_0
    return data, STATUS_0

def _send_get_request(news_url: str) -> tuple[requests.Request, str]:
    """
    Sends a GET request to a news URL to retrieve content.

    Args:
        news_url (str): The URL of the news article to send a GET request to.

    Returns:
        tuple[requests.Response, int]: A tuple containing two elements:
            - The first element is a requests.Response object containing the response data.
            - The second element is an integer representing the status code of the request.

    """
    try:
        response = requests.get(news_url, 
                                headers=HEADERS, 
                                timeout=MEDIA_GET_REQ_TIMEOUT)
    except requests.exceptions.TooManyRedirects as e1:
        return None, STATUS_1_1
    except requests.exceptions.RequestException as e2:
        return None, STATUS_1_2
    except UnicodeDecodeError as e3:
        return None, STATUS_1_3
    except Exception as e4:
        return None, STATUS_1_2
    return response, STATUS_0

def filter_old_url(parsed_html):
    """
    Filters and validates a news article URL based on publication date 
    is missing, too old, or not in the expected format.

    Args:
        parsed_html (BeautifulSoup): A BeautifulSoup object representing the parsed HTML content.

    Returns:
        int: An integer representing the status code for URL filtering.

    """

    try:
        meta_tag_published_time = parsed_html.html.head.find("meta", 
                                                             attrs={"property": regex_publication_ts})
        if meta_tag_published_time is None:
            return STATUS_5_2
        publ_tsm = meta_tag_published_time.attrs["content"]
        if not publ_tsm:
            return STATUS_5_3
        dtime_diff = (TODAY_LOCAL_DATETIME - datetime.fromisoformat(publ_tsm)).total_seconds() / SECONDS_IN_DAY
        if dtime_diff > N_MAX_DAYS_OLD:
            return STATUS_5_4
    except:
        return STATUS_5_1
    return STATUS_0

def order_dict_keys(keys_values: dict, 
                    only_values: bool=True):
    if only_values:
        return tuple([keys_values.get(target_k, "") for target_k in ORDER_KEYS])
    else:
        return {target_k: keys_values.get(target_k, "")  for target_k in ORDER_KEYS}
    
def read_news_for_media(where_params):
    with sqlite3.connect(DB_NAME_NEWS, 
                         timeout=DB_TIMEOUT) as conn:
        cursor = conn.cursor()
        if not isinstance(where_params, (tuple, list)):
            where_params = (where_params, )
    return cursor.execute(SELECT_ALL_URLS_IN_MEDIA_QRY_STR, 
                          where_params).fetchall()

def insert_news(data: tuple[tuple]):
    with sqlite3.connect(DB_NAME_NEWS, 
                         timeout=DB_TIMEOUT) as conn:
        cursor = conn.cursor()
        if not isinstance(data, tuple):
            data = tuple(data)
        cursor.executemany(INSERT_QRY_STR, 
                           data)
        conn.commit()

def run_multi_process(file_path: str, 
                      num_processes: int, 
                      process_function):
    """
    Runs a specified process function in parallel using multiple processes.

    Args:
        file_path (str): The path to the input file containing data for parallel processing.
        num_processes (int): The number of parallel processes to create and run.
        process_function (function): The function to be executed in parallel processes.

    Notes:
    - This function reads data from the input file, creates a manager to share data between processes, and then spawns
      multiple processes to execute the specified process function in parallel.
    - The function provides a way to distribute the workload across multiple processes for efficient parallel execution.

    """

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
    """
    Process news articles in parallel from queued media data using multi-threading.

    Args:
        queued_media_data (dict): A dictionary containing media URLs and associated sections and scores.

    Notes:
    - This function processes news articles in parallel from queued media data using multi-threading.
    - It iterates through media URLs and their associated sections and scores, making HTTP requests to extract news article URLs.
    - Extracted URLs are processed, and statistics are recorded for the number of processed articles and those missing body content.
    - The function is designed for parallel execution to improve processing efficiency.

    """
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
                try:
                    response = requests.get(section_url, 
                                            headers=HEADERS, 
                                            timeout=NEWS_ARTICLE_GET_REQ_TIMEOUT)
                except Exception as e:
                    print(f"Unexpected exception at second request attempt to url {section_url}:\n{e}")
                    continue
                #ErrorReporter(f"Get request general exception, {e}, at {section_url}")
                media_url = media_url.replace("www.", "")
                print("Request success without 'www'")
            
            parsed_hmtl = BeautifulSoup(response.content, 
                                        "html.parser")
            tags_with_url = parsed_hmtl.html.body \
                                       .find_all("a", href=regex_valid_url_format)
                                                # href=re.compile("https?:.*"))
            raw_urls = [x.attrs["href"] for x in tags_with_url if x.attrs.get("href", False)]
            for raw_url in raw_urls:
                if regex_url_has_query.search(raw_url):
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
    media_stats_manager.write_stats(("Process summary", 
                                    n_processed, 
                                    n_no_body), 
                                    input_iso_datetime=TODAY_LOCAL_DATETIME,
                                    pid="summary"
                                    )

if __name__ == "__main__":
    print(f"\n...Datetime of process: {CURRENT_DATE} {CURRENT_TIME}...\n")

    # Extract the last version
    num_last_version = ut.get_last_sections_file_num(PATH_SECTIONS_FILE)
    run_multi_process(
        os.path.join(PATH_DATA, 
                     f"final_url_sections_v{num_last_version}.json"), 
        NUM_CORES, 
        main_multi_threading_process)
    #for f in glob.glob(os.path.join(PATH_DATA, "checkpoint_*.json")):
    #    os.remove(os.path.join(PATH_DATA, f))
    print("\n...The process ended...")
