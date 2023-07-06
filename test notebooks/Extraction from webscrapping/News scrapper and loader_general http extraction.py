import json
from pprint import pprint
import os
from datetime import datetime
import requests
import openai
from bs4 import BeautifulSoup
import re
import pandas as pd
from lxml import etree
import html
import sqlite3
import json
import os
# Dynamical global variables
openai.api_key = os.getenv("OPENAI_API_KEY")
with open("./data/role prompt_keys extraction.txt", "r") as file:
    prompt_role = file.read()
prompt_role_summary_only = """
You are a journalist. Output the next format:

Body Summary: [body summary]

Replace [body summary] with the summary from the incoming text body. 
The summarization instructions are: make a summary in third-person 
point of view. Translate the summary into the incoming text language. 
Do not mention the article itself. Add only important details. 
This is the text to summarize:
"""
conn = sqlite3.connect("./data/db.sqlite3")
cursor = conn.cursor()
scheduled_process = True
current_date, current_time = str(datetime.today()).split(" ")
regex_title = re.compile("title|headline|titular|titulo")
regex_body = re.compile("articleBody")
regex_date_published = re.compile("date.*[pP]ub.*")
regex_date_modified = re.compile("date.*[mM]od.*")
regex_tags = re.compile("tag[s]?|topic[s]?|tema[s]?|etiqueta[s]?|keyword[s]?")
node_ignore = ["europa", 
               "africa",
               "asia",
               "oceania",
               "prensadigital"
              ]

# Constants
DIGITAL_MEDIAS_MAIN_ROOT = "https://www.prensaescrita.com"
DIGITAL_MEDIAS_URL = "https://www.prensaescrita.com/prensadigital.php"
#PATH = os.path.join("News storage", API_SOURCE, current_date)
#FILE_NAME = current_date + "_" + API_SOURCE + "_" + "extracted_news.json"
#FILE_PATH = os.path.join(PATH, FILE_NAME)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
}

# Functions

def _update_date(current_date, current_time):
    new_date, new_time = str(datetime.today()).split(" ")
    if new_date != current_date:
        return new_date, new_time
    return current_date, current_time

def find_keys(elements_json):
    news_data = {}
    title_found = False
    body_found = False
    date_pub_found = False
    date_mod_found = False
    tag_found = False
    news_data["title"] = None
    news_data["body"] = None
    news_data["creation_date"] = None
    news_data["modification_date"] = None
    news_data["tags"] = None
    
    for element_json in elements_json:
        json_file = json.loads(element_json.get_text(), 
                               strict=False)
        is_empty = False
        if isinstance(json_file, list):
            if json_file:
                json_file = json_file[0]
            else:
                is_empty = True
        for key in json_file:
            if is_empty:
                break
            if not title_found and regex_title.search(key):
                news_data["title"] = json_file[key]
                title_found = True
            elif not body_found and regex_body.search(key):
                news_data["body"] = remove_tags(json_file[key])
                news_data["body"] = generate_summary(news_data["body"])
                body_found = True
            elif not date_pub_found and regex_date_published.search(key):
                news_data["creation_date"] = json_file[key]
                date_pub_found = True
            elif not date_mod_found and regex_date_modified.search(key):
                news_data["modification_date"] = json_file[key]
                date_mod_found = True
            elif not tag_found and regex_tags.search(key):
                news_data["tags"] = json_file[key]
                news_data["tags"] = ";".join(news_data["tags"])
                tag_found = True
            if all([title_found, body_found, date_pub_found, date_mod_found, tag_found]):
                break
    return news_data

def find_keys_gpt(parsed_code):
    tags_with_text = parsed_code.find_all(lambda tag: (tag.name == "p" and not tag.attrs) or ("h" in tag.name and not tag.attrs))
    text_clean_from_tags = "".join([re.sub("\n+", "\n", tag.get_text()) for tag in tags_with_text])
    openai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_role},
            {"role": "user", "content": text_clean_from_tags},
        ]
    )
    message_content = openai_response["choices"][0].message["content"]

    data = [x.split(": ")[1].strip() for x in message_content.split("\n")]
    keys = (("title", 0), ("body", 4), ("creation_date", 2), ("update_date", 3), ("tags", 1))
    data = {key: data[idx] for (key, idx)  in keys}
    data["image"] = None
    return data

def generate_summary(text):
    openai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_role_summary_only},
            {"role": "user", "content": text},
        ]
    )
    message_content: str = openai_response["choices"][0].message["content"]

    if "Body Summary: " in message_content:
        body_summary = message_content.split("Body Summary: ")[-1]
    else:
        body_summary = message_content
    return body_summary

def remove_tags(text):
    return re.sub("<.*?>", "", text)

#def clean_characters_entities(text):
#    return html.unescape(text)

def _get_url_medias_from_region(url):
    response = requests.get(url, headers=HEADERS)
    parsed_hmtl = BeautifulSoup(response.content, "html.parser")
    links = parsed_hmtl.find_all("a", string=lambda text: text and "www." in text)
    return [l["href"] for l in links]

def get_region_to_url(ignores, file_path=None, on_save=False, on_override=False):
    response = requests.get(DIGITAL_MEDIAS_URL, headers=HEADERS)
    parsed_hmtl = BeautifulSoup(response.content, "html.parser")

    tags = parsed_hmtl.find_all(lambda tag: tag.name == "a" and tag.attrs["href"] is not None)
    region_urls = [x.attrs["href"] for x in tags]
    region_urls = [x for x in region_urls if x.endswith(".php") and x.startswith("/") and not any(1 if node in x else 0 for node in ignores)]
    region_names = [x[1:-4] for x in region_urls]
    region_urls = [DIGITAL_MEDIAS_MAIN_ROOT + url for url in region_urls]
    region_to_url = {name: url for url, name in zip(region_urls, region_names)}

    region_to_media_urls = {}
    for region, media_url in region_to_url.items():
        print(media_url)
        region_to_media_urls[region] = _get_url_medias_from_region(media_url)

    print(f"Number of regions: {len(region_to_media_urls)};\nNumber of total digital media available: {sum(len(region_to_media_urls[k]) for k in region_to_media_urls)}")
    if on_save:
        saving_status = _save_media_urls(file_path, region_to_media_urls, on_override)
        if saving_status:
            print(f"Media urls have been saved successfully. Override: {on_override}")
        else:
            print("The file already exists")
    return region_to_media_urls

def _save_media_urls(file_path, data, on_override=False):
    if not os.path.exists(file_path) or on_override:
        with open(file_path, "w") as file:
            json.dump(data, file)
def read_media_urls_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def _save_checkpoint(last_url):
    with open("./data/extraction checkpoint.json", "w") as file:
        json.dump({"last_media_url": last_url}, file)
def read_checkpoint():
    with open("./data/extraction checkpoint.json", "r") as file:
        return json.load(file)["last_media_url"]

def has_http_attribute_value(tag):
    for attr in tag.attrs:
        attr_value = tag.attrs[attr]
        if isinstance(attr_value, str) and ('http://' in attr_value or 'https://' in attr_value) :
            return True
    return False

def store_failed_urls(wrapped_func):
    failed_urls = []

    def wrapper(*args, **kwargs):
        try:
            result = wrapped_func(*args, **kwargs)
            return result
        except Exception as e:
            if 'url' in kwargs:
                failed_urls.append((wrapped_func, kwargs['url']))
            print(f"Error processing URL: {kwargs.get('media_url')}")
            print(f"Error message: {str(e)}")

    wrapper.failed_urls = failed_urls
    return wrapper

def find_news(media_url, date, time):
    y, m = date.split("-")[:2]
    # With headers
    #response = requests.get(media_url, headers=HEADERS)
    try:
        response = requests.get(media_url, headers=HEADERS, timeout=10)  # Set an appropriate timeout value (in seconds)
    except requests.exceptions.Timeout:
        # Handle the timeout exception
        print("The request timed out.")
        return []
    except requests.exceptions.RequestException as e:
        # Handle other request exceptions
        print(f"An error occurred: {str(e)}")
        return []

    # HTTP status code
    #print(f"Status code {response.status_code} for {media_url}")

    parsed_hmtl = BeautifulSoup(response.content, "html.parser")

    tags_with_url = parsed_hmtl.find_all("a", href=re.compile("https?:.*"))
    
    tags_with_url = pd.Series(tags_with_url).unique().tolist()

    valid_news_urls = []
    not_valid_news_urls = []
    for tag_with_url in tags_with_url:
        hiperlink = tag_with_url.attrs["href"]
        # Clean query ~Filter out urls with query symbols~
        if re.search("[?=&+%@#]{1}", hiperlink):
            #continue
            search_spec_char = re.search("[?=&+%@#]{1}", hiperlink)
            query_start = search_spec_char.span()[0]
            hiperlink = hiperlink[:query_start]
        if media_url in hiperlink or media_url.replace("www.", "") in hiperlink: 
            valid_news_urls.append(hiperlink)
        else:
            print("\t\t Skipped from 1-find_news(): " + hiperlink)
                    
    return list(set(valid_news_urls))

def find_news_data(media_news_urls, author):
    news_media_data = []
    for i, news_url in enumerate(media_news_urls):
        resp_url_news = requests.get(news_url, 
                                     headers=HEADERS)
        parsed_news_hmtl = BeautifulSoup(resp_url_news.content, 
                                         "html.parser")
        elements_json = parsed_news_hmtl.find_all("script", 
                                                  attrs={"type": re.compile(".+json$")})
        try:
            data = find_keys(elements_json)
            if not data["title"] or not data["body"]:
                data = find_keys_gpt(parsed_news_hmtl)
                
            #print("\t\t" + "Skipped from 2-find_news_data(): " + news_url)
            #continue
        except:
            data = find_keys_gpt(parsed_news_hmtl)
        # Skip data element if either title or article keys were not found, i.e. None value on both.
        data["url"] = news_url
        if not data["tags"]:
            elements_with_tags = parsed_news_hmtl.find_all("meta", 
                                                           attrs={"property": "article:tag"})
            data["tags"] = [x.attrs["content"] for x in elements_with_tags]
            data["tags"] = ";".join(data["tags"])
        if not data["tags"]:
            elements_with_tags = parsed_news_hmtl.find_all("meta", 
                                                           attrs={"name": "keywords"})
            data["tags"] = [x.attrs["content"] for x in elements_with_tags]
            data["tags"] = ";".join(data["tags"])
        # TODO complete this
        data["source"] = author
        data["country"] = None
        data["image_url"] = None
        news_media_data.append(data)
    return news_media_data

def order_dict_keys(data, ord_keys, only_values=True):
    if only_values:
        return [tuple(dict_elem[key] for key in ord_keys) for dict_elem in data]
    else:
        return [{key: dict_elem[key] for key in ord_keys} for dict_elem in data]
    
def read_news(where_params):
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
            tags TEXT
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
            tags
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
                ?
            )
            ;
        """
    cursor.executemany(query_str, data)

def main():
    file_name = "region_to_media_urls.json"
    file_path = os.path.join(".", "data", file_name)
    if os.path.exists(file_path):
        print("Reading file of urls of regions...")
        region_to_media_urls = read_media_urls_file(file_path)
    else:
        print("Searching urls for regions")
        region_to_media_urls = get_region_to_url(node_ignore,
                                                 file_path=file_path,
                                                 on_save=True, 
                                                 )
    last_media_url = read_checkpoint()
    checkpoint_started = False
    region_to_media_to_news_counts = {}
    c = 0
    for region, media_urls in region_to_media_urls.items():
        if region in ["madrid"]:
            print(f"Processing news from {region} region...")
            for media_url in media_urls:
                if not checkpoint_started:
                    if media_url == last_media_url or not last_media_url:
                        checkpoint_started = True
                    continue
                # Extract the author
                #author = [x.split("/")[2] for x in region_to_media_urls[region]][0]
                author = media_url
                #print("author:", author)
                select_urls = pd.DataFrame(read_news(author), 
                                           columns=["url2"])
                extraction_urls = pd.DataFrame(find_news(media_url=media_url, date=current_date, time=current_time), 
                                               columns=["url1"]
                                               )
                
                are_new = extraction_urls.merge(select_urls, 
                                                left_on="url1", 
                                                right_on="url2", 
                                                how="left").isnull().any(axis=1)
                # Merge extracted and loaded news in order to process the new ones
                news_to_process = extraction_urls.loc[are_new, "url1"].tolist()
                # Process new news
                news_to_process_data = find_news_data(media_news_urls=news_to_process, author=author) # unordered
                news_to_process_data = order_dict_keys(news_to_process_data, 
                                                       ("title", 
                                                        "body", 
                                                        "source",
                                                        "country",
                                                        "creation_date",
                                                        "modification_date",
                                                        "url",
                                                        "image_url",
                                                        "tags"
                                                        )
                                                       ) # ordered
                insert_news(news_to_process_data)
                # Statistics
                print(f"\tmedia: {media_url} counts: {len(news_to_process_data)}")
                #break
                c += 1
                #if c > 5:
                #    break
                _save_checkpoint(media_url)

if __name__ == "__main__":
    current_date, current_time = _update_date(current_date, current_time)
    print(f"\n...Datetime of process: {current_date} {current_time}...\n")
    main()
    conn.commit()
    print("\n...The process ended...")