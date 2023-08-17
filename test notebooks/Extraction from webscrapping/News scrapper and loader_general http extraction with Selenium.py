import json
import os
from datetime import datetime
import requests
import openai
from bs4 import BeautifulSoup
import re
import pandas as pd
import sqlite3
import json
import os
import multiprocessing

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

# Dynamical global variables
openai.api_key = os.getenv("OPENAI_API_KEY")
with open("./data/role prompt_keys extraction.txt", "r") as file:
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
current_date, current_time = str(datetime.today()).split(" ")
regex_title = re.compile("title|headline|titular|titulo")
regex_body = re.compile("articleBody")
regex_date_published = re.compile("date.*[pP]ub.*")
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

# Constants
MAX_N_MEDIAS = 100
N_SAVE_CHECKPOINT = 5
#PATH = os.path.join("News storage", API_SOURCE, current_date)
#FILE_NAME = current_date + "_" + API_SOURCE + "_" + "extracted_news.json"
#FILE_PATH = os.path.join(PATH, FILE_NAME)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
}

# Functions
def find_news_body(html, data):
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

def remove_body_tags(text):
    return re.sub("<.*?>", "", text)

def find_news_body_gpt(url):
    print("..Body through gpt..")
    driver.get(url)
    driver.find_elements(By.XPATH, "html/body//p|h1|h2")
    paragraphs = driver.find_elements(By.XPATH, "html/body//div/p|h1|h2")
    parag_texts = str({i: x.text if x.text else "\n" for i, x in enumerate(paragraphs)})[1:-1]
    
    openai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_role},
            {"role": "user", "content": parag_texts},
        ]
    )
    message_content = openai_response["choices"][0].message["content"]
    try:
        n_tokens = regex_n_tokens.search(message_content).groups()[0]
    except:
        n_tokens = 0
    try:
        body = regex_only_summary.search(message_content).groups()[0]
    except:
        body = None
    data = {
        "n_tokens": n_tokens,
        "body": body
    }
    return data

def get_body_summary(text):
    openai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_role_summary_only},
            {"role": "user", "content": text},
        ]
    )
    message_content: str = openai_response["choices"][0].message["content"]
    n_tokens = regex_n_tokens.search(message_content).groups()[0]
    body_summary = regex_only_summary.search(message_content).groups()[0]
    return body_summary, n_tokens

def find_media_section(base_url, pid: str):
    try:
        response = requests.get(base_url, headers=HEADERS, timeout=10)  # Set an appropriate timeout value (in seconds)
    except requests.exceptions.Timeout:
        # Handle the timeout exception
        print("The request timed out.")
        ErrorReporter(f"Get request timeout exception at {base_url}")
        return []
    except requests.exceptions.RequestException as e:
        # Handle other request exceptions
        print(f"An error occurred: {str(e)}")
        ErrorReporter(f"Get request general exception, {e}, at {base_url}")
        return []

    parsed_hmtl = BeautifulSoup(response.content, "html.parser")
    for x in parsed_hmtl.body.find_all("a"):
        x.attrs.get("href", None)
    try:
        links = [x.attrs.get("href", None) for x in parsed_hmtl.body.find_all("a")]
    except Exception as e:
        print(e)
        return None

    links_serie = pd.Series(links).dropna()
    nodes = links_serie.str.replace(base_url, "", regex=True)

    nodes_split = nodes.str.split("/")
    nodes_split_clean = nodes_split.apply(lambda x: [elem for elem in x if elem])

    nodes_split_clean_filter = nodes_split_clean.str.len().eq(1)

    links_valid = links_serie[nodes_split_clean_filter]
    links_valid_complete = [base_url + x if not x.startswith(base_url) else x for x in links_valid]
    links_valid_complete_unique = pd.Series(links_valid_complete).unique().tolist()

    last_media_section_url, last_media_news_url = read_news_checkpoint(pid)
    if not last_media_section_url:
        checkpoint_started = False
    else:
        checkpoint_started = True
    media_stats_reporter = StatisticsReporter()
    all_urls = []
    for media_section_url in links_valid_complete_unique:
        if checkpoint_started:
            if media_section_url == last_media_section_url or not last_media_section_url:
                checkpoint_started = False
            else:
                print("Skip", media_section_url)
                continue
        urls, checkpoint_started = find_news_urls(pid,
                                                  media_section_url, 
                                                  current_date, 
                                                  current_time, 
                                                  checkpoint_started, 
                                                  last_media_section_url, 
                                                  last_media_news_url)
        #all_urls.extend(urls)
        select_urls = pd.DataFrame(read_stored_news(base_url), 
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
        processed_news_data = find_news_data(media_news_urls=news_to_process, author=base_url) # unordered
        processed_news_data = order_dict_keys(processed_news_data, 
                                              ("title", 
                                               "body", 
                                               "source",
                                               "country",
                                               "published_time",
                                               "modified_time",
                                               "url",
                                               "image",
                                               "tags",
                                               "n_tokens"
                                               )
                                               ) # ordered
        insert_news(processed_news_data)
        # Statistics
        counts = str(len(processed_news_data))
        media_stats_reporter.write_extraction_stats((media_section_url, counts, ))
        print(f"\tmedia name: {media_section_url} counts: {counts}")
    return None #pd.Series(all_urls).unique().tolist()

def find_news_urls(pid, media_url, date, time, on_start_checkpoint, last_media, last_news_url) -> (list, bool):
    try:
        response = requests.get(media_url, headers=HEADERS, timeout=10)  # Set an appropriate timeout value (in seconds)
    except requests.exceptions.Timeout:
        # Handle the timeout exception
        print("The request timed out.")
        ErrorReporter(f"Get request timeout exception at {media_url}")
        return [], on_start_checkpoint
    except requests.exceptions.RequestException as e:
        # Handle other request exceptions
        print(f"An error occurred: {str(e)}")
        ErrorReporter(f"Get request general exception, {e}, at {media_url}")
        return [], on_start_checkpoint

    parsed_hmtl = BeautifulSoup(response.content, "html.parser")

    tags_with_url = parsed_hmtl.find_all("a", href=re.compile("https?:.*"))
    
    tags_with_url = pd.Series(tags_with_url).unique().tolist()

    valid_news_urls = []

    for i, tag_with_url in enumerate(tags_with_url):
        news_url = tag_with_url.attrs["href"]
        # Filter out urls with query symbols
        if re.search("[?=&+%@#]{1}", news_url):
            search_spec_char = re.search("[?=&+%@#]{1}", news_url)
            query_start = search_spec_char.span()[0]
            news_url = news_url[:query_start]
        if not re.search("/(\D+-+\D+-?)+/?", news_url) and "html" not in news_url:
            continue
        if news_url.endswith("/"):
            continue
        if re.search("https?:[\/]{2}", news_url):
            if media_url not in news_url:
                continue
        if media_url in news_url or media_url.replace("https://www.", "") in news_url or media_url.replace("http://www.", "") in news_url:
            if on_start_checkpoint:
                if news_url == last_news_url or not last_news_url:
                    on_start_checkpoint = False
                else:
                    continue
            valid_news_urls.append(news_url)
            # Save new checkpoint
            if not on_start_checkpoint and (i % N_SAVE_CHECKPOINT == 0):
                save_news_checkpoint(pid, media_url, news_url)
        #if i > 5:
        #    break
    return list(set(valid_news_urls)), on_start_checkpoint

def find_news_data(media_news_urls, author):
    news_media_data = []
    for news_url in media_news_urls:
        print(news_url)
        resp_url_news = requests.get(news_url, 
                                     headers=HEADERS)
        parsed_news_hmtl = BeautifulSoup(resp_url_news.content, 
                                         "html.parser")
        meta_data = parsed_news_hmtl.select("html head meta[property]")
        try:
            target_keys = ("title", "published_time", "modified_time", "tags", "image")
            data = {}
            data["published_time"] = ""
            data["modified_time"] = ""
            tags = []
            for tag in meta_data:
                _property = tag.attrs["property"].split(":")[-1]
                content = tag.attrs["content"]
                if _property.endswith("tag"):
                    tags.append(content)
                elif _property in target_keys:
                    data[_property] = content
            data["tags"] = ";".join(tags)
            data = find_news_body(parsed_news_hmtl, data)
            if not data["title"]:
                continue
            if not data["body"]:
                data.update(find_news_body_gpt(parsed_news_hmtl))
        except:
            data = find_keys_gpt(parsed_news_hmtl)
        if data is None or (data["title"] is None or data["body"] is None) or ("N/A" in data["title"] or "N/A" in data["body"]):
            continue
        # Skip data element if neither title nor article were found, i.e. None value on both.
        data["url"] = news_url
        if not data["tags"]:
            elements_with_tags = parsed_news_hmtl.find_all("meta", 
                                                           attrs={"name": "news_keywords"})
            data["tags"] = [x.attrs["content"] for x in elements_with_tags]
            data["tags"] = ";".join(data["tags"])
        # TODO complete this
        data["source"] = author
        try:
            data["country"] = parsed_news_hmtl.html.attrs["lang"]
        except:
            data["country"] = ""
        news_media_data.append(data)
    return news_media_data

def find_keys_gpt(parsed_code):
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
        creation_date = regex_creation_datetime.search(message_content).groups()[0]
    except:
        creation_date = None
    try:
        modified_time = regex_update_datetime.search(message_content).groups()[0]
    except:
        modified_time = None
    try:
        body = regex_only_summary.search(message_content).groups()[0]
    except:
        body = None
    data = {
        "n_tokens": n_tokens,
        "title": title,
        "tags": tags,
        "published_time": creation_date,
        "modified_time": modified_time,
        "body": body
        }
    data["image"] = None
    return data

def order_dict_keys(data, ord_keys, only_values=True):
    if only_values:
        return [tuple(dict_elem[key] for key in ord_keys) for dict_elem in data]
    else:
        return [{key: dict_elem[key] for key in ord_keys} for dict_elem in data]
    
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

def process_file(split_file_path):
    # Add your processing logic here
    with open(split_file_path, 'r') as file:
        data = file.read()
        # Process the data from the split_file_path

def split_file_and_process(input_file_path, num_splits, process_function):
    # Split the input file into chunks
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    chunk_size = len(lines) // num_splits
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]

    # Create a process for each chunk and run in parallel
    processes = []
    for i, chunk in enumerate(chunks):
        split_file_path = f"split_{i}.txt"
        with open(os.path.join("data", split_file_path), 'w') as split_file:
            split_file.writelines(chunk)

        process = multiprocessing.Process(target=process_function, args=(split_file_path, chunk, str(i)))
        processes.append(process)
        process.start()

    # Wait for all processes to finish
    for process in processes:
        process.join()

    # Clean up the split files (optional)
    for i in range(num_splits):
        save_news_checkpoint(str(i), "", "")
        split_file_path = os.path.join("data", f"split_{i}.txt")
        os.remove(split_file_path)

def main():
    file_name = "spain_media name_to_url.json"
    file_path = os.path.join(".", "data", file_name)
    if os.path.exists(file_path):
        print("Reading file of urls of regions...")
        name_to_media_urls = read_media_urls_file(file_path)

    region = "Spain"
    print(f"Processing news from {region} region...")
    media_stats_reporter = StatisticsReporter()
    for (_, media_url) in name_to_media_urls.items():
        media_stats_reporter.restart_time()
        #if i > MAX_N_MEDIAS:
        #    break
        #if "elpais" in media_url or "elmundo" in media_url:
        #    continue
        media_url = "https://www." + media_url
        # Extract the author
        author = media_url
        # Read stored news
        urls = find_media_section(media_url)

def main_multi_threading_process(file_path, chunk, pid: str):
    region = "Spain"
    print(f"Processing news from {region} region...")
    #with open("data/plain_spain_media name_to_url.txt", "w") as file_r:
    #    name_to_media_urls = file_r.read().split("\n")
    #print(os.path.join("data", file_path))
    #with open(os.path.join("data", file_path), "w") as file_r:
    #    name_to_media_urls = file_r.read().split("\n")
    media_stats_reporter = StatisticsReporter()
    #for media_line in name_to_media_urls.items():
    #for media_line in name_to_media_urls:
    for media_line in chunk:
        (_, media_url) = media_line.split(";")
        media_url = media_url.strip()
        media_stats_reporter.restart_time()
        #if i > MAX_N_MEDIAS:
        #    break
        #if "elpais" in media_url or "elmundo" in media_url:
        #    continue
        media_url = "https://www." + media_url
        # Extract the author
        author = media_url
        # Read stored news
        urls = find_media_section(media_url, pid)

if __name__ == "__main__":
    current_date, current_time = update_date(current_date, current_time)
    print(f"\n...Datetime of process: {current_date} {current_time}...\n")
    #main()

    num_cores = 6  # Set the number of cores you want to use
    file_name = "plain_spain_media name_to_url.txt"
    file_path = os.path.join(".", "data", file_name)

    # Use the split_file_and_process function to process the file in parallel
    split_file_and_process(file_path, num_cores, main_multi_threading_process)

    conn.commit()
    print("\n...The process ended...")