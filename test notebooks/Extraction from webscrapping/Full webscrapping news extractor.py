import json
from pprint import pprint
import os
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# Constants
API_SOURCE = "mediastack"

def get_parent_p_tags(parsed_hmtl):
    p_tags = parsed_hmtl.find_all("p")
    common_parent_p_tags = None
    for p_tag in p_tags:
        parent_p_tags = p_tag.parent.find_all("p", recursive=False)
        if common_parent_p_tags is None or len(parent_p_tags) > len(common_parent_p_tags):
            common_parent_p_tags = parent_p_tags
            print("\nNumber of p tags:", len(common_parent_p_tags))
            print(common_parent_p_tags)
    return common_parent_p_tags

def main():

    current_date, time = str(datetime.today()).split(" ")
    current_date

    path = os.path.join("News storage", API_SOURCE, current_date)

    file_name = current_date + "_" + API_SOURCE + "_" + "extracted_news.json"
    file_path = os.path.join(path, file_name)
    
    url = input("Introduce url from news media:\n")

    response = requests.get(url)
    print(response.status_code)  # HTTP status code

    html = BeautifulSoup(response.text)
    html.text

    common_parent_p_tags = get_parent_p_tags(html)
    # show text
    p_tags_text = [x.text for x in common_parent_p_tags]
    print(p_tags_text)

if __name__ == "main":
    main()
