import os
import re
import requests
import json
from datetime import datetime
import glob as glob

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.stem import SnowballStemmer

from constants import (
    HEADERS_REQUEST,
    MAX_HYPHENS,
    MIN_TOPIC_OCCURRENCE,
    DEFAULT_STEMMER_LANGUAGE,
    PATH_FILE_RANKING,
    DEFAULT_SEPARATOR
)

# Set the stemmer for topic location
DEFAULT_STEMMER = SnowballStemmer(DEFAULT_STEMMER_LANGUAGE)

# Functions utilities for reading files
def read_json_domain_urls_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def is_valid_url(url, domain):
    """Validates URLs against common patterns."""
    return not (
        url.count("-") > MAX_HYPHENS or
        any(substring in url for substring in [
            "@", "php", "javascript", "mailto", "cookie", "feed", "contact", "legal",
            "session", "ads", "publicidad", "privacidad", "condiciones", "tags", 
            "premium", "archiv", "sorteo", "newsletter", "podcast", "logout", "login",
            "notifica", "push", "servicio", "esquela", "defunci", "favorito", "firma", 
            "suscri", "subscrib", "pasatiempo", "compra", "tienda", "gráfico", "grafic", 
            "galeria", "opinion", "hemeroteca", "video", "play", "patrocin", "autor", 
            "author", "mapa", "blog", "index", "obituario", "visual", "rss", "page", 
            "person"]) or
        re.compile(r"\d{3,}").search(url) or
        ("?" in url or "=" in url or "%" in url) or
        (url.startswith("//") and domain not in url) or
        (not url.startswith("/") and domain not in url)
    )

def clean_and_format_url(target_url, response_url):
    """Formats and cleans URL based on initial patterns."""
    new_url = target_url
    
    # Format URL start
    if target_url.startswith("//"):
        new_url = "https:" + target_url
    elif target_url.startswith("/"):
        new_url = response_url.rstrip("/") + target_url
    elif target_url.startswith("www."):
        new_url = "https://" + target_url
    
    # Format URL end
    if "#" in new_url:
        new_url = new_url.split("#")[0]
    if not new_url.endswith("/"):
       new_url += "/"
    
    return new_url

def fetch_page_content(base_url):
    """Fetches the content of the given URL with retries."""
    try:
        response = requests.get(base_url, headers=HEADERS_REQUEST, timeout=6.0)
        response.raise_for_status()
        return response
    except requests.exceptions.Timeout:
        print(f"\tRequest timed out for {base_url}")
        return None
    except requests.exceptions.RequestException:
        # Retry with 'www.' prefix
        base_url = "https://www." + base_url
        print(f"\tRetrying with prefix 'www.': {base_url}")
        try:
            response = requests.get(base_url, headers=HEADERS_REQUEST, timeout=6.0)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException:
            return None

def extract_section_urls(parsed_html, response_url, domain):
    """Extracts and filters valid URLs from the parsed HTML."""
    
    def _validate_url(url, response_url, domain):
        url = clean_and_format_url(url.lower(), response_url)

        # Validate URL from undesired keywords
        if is_valid_url(url, domain):
            return url
        return ""
    
    valid_urls = []
    a_tags = parsed_html.find_all("a", href=True)
    print(f"\tExtracted {len(a_tags)} URLs.")
    for a_tag in a_tags:
        next_url = a_tag.attrs.get("href", "")
        
        next_url = _validate_url(next_url, response_url, domain)
        if next_url != "":
            valid_urls.append(next_url)
    
    return list(set(valid_urls))

def fetch_section_urls(domains):
    """Main function to search and process URLs from a file."""
    section_urls = []
    
    for domain in domains:
        print(f"Fetching domain {domain} ...")
        domain = domain.strip()
        base_url = f"https://{domain}"

        response = fetch_page_content(base_url)
        if response is None:
            continue
        
        parsed_html = BeautifulSoup(response.content, "html.parser")
        urls = extract_section_urls(parsed_html, response.url, domain)
        
        print(f"\tFound {len(urls)} section URLs.\n")
        section_urls.extend(urls)
        
    print(f"Total unique possible section links: {len(set(section_urls))}")
    
    return section_urls

def get_most_recent_ranking_file(file_pattern="SCImago*Spanish*.xlsx"):
    """Return the most recent SCImago ranking file."""
    path = f"data/sources/{file_pattern}"
    
    ranking_files = [(file, datetime.strptime(file.split("_")[-1].split(".")[0], "%d%m%Y")) 
                     for file in glob.glob(path)]
    ranking_files_sorted = sorted(ranking_files, key=lambda x: x[1])
    return ranking_files_sorted[-1][0] if ranking_files_sorted else None

def extract_general_media(file_path):
    """Load general ranking data, sort by 'overall', and clean up the columns."""
    rankings = pd.read_excel(file_path).sort_values("Overall")
    
    # Change  column names to lowercase for consistency
    old_columns = rankings.columns
    new_columns = [c.lower() for c in old_columns]
    rankings.rename(columns=dict(zip(old_columns, new_columns)), inplace=True)
    
    rankings["media"] = rankings["media"].str.strip()
    rankings["domain"] = rankings["domain"].str.strip()
    rankings.drop("global_rank", axis=1, inplace=True)
    return rankings

def extract_special_media(file_name="special_media.csv"):
    """Load and clean special media data."""
    file_path = os.path.join("data", "sources", file_name)
    special_ranks = pd.read_csv(file_path, sep=DEFAULT_SEPARATOR)
    
    # Change  column names to lowercase for consistency
    old_columns = special_ranks.columns
    new_columns = [c.lower() for c in old_columns]
    special_ranks.rename(columns=dict(zip(old_columns, new_columns)), inplace=True)
    
    special_ranks["media"] = special_ranks["media"].str.strip()
    special_ranks["domain"] = special_ranks["domain"].str.strip()
    special_ranks["overall"] = 1.0  # Default score for special medias
    special_ranks["country"] = "Spain"
    special_ranks["region"] = "Western Europe"
    special_ranks["language"] = "Spanish"
    return special_ranks

def search_valid_urls(urls):
    """Perform web scraping and filtering for valid URLs, returning a DataFrame with URL and topic."""
    from urllib.parse import urlparse
    
    filtered_urls = []
    topics = []

    for url in urls:
        parsed_url = urlparse(url)
        path_segments = parsed_url.path.strip("/").split('/')
        topic = path_segments[0]
        
        # Check if the domain has exactly one segment in the path
        if len(path_segments) == 1 and topic != "":
            filtered_urls.append(url)
            topics.append(topic)  # Capture the topic name without slashes

    # Create a DataFrame with the filtered URLs and their corresponding topics
    df = pd.DataFrame({
        'url': filtered_urls,
        'topic': topics
    })

    return df

def stem_topic(df):
    """Process and normalize the URLs for analysis."""
    #section_urls = valid_urls.str.replace("/{0,2}$", "", regex=True) \
    #                         .str.replace("([^/][.][^/]$)?", "", regex=True) \
    #                         .str.extract(".*/{1}(?P<topic>.*)", expand=True)
    #section_urls["topic"] = section_urls["topic"].str.replace("[.].*", "", regex=True)
    
    df["topic_stem"] = df["topic"].apply(lambda x: DEFAULT_STEMMER.stem(x))
    return df

def extract_domain_from_url(df):
    # Extract section and domain from the URLs
    #section_urls[["section", "domain"]] = valid_urls.str.extract("(?P<topic>https?://(?P<domain>[^/]+/).*)")
    df["domain"] = df["url"].str.extract(r"^(?:https?:\/\/)?(?P<domain>[^\/]+)\/?.*")
    return df

def compute_and_topic_scores(df):
    # Calculate topic and domain scores
    #topic_counts = section_urls["topic_stem"].value_counts().rename("topic_score")
    #section_urls = section_urls.merge(
    #    (topic_counts - topic_counts.min()) / (topic_counts.max() - topic_counts.min()), 
    #    left_on="topic_stem", right_index=True, how="inner"
    #)
    #section_urls["topic_score"] = section_urls.se ["topic_stem"].value_counts().rename("topic_score")
    df["topic_score"] = df.groupby("topic_stem")["topic_stem"] \
                          .transform("count")
    
    #domain_counts = section_urls["domain"].value_counts().rename("domain_score")
    #domain_counts_norm = (domain_counts - domain_counts.min()) / (domain_counts.max() - domain_counts.min())
    #section_urls = section_urls.merge(domain_counts_norm, left_on="domain", right_index=True, how="inner")
    df["domain_score"] = df.groupby("domain")["domain"] \
                           .transform("count")
    return df

def get_records_by_topic_frequency(df):
    # Get sections that have a minimum count and filter the urls.
    #section_urls_filtered = section_urls["topic"].apply(lambda topic: any(section in topic for section in relevant_sections))
    relevant_topic_stems = df.loc[df["topic_score"] > MIN_TOPIC_OCCURRENCE, "topic_stem"] \
                             .tolist()
                             
    are_relevant_topics = df["topic_stem"].isin(relevant_topic_stems)
    return df[are_relevant_topics]

def merge_with_ranking(section_urls, df_ranking):
    """Merge processed URLs with the media rankings and calculate the score."""
    section_urls["temp_domain"] = section_urls["domain"].str.replace("www.", "").str.replace("/", "")
    section_urls.dropna(inplace=True)

    columns_to_save = ["domain", "url", "topic", "domain_score", "topic_score", "overall"]

    sections_with_scores = pd.merge(section_urls, df_ranking, left_on="temp_domain", right_on="domain", how="inner")
    sections_with_scores.drop("domain_y", axis=1, inplace=True)
    sections_with_scores.rename(columns={"domain_x": "domain"}, inplace=True)
    sections_with_scores = sections_with_scores[columns_to_save]
    
    # Normalize the domain score and calculate the final score
    domain_score = sections_with_scores["overall"]
    sections_with_scores["domain_score"] = (domain_score - domain_score.min()) / (domain_score.max() - domain_score.min())
    sections_with_scores["score"] = np.sqrt(sections_with_scores[["domain_score", "topic_score"]].pow(2).sum(axis=1))

    return sections_with_scores.drop(["domain_score", "topic_score"], axis=1)

def save_processed_data(sections_with_scores, version_num):
    """Save the processed data to JSON."""
    print(f"save_processed_data 1: {sections_with_scores}")
    
    #result = sections_with_scores.groupby("domain")[["topic", "score"]].apply(lambda x: tuple(x.values)).reset_index()
    #result.set_index("domain").to_json(f"data/sources/source_urls_v{version_num}.json", orient='index')
    
    #sections_with_scores[["domain", "url", "score"]].to_csv(f"data/sources/source_urls_v{version_num}.csv", index=False)

    #print("a", result.head())

    result = sections_with_scores.groupby("domain")[["url", "topic", "score"]] \
                                 .apply(lambda x: tuple(x.values)) \
                                 .to_frame("records")
    
    #print("a", result.head())    
    result.to_json(f"data/sources/source_urls_v{version_num}.json", orient='index')

def main():
    """Main process for handling SCImago ranking and URL data."""
    on_build_ranking = False
    
    if on_build_ranking or not os.path.exists("data/sources/ranking_media.csv"):
        most_recent_file = get_most_recent_ranking_file()
        if not most_recent_file:
            raise FileNotFoundError("No SCImago ranking files found.")
        
        # General media preparation
        general_ranking = extract_general_media(most_recent_file)

        # Special media preparation
        special_ranking = extract_special_media()

        # Merge general and special rankings
        merged_ranking = pd.merge(general_ranking, special_ranking, how="outer")
        merged_ranking.to_csv(PATH_FILE_RANKING, sep=DEFAULT_SEPARATOR, index=False)

    else:
        merged_ranking = pd.read_csv(PATH_FILE_RANKING, sep=DEFAULT_SEPARATOR)

    raw_section_urls = fetch_section_urls(merged_ranking["domain"])
    
    # Process and analyze URLs
    df_section_urls_plus_data = search_valid_urls(raw_section_urls)
    
    df_section_urls_plus_data = stem_topic(df_section_urls_plus_data)
    df_section_urls_plus_data = extract_domain_from_url(df_section_urls_plus_data)
    df_section_urls_plus_data = compute_and_topic_scores(df_section_urls_plus_data)
    df_section_urls_plus_data = get_records_by_topic_frequency(df_section_urls_plus_data)

    # Merge URLs with rankings and compute final score
    df_main_sources = merge_with_ranking(df_section_urls_plus_data, merged_ranking)
    
    # Versioning and saving data
    files_section = glob.glob("data/sources/source_urls_v*.json")
    version_num = max([int(f.split("_v")[-1][:-5]) for f in files_section]) + 1 if files_section else 0
    save_processed_data(df_main_sources, version_num)

if __name__ == "__main__":
    main()