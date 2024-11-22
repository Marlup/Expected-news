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
    HEADERS,
    MAX_HYPHENS,
    N_EXAMPLES,
    MIN_TOPIC_OCCURRENCE,
    DEFAULT_STEMMER_LANGUAGE,
    REGEX_HTTPS,
    REGEX_NO_HTTPS,
    REGEX_TOPIC,
    REGEX_SUB
)

# Set the stemmer for topic location
DEFAULT_STEMMER = SnowballStemmer(DEFAULT_STEMMER_LANGUAGE)

# Functions utilities for reading files
def read_json_domain_urls_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def read_plain_domain_urls_file(file_path):
    with open(file_path, "r", encoding="latin-1") as file:
        return file.readlines()

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
            "author", "mapa", "blog", "index", "obituario", "visual", "rss"]) or
        re.compile(r"\d{3,}").search(url) or
        ("?" in url or "=" in url or "%" in url) or
        (url.startswith("//") and not re.sub(REGEX_SUB, "", url).startswith(domain)) or
        (not url.startswith("/") and not re.sub(REGEX_SUB, "", url).startswith(domain))
    )

def clean_and_format_url(target_url, response_url):
    """Formats and cleans URL based on initial patterns."""
    if target_url.startswith("//"):
        return "https:" + target_url
    elif target_url.startswith("/"):
        return response_url.rstrip("/") + target_url
    elif target_url.startswith("www."):
        return "https://" + target_url
    return target_url

def fetch_page_content(base_url):
    """Fetches the content of the given URL with retries."""
    try:
        response = requests.get(base_url, headers=HEADERS, timeout=6.0)
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
            response = requests.get(base_url, headers=HEADERS, timeout=6.0)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException:
            return None

def extract_valid_urls(parsed_html, response_url, domain):
    """Extracts and filters valid URLs from the parsed HTML."""
    
    def _validate_url(url, response_url, domain):
        if not re.search(REGEX_HTTPS, url) \
        and not re.search(REGEX_NO_HTTPS, url) \
        and not re.search(REGEX_TOPIC, url):
            return ""
            
        url = clean_and_format_url(url.lower(), response_url)
        if "#" in url:
            url = url.split("#")[0]
        if not url.endswith("/"):
            url += "/"
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

def fetch_valid_urls():
    """Main function to search and process URLs from a file."""
    file_name = "ranking_media.txt"
    file_path = os.path.join("data", file_name)

    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return

    print("Reading file of URLs...")
    domains_to_urls = read_plain_domain_urls_file(file_path)
    valid_urls = []
    
    for line in domains_to_urls:
        _, domain, _ = line.split(";")
        print(f"Fetching domain {domain} ...")
        domain = domain.strip()
        base_url = f"https://{domain}"

        response = fetch_page_content(base_url)
        if response is None:
            continue
        
        parsed_html = BeautifulSoup(response.content, "html.parser")
        urls = extract_valid_urls(parsed_html, response.url, domain)
        
        print(f"\tFound {len(urls)} valid URLs.\n")
        valid_urls.extend(urls)

    with open("valid_urls.txt", "w") as output_file:
        output_file.write("\n".join(valid_urls))
    print(f"Total unique valid links: {len(set(valid_urls))}")
    
    return valid_urls

def get_most_recent_ranking_file(directory="data", prefix="SCImago", pattern="Spanish_*.xlsx"):
    """Return the most recent SCImago ranking file."""
    ranking_files = [(file, datetime.strptime(file.split("_")[-1].split(".")[0], "%d%m%Y")) 
                     for file in glob.glob(f"{directory}/{prefix}*{pattern}")]
    ranking_files_sorted = sorted(ranking_files, key=lambda x: x[1])
    return ranking_files_sorted[-1][0] if ranking_files_sorted else None

def load_and_prepare_general_ranking(file_path, n_examples=N_EXAMPLES):
    """Load general ranking data, sort by 'overall', and clean up the columns."""
    rankings = pd.read_excel(file_path).sort_values("Overall").head(n_examples)
    
    # Change  column names to lowercase for consistency
    old_columns = rankings.columns
    new_columns = [c.lower() for c in old_columns]
    rankings.rename(columns=dict(zip(old_columns, new_columns)), inplace=True)
    
    rankings["media"] = rankings["media"].str.strip()
    rankings["domain"] = rankings["domain"].str.strip()
    rankings.drop("global_rank", axis=1, inplace=True)
    return rankings

def load_and_prepare_special_media(file_name="specialized_medias.csv", n_examples=N_EXAMPLES):
    """Load and clean specialized media data."""
    file_path = os.path.join("data", file_name)
    special_ranks = pd.read_csv(file_path, sep=";").head(n_examples)
    
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

def process_and_normalize_urls(valid_urls):
    """Process and normalize the URLs for analysis."""
    section_urls = valid_urls.str.replace("/{0,2}$", "", regex=True) \
                             .str.replace("([^/][.][^/]$)?", "", regex=True) \
                             .str.extract(".*/{1}(?P<topic>.*)", expand=True)
    section_urls["topic"] = section_urls["topic"].str \
                                                 .replace("[.].*", "", regex=True)
    section_urls["topic_stem"] = section_urls["topic"].apply(lambda x: DEFAULT_STEMMER.stem(x))

    # Extract section and domain from the URLs
    section_urls[["section", "domain"]] = valid_urls.str \
                                                    .extract("(?P<section>https?://(?P<domain>[^/]+/).*)")

    # Calculate topic and domain scores
    #topic_counts = section_urls["topic_stem"].value_counts().rename("topic_score")
    #section_urls = section_urls.merge(
    #    (topic_counts - topic_counts.min()) / (topic_counts.max() - topic_counts.min()), 
    #    left_on="topic_stem", right_index=True, how="inner"
    #)
    #section_urls["topic_score"] = section_urls.se ["topic_stem"].value_counts().rename("topic_score")
    section_urls["topic_score"] = section_urls.groupby("topic_stem")["topic_stem"] \
                                              .transform("count")
    
    #domain_counts = section_urls["domain"].value_counts().rename("domain_score")
    #domain_counts_norm = (domain_counts - domain_counts.min()) / (domain_counts.max() - domain_counts.min())
    #section_urls = section_urls.merge(domain_counts_norm, left_on="domain", right_index=True, how="inner")
    section_urls["domain_score"] = section_urls.groupby("domain")["domain"] \
                                               .transform("count")

    # Get sections that have a minimum count and filter the urls.
    #section_urls_filtered = section_urls["topic"].apply(lambda topic: any(section in topic for section in relevant_sections))
    relevant_topic_stems = section_urls.loc[section_urls["topic_score"] > MIN_TOPIC_OCCURRENCE, "topic_stem"] \
                                       .tolist()
    print(section_urls["topic_score"])
    print(relevant_topic_stems)
    are_relevant_topics = section_urls["topic_stem"].isin(relevant_topic_stems)
    return section_urls[are_relevant_topics]

def merge_with_media_rankings(section_urls, media_ranks):
    """Merge processed URLs with the media rankings and calculate the score."""
    section_urls["temp_domain"] = section_urls["domain"].str.replace("www.", "").str.replace("/", "")
    section_urls.dropna(inplace=True)

    columns_to_save = ["section", "domain", "overall", "domain_score", "topic_score"]

    sections_with_scores = pd.merge(section_urls, media_ranks, left_on="temp_domain", right_on="domain", how="inner")[columns_to_save]
    
    # Normalize the domain score and calculate the final score
    domain_score = sections_with_scores["overall"]
    sections_with_scores["domain_score"] = (domain_score - domain_score.min()) / (domain_score.max() - domain_score.min())
    sections_with_scores["final_score"] = np.sqrt(sections_with_scores[["domain_score", "topic_score"]].pow(2).sum(axis=1))

    return sections_with_scores.drop(["domain_score", "topic_score"], axis=1)

def save_processed_data(sections_with_scores, version_num):
    """Save the processed data to JSON."""
    result = sections_with_scores.groupby('domain')[['section', 'final_score']].apply(lambda x: tuple(x.values)).reset_index()
    result.columns = ['domain', 'data']
    result.set_index("domain").to_json(f"data/medias/medias_urls_v{version_num}.json", orient='index')

def main():
    """Main process for handling SCImago ranking and URL data."""
    most_recent_file = get_most_recent_ranking_file()
    if not most_recent_file:
        raise FileNotFoundError("No SCImago ranking files found.")
    
    # General media preparation
    general_ranking = load_and_prepare_general_ranking(most_recent_file)

    # Special media preparation
    special_ranking = load_and_prepare_special_media()

    # Merge general and special rankings
    merged_ranking = pd.merge(general_ranking, special_ranking, how="outer")

    # Process and analyze URLs
    valid_urls = pd.Series(fetch_valid_urls())
    section_urls = process_and_normalize_urls(valid_urls)
    #print(f"valid_urls: {valid_urls}")
    #print(f"section_urls: {section_urls}")
    #print(f"merged_ranking: {merged_ranking}")

    # Merge URLs with rankings and compute final score
    sections_with_scores = merge_with_media_rankings(section_urls, merged_ranking)

    # Versioning and saving data
    files_section = glob.glob("data/medias/medias_urls_v*.json")
    version_num = max([int(f.split("_v")[-1][:-5]) for f in files_section]) + 1 if files_section else 0
    save_processed_data(sections_with_scores, version_num)

if __name__ == "__main__":
    main()