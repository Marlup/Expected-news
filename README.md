# Web Scraping for News Data Extraction

# Table of Contents

1. [Functions](#functions)
   - [extract_data_from_jsons](#extract_data_from_jsons)
   - [extract_data_from_metadata](#extract_data_from_metadata)
   - [extract_keys_with_gpt](#extract_keys_with_gpt)
   - [find_body_with_gpt](#find_body_with_gpt)
   - [get_body_summary](#get_body_summary)
   - [treat_raw_urls](#treat_raw_urls)
   - [find_valid_urls](#find_valid_urls)
   - [find_urls_data](#find_urls_data)
   - [run_multi_process](#run_multi_process)
   - [main_multi_threading_process](#main_multi_threading_process)
   - [filter_old_url](#filter_old_url)
   - [order_dict_keys](#order_dict_keys)
   - [read_stored_news and create_news_table](#read_stored_news-and-create_news_table)
   - [insert_news](#insert_news)
2. [Running the Script](#running-the-script)

This Python script is designed to extract news data from pre-extracted news media section URLs, filter and clean the data, optionally summarize it using the GPT API, and store the information in a structured database. The script is organized into several functions, each serving a specific purpose in the data extraction process.

## Functions:

### `extract_data_from_jsons`

This function extracts news data from JSON objects within the HTML content of a news webpage. It looks for various attributes like the article body, article type, creation date, and more. The data is stored in a dictionary and returned.

### `extract_data_from_metadata`

This function extracts additional data from meta tags in the HTML of the news webpage. It retrieves information like the title, description, creation date, and more. The extracted data is added to the dictionary passed to the function.

### `extract_keys_with_gpt`

This function uses the GPT API to extract additional keys from the text content of the webpage. It looks for relevant information such as the number of tokens, title, tags, creation date, and more.

### `find_body_with_gpt`

This function extracts the main body of the news article using the GPT API. It processes the text content of the webpage and returns the article body and the number of tokens.

### `get_body_summary`

This function uses the GPT API to summarize the article body, providing a brief summary of the news content. It takes the text, URL, and media name as input and returns the summarized body and the number of tokens.

### `treat_raw_urls`

This function processes a list of raw news URLs, filtering out invalid or undesirable URLs. It also interacts with a database to determine which URLs are already stored.

### `find_valid_urls`

This function filters out invalid URLs based on various criteria, including query symbols, file extensions, and specific URL patterns. It also checks if URLs are already present in the database.

### `find_urls_data`

This function processes a list of news URLs, extracts the necessary data from each URL, and stores it in a structured format. It also identifies URLs that couldn't be processed.

### `run_multi_process`

This function orchestrates the multi-process execution of the web scraping script. It divides the workload among multiple processes, each handling a specific set of media URLs.

### `main_multi_threading_process`

This function represents the main logic for processing news URLs in a multi-threaded environment. It iterates through media URLs, retrieves the HTML content of news sections, and processes the news URLs.

### `filter_old_url`

This function checks the publication date of a news article to determine if it's too old. If the article is older than a specified threshold, it is filtered out.

### `order_dict_keys`

This function arranges the keys in a dictionary in a specified order, making the resulting dictionary more structured and easier to work with.

### `read_stored_news` and `create_news_table`

These functions interact with a database to store and retrieve news data. `read_stored_news` retrieves URLs that are already stored, and `create_news_table` initializes the database table if it doesn't exist.

### `insert_news`

This function inserts news data into the database table, making it available for later retrieval and analysis.

## Running the Script

The script can be run to extract news data from various media sources. It processes a list of media URLs and their associated sections, extracting news URLs, and storing the data in a structured database. The script is designed to be executed in a multi-process and multi-threading environment to improve efficiency.

Please note that the script relies on external APIs, so you'll need to set up the necessary API keys and credentials for GPT and the database.

To run the script, simply execute it in a Python environment, and it will start processing the specified media URLs.

This script can be a valuable tool for collecting and analyzing news data from various sources, helping you stay informed and updated.

Happy web scraping!
