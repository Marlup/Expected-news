import os
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
PATH_DATA = "../data"
FILE_NAME_EXTRACTION_ERRORS = os.path.join(PATH_DATA, "extraction-errors.txt")
FILE_NAME_PROMPT_ROLE_KEYS = os.path.join(PATH_DATA, "role-prompt-keys-extraction.txt")
FILE_NAME_PROMPT_ROLE_SUMMARY = os.path.join(PATH_DATA, "role-prompt-body-extraction.txt")
# Data structures
ORDER_KEYS = ("url",
              "media_url",
              "creation_datetime",
              "update_datetime",
              "title",
              "description",
              "body", 
              "main_topic",
              "other_topic",
              "image_url",
              "country",
              "n_tokens",
              "score"
              )
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
}
