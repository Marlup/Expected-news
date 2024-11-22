import os
## Globals (mutable)
ASSISTANT_ID = ""
## Constants
FULL_START = True
BLOCK_API_CALL = True
N_BATCH_ELEMENTS = 20
DB_TIMEOUT = 10.0
MEDIA_GET_REQ_TIMEOUT = 10.0
NEWS_ARTICLE_GET_REQ_TIMEOUT = 8.0
SECONDS_IN_DAY = 86_400
N_MAX_DAYS_OLD = 1
NUM_CORES = 10
MAX_N_MEDIAS = 50
N_SAVE_CHECKPOINT = 5
ASSISTANT_ID = ""

# Constants for medias process
MAX_HYPHENS = 1
N_EXAMPLES = 1000
MIN_TOPIC_OCCURRENCE = 5
DEFAULT_STEMMER_LANGUAGE = "spanish"
REGEX_SEARCH_URL_1_SEGMENT = r"^(?:\w*:?)\/{2}[^\/]+\/[^\/]*\/?$"
REGEX_SEARCH_PROTOCOL = r"^(?!(\w*:?\/{1,2}))[^\/]+\/[^\/]+\/?$"
REGEX_SEARCH_TOPIC = r"^\/[^\/-]+-?[^\/-]*\/?$"
REGEX_SUB = r"^(?:\w*:?\/{1,2})"

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
STATUS_5_1 = "5_1"
STATUS_5_2 = "5_2"
STATUS_5_3 = "5_3"
STATUS_5_4 = "5_4"
STATUS_5_5 = "5_5"
STATUS_5_6 = "5_6"
# File and path names
DB_NAME_NEWS = "db.sqlite3"
DIGITAL_MEDIAS_URL = "https://www.prensaescrita.com/prensadigital.php"
DIGITAL_MEDIAS_MAIN_ROOT = "https://www.prensaescrita.com"
PATH_DATA = "data"
PATH_STATS = "data/webscraping_statistics"
PATH_ERRORS = "data/logs/errors"
PATH_GARBAGE_URLS = "data/garbage_urls"
PATH_MEDIA_SECTIONS_FILE = "data/sources/source_urls_v*.json"
FILE_PATH_EXTRACTION_ERRORS = "data/logs/extraction_errors.txt"
#FILE_PATH_PROMPT_ROLE_KEYS = os.path.join(PATH_DATA, "role_prompt_keys.txt")
FILE_PATH_PROMPT_ROLE_SUMMARY = "data/sources/role_prompt_body.txt"
# Data structures
ORDER_KEYS = (
    "url",
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
# Symbols
SYMBOLS = (
    "?",
    "¿",
    "!",
    "¡",
    "*",
    "&",
    "#",
    "(",
    ")",
    "%"
)
# HTTP setups
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
}

# Query strings
INSERT_GARBAGE_QUERY = """
    INSERT INTO garbage
        (
        url,
        mediaUrl,
        insertDatetime,
        statusCode
        )
            VALUES
        (
            ?,
            ?,
            DATETIME('now', 'localtime', 'utc'),
            ?
        )
        ;
"""
INSERT_NEWS_QUERY = """
    INSERT INTO news
        (
            url,
            mediaUrl,
            creationDate,
            insertDate,
            updateDate,    
            title,
            description,
            articleBody,
            mainTopic,
            otherTopic,
            imageUrl,
            country,
            nTokens,
            score,
            preprocessed
        )
    VALUES
        (
            ?,
            ?,
            ?,
            DATETIME('now', 'localtime', 'utc'),
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            False
        )
    ;
"""

SELECT_ALL_URLS_IN_MEDIA_QRY_STR = """
    SELECT 
        url
    FROM 
        news
    WHERE
        mediaUrl = ?
    ;
"""
