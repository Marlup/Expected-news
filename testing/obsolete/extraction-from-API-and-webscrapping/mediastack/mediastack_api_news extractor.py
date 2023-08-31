import http.client, urllib.parse
import json
from datetime import datetime
import os

# Constants
API_SOURCE = "mediastack"
API_KEY = "f0c57024677e5fcea8cb4ecdfc780e70"
MAX_LIMIT = 100
OFFSET_STEP = 100

# Env variables
current_date, time = str(datetime.today()).split(" ")
countries = "gb"
languages = "en,es"
# Connection
conn = http.client.HTTPConnection('api.mediastack.com')

# Available categories
"""
general - Uncategorized News
business - Business News
entertainment - Entertainment News
health - Health News
science - Science News
sports - Sports News
technology - Technology News
"""
# Available languages
"""
ar - Arabic
de - German
en - English
es - Spanish
fr - French
he - Hebrew
it - Italian
nl - Dutch
no - Norwegian
pt - Portuguese
ru - Russian
se - Swedish
zh - Chinese
"""

# Available countries
"""
 Argentina ar  
 Australia au  
 Austria at  
 Belgium be  
 Brazil br  
 Bulgaria bg  
 Canada ca  
 China cn  
 Colombia co  
 Czech Republic cz  
 Egypt eg  
 France fr  
 Germany de  
 Greece gr  
 Hong Kong hk  
 Hungary hu  
 India in  
 Indonesia id  
 Ireland ie  
 Israel il  
 Italy it  
 Japan jp  
 Latvia lv  
 Lithuania lt  
 Malaysia my  
 Mexico mx  
 Morocco ma  
 Netherlands nl  
 New Zealand nz  
 Nigeria ng  
 Norway no  
 Philippines ph  
 Poland pl  
 Portugal pt  
 Romania ro  
 Saudi Arabia sa
 Serbia rs  
 Singapore sg  
 Slovakia sk  
 Slovenia si  
 South Africa za
 South Korea kr
 Sweden se  
 Switzerland ch  
 Taiwan tw  
 Thailand th  
 Turkey tr  
 UAE ae  
 Ukraine ua  
 United Kingdom gb
 United States us
 Venuzuela ve
 """

params = urllib.parse.urlencode({
    'access_key': API_KEY,
	'country': countries,
    'languages': languages,
    'categories': '-general,-sports',
    'date': current_date,
    'sort': "popularity", #'published_desc',
    'offset': 0,
    'limit': MAX_LIMIT,
    })

conn.request('GET', '/v1/news?{}'.format(params))

res = conn.getresponse()
data_bytes = res.read()
data_str = data_bytes.decode('utf-8')
print(data_str)

path = os.path.join("News storage", API_SOURCE, current_date)
if not os.path.exists(path): 
    os.makedirs(path)

file_name = current_date + "_" + API_SOURCE + "_" + "extracted_news.json"
file_path = os.path.join(path, file_name)
with open(file_path, "w") as f:
    json.dump(json.loads(data_str), f)
