import json
import os
import requests
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime, timedelta, timezone
import time
import multiprocessing

PATH_DATA = "../data"
DIGITAL_MEDIAS_URL = "https://www.prensaescrita.com/prensadigital.php"
DIGITAL_MEDIAS_MAIN_ROOT = "https://www.prensaescrita.com"
DIGITAL_MEDIAS_URL = "https://www.prensaescrita.com/prensadigital.php"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
}

def update_date(current_date, current_time):
    new_date, new_time = str(datetime.today()).split(" ")
    if new_date != current_date:
        return new_date, new_time
    return current_date, current_time

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

def _get_url_medias_from_region(url):
    response = requests.get(url, headers=HEADERS)
    parsed_hmtl = BeautifulSoup(response.content, "html.parser")
    links = parsed_hmtl.find_all("a", string=lambda text: text and "www." in text)
    return [l["href"] for l in links]

def _save_media_urls(file_path, data, on_override=False):
    if not os.path.exists(file_path) or on_override:
        with open(file_path, "w") as file:
            json.dump(data, file)
def read_media_urls_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def save_checkpoint(pid, last_value):
    with open(os.path.join(PATH_DATA, f"checkpoint_{pid}.json"), "w") as file:
        json.dump({"last_value": last_value}, file)
def read_news_checkpoint(pid: str):
    with open(os.path.join(PATH_DATA, f"checkpoint_{pid}.json"), "r") as file:
        data = json.load(file)
        return data["last_value"]

def read_media_sections(pid: str):
    with open(os.path.join(PATH_DATA, f"media_sections_{pid}.txt"), "r") as file:
        sections = file.readline()
        return sections

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

class StatisticsManager():
    def __init__(self) -> None:
        self.last_process_duration = -1.0
    
    def restart_time(self):
        self.time_start = time.time()
        return self
    def set_process_duration(self, 
                             round_digits: int=1
                             ):
        self.last_process_duration = round(time.time() - self.time_start, 
                                           round_digits)
        self.time_start = time.time()
        return self.last_process_duration

    def write_extraction_stats(self, data, 
                               header=None, 
                               input_iso_datetime="",
                               pid: str=0, 
                               main_folder=os.path.join(PATH_DATA, "scrapping-statistics"),
                               subfolder=""
                               ):
        process_time = str(self.set_process_duration())
        if input_iso_datetime:
            dt_fmt = input_iso_datetime.strftime("%Y-%m-%d_%H %M %S")
            date = input_iso_datetime.strftime("%Y-%m-%d")
        else:
            #iso_datetime = datetime.now().replace(tzinfo=timezone.utc)
            #input_date, input_time = str(datetime.today()).split(" ")
            #input_time = input_time.split(" ")[-1].split(".")[0].replace(":", "-")
            process_dt = datetime.now().replace(tzinfo=timezone.utc)
            dt_fmt = process_dt.strftime("%Y-%m-%d_%H %M %S")
            date = process_dt.strftime("%Y-%m-%d")
        
        if not subfolder:
            subfolder = f"processes_{date}"
        if isinstance(data, tuple):
            data = list(data)
        if not os.path.exists(os.path.join(main_folder, subfolder)):
            os.makedirs(os.path.join(main_folder, subfolder))

        stats_path = os.path.join(main_folder, 
                                  subfolder, 
                                  f"process_{dt_fmt}_pid_{pid}.csv",
                                  )
        with open(stats_path, "a") as file:
            if not os.path.exists(stats_path):
                if header is None:
                    header = "url;news_count;process_time\n"
                file.write(header)
            file.write(";".join([str(x) for x in data] + [process_time + "\n"]))

class FileManager():
    def __init__(self) -> None:
        self.files_map = {}
    def add_files(self, 
                  files: list, 
                  **kwargs
                  ):
        for file_name in files:
            self._add_file(file_name, 
                           **kwargs)
    
    def _add_file(self, 
                  file_name: str,
                  **kwargs
                  ):
        if kwargs.get("open_mode", False):
            open_mode = kwargs["open_mode"]
        else:
            open_mode = "a"
        
        self.files_map[file_name] = open(file_name, open_mode)
    def write_on_file(self, 
                      file_name: str,
                      msgs: [list[dict], tuple[dict]]
                      ):
        lock = multiprocessing.Lock()
        with lock:
            for msg in msgs:
                self._write_on_file(file_name, 
                                    f"{msg['status_code']};{msg['id']}\n")
    
    def _write_on_file(self, 
                       file_name: str,
                       msg: str
                       ):
        self.files_map[file_name].write(msg)
    def close_all_files(self):
        for file in self.files_map.values():
            file.close()