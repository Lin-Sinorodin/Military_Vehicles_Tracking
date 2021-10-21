import io
import os
import time
import hashlib
import requests
from PIL import Image
from kora.selenium import wd
from tqdm.autonotebook import tqdm


def get_urls_from_search(search_url, wd, sleep_between=1, page_scrolls=1):
    """Returns a list of images url from a given google images search_url"""                     
    def scroll_to_end(wd_old):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between)    
    
    # load the page
    wd.get(search_url)
    if page_scrolls:
        for i in range(page_scrolls):
            scroll_to_end(wd)

    # get all image thumbnail results
    thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")

    image_urls = set()
    for img in tqdm(thumbnail_results):
        # try to click every thumbnail to get the real image behind it
        try:
            img.click()
            time.sleep(sleep_between)
        except Exception:
            continue

        # extract image urls    
        actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
        for actual_image in actual_images:
            curr_url = actual_image.get_attribute('src')
            if curr_url and 'http' in curr_url:
                image_urls.add(curr_url)

    return image_urls


def save_image_from_url(folder_path: str, url: str, verbose: bool=True):
        success = False
        try:
            image_content = requests.get(url).content

        except Exception as e:
            print(f"ERROR - Could not download {url} - {e}")
            
        try:
            image_file = io.BytesIO(image_content)
            Image.url = ''
            image = Image.open(image_file).convert('RGB')
            image.url = url
            file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
            with open(file_path, 'wb') as f:
                image.save(f, "JPEG", quality=85)
            if verbose:  print(f"SUCCESS - saved {url} - as {file_path}")
            success = True
        except Exception as e:
            print(f"ERROR - Could not save {url} - {e}")
        return success