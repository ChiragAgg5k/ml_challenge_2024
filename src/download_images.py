import os
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import re

def extract_unique_filename(url):
    # Extract the unique filename using regex
    match = re.search(r'/([A-Za-z0-9]+)\.jpg$', url)
    if match:
        return match.group(1)
    else:
        return url.split("/")[-1]  # Fallback to the old method if regex fails

def download_image(url, SAVE_DIR):
    try:
        img_data = requests.get(url, timeout=5).content
        unique_filename = extract_unique_filename(url)
        img_name = os.path.join(SAVE_DIR, f"{unique_filename}.jpg")
        with open(img_name, 'wb') as handler:
            handler.write(img_data)
        return url, True
    except Exception as e:
        return url, False

def download_images_in_parallel(image_urls, SAVE_DIR):
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(lambda url: download_image(url, SAVE_DIR), image_urls), total=len(image_urls)))
    return results

if __name__ == "__main__":
    SAVE_DIR = '../images'
    DATASET_FOLDER = 'dataset/'

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    train_data = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    image_urls = train_data['image_link'].tolist()

    results = download_images_in_parallel(image_urls, SAVE_DIR)

    successful_downloads = [url for url, status in results if status]
    failed_downloads = [url for url, status in results if not status]

    print(f"Successfully downloaded {len(successful_downloads)} images.")
    if failed_downloads:
        print(f"Failed to download {len(failed_downloads)} images.")