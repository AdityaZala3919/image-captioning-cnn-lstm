import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Set these constants
DATASET_SLUG = "adityajn105/flickr8k"
DOWNLOAD_PATH = "data/Flickr8k"

def download_and_extract():
    api = KaggleApi()
    api.authenticate()

    os.makedirs(DOWNLOAD_PATH, exist_ok=True)

    print("🔽 Downloading dataset from Kaggle...")
    api.dataset_download_files(DATASET_SLUG, path=DOWNLOAD_PATH, unzip=False)

    zip_path = os.path.join(DOWNLOAD_PATH, f"{DATASET_SLUG.split('/')[-1]}.zip")
    
    if os.path.exists(zip_path):
        print("📦 Extracting ZIP...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DOWNLOAD_PATH)
        os.remove(zip_path)
        print("✅ Done!")
    else:
        print("❌ Zip file not found.")

if __name__ == "__main__":
    download_and_extract()
