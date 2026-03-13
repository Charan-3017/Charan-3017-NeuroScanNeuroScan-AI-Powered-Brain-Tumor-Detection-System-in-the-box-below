
import os
import requests
import zipfile

ARTICLE_ID = 1512427
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}/files"
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def download_file(url, filename):
    local_path = os.path.join(DATA_DIR, filename)
    print(f"Downloading {filename}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Saved to {local_path}")
    return local_path

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print(f"Fetching file list for article {ARTICLE_ID}...")
    response = requests.get(API_URL)
    response.raise_for_status()
    files = response.json()

    for file_info in files:
        filename = file_info['name']
        download_url = file_info['download_url']
        
        # Only download the .zip files (the dataset is usually split into 4 parts)
        if filename.endswith('.zip'):
            file_path = download_file(download_url, filename)
            
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            
            # Optional: remove zip file after extraction to save space using os.remove(file_path)
            # keeping it for now in case of re-run needs

    print("Download and extraction complete.")

if __name__ == "__main__":
    main()
