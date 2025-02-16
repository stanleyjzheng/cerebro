import csv
import os
import requests
import threading

# Configuration
CSV_FILE = "image_urls.csv"  # Path to your CSV file
DOWNLOAD_FOLDER = "images"   # Folder to save downloaded images
MAX_THREADS = 1000  # Number of concurrent downloads

# Create the folder if it doesn't exist
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

def download_image(url, file_path):
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded: {file_path}")
        else:
            print(f"Failed to download {url}: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def process_row(row, index):
    if len(row) < 2:
        print(f"Skipping row {index}: not enough columns.")
        return

    image_name = row[0].strip()
    url = row[1].strip()

    # Derive file extension from URL headers if possible
    file_extension = url.split('.')[-1].split('?')[0]
    if len(file_extension) > 5 or '/' in file_extension:
        file_extension = 'jpg'  # Default fallback

    safe_image_name = "".join(c for c in image_name if c.isalnum() or c in (' ', '.', '_')).rstrip()
    file_name = f"{safe_image_name}.{file_extension}"
    file_path = os.path.join(DOWNLOAD_FOLDER, file_name)
    
    download_image(url, file_path)

def main():
    threads = []
    
    with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader):
            if len(threads) >= MAX_THREADS:
                for t in threads:
                    t.join()
                threads.clear()
            
            t = threading.Thread(target=process_row, args=(row, index))
            t.start()
            threads.append(t)
    
    # Ensure all threads finish
    for t in threads:
        t.join()

if __name__ == "__main__":
    main()
