import csv
import os
import requests

# Configuration
CSV_FILE = "image_urls.csv"  # Path to your CSV file
DOWNLOAD_FOLDER = "images"   # Folder to save downloaded images

# Create the folder if it doesn't exist
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

def download_image(url, file_path):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded: {file_path}")
        else:
            print(f"Failed to download {url}: HTTP {response.status_code}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")

def main():
    with open(CSV_FILE, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for index, row in enumerate(reader):
            if len(row) < 2:
                print(f"Skipping row {index}: not enough columns.")
                continue
            image_name = row[0].strip()
            url = row[1].strip()

            # Derive the file extension from the URL or default to jpg
            file_extension = url.split('.')[-1].split('?')[0]
            if len(file_extension) > 5 or '/' in file_extension:  # crude check for invalid extension
                file_extension = 'jpg'
            
            # Ensure the image name is safe for use as a filename
            safe_image_name = "".join(c for c in image_name if c.isalnum() or c in (' ', '.', '_')).rstrip()
            file_name = f"{safe_image_name}.{file_extension}"
            file_path = os.path.join(DOWNLOAD_FOLDER, file_name)
            download_image(url, file_path)

if __name__ == "__main__":
    main()
