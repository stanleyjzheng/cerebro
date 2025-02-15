import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def get_images(url, folder="downloaded_images"):
    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return
    
    soup = BeautifulSoup(response.text, 'html.parser')
    images = soup.find_all('img')
    
    if not images:
        print("No images found on the page.")
        return
    
    print(f"Found {len(images)} images. Downloading...")
    
    for img in images:
        img_url = img.get("src")
        if not img_url:
            continue
        
        img_url = urljoin(url, img_url)
        
        try:
            img_data = requests.get(img_url, headers=headers).content
            img_name = os.path.join(folder, os.path.basename(img_url).split("?")[0])
            
            with open(img_name, "wb") as img_file:
                img_file.write(img_data)
                print(f"Downloaded: {img_name}")
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {img_url}: {e}")

if __name__ == "__main__":
    website_url = input("Enter the website URL: ")
    get_images(website_url)
