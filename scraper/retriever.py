import os
import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from urllib.parse import urljoin

def get_images(url, folder="downloaded_images"):
    # Create folder if it doesn't exist
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Set up Selenium WebDriver (headless mode for faster execution)
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")

    # Start Selenium WebDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    try:
        print(f"Fetching {url}...")
        driver.get(url)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(5)  # Wait for JavaScript to load images

        # Scroll down to load more images
        last_height = driver.execute_script("return document.body.scrollHeight")
        for _ in range(3):  # Adjust the range to scroll more
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)  # Wait for images to load
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # Extract page source after JS loads
        soup = BeautifulSoup(driver.page_source, "html.parser")
        images = soup.find_all("img", class_="img")

        if not images:
            print("No observation images found.")
            return
        
        print(f"Found {len(images)} images. Downloading...")

        for idx, img in enumerate(images):
            img_url = img.get("data-src") or img.get("src")
            if not img_url:
                continue

            img_url = urljoin(url, img_url)

            try:
                img_data = requests.get(img_url).content
                img_name = os.path.join(folder, f"image_{idx}.jpg")
                
                with open(img_name, "wb") as img_file:
                    img_file.write(img_data)
                    print(f"Downloaded: {img_name}")
            except requests.exceptions.RequestException as e:
                print(f"Failed to download {img_url}: {e}")

    finally:
        driver.quit()

if __name__ == "__main__":
    website_url = input("Enter the iNaturalist location URL: ")
    get_images(website_url)
