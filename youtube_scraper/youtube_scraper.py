#!/usr/bin/env python3
"""
video_scraper_selenium.py

A Selenium-based scraper for normal video streams.

Features:
- Opens a list of video URLs (assumed to be normal videos).
- Uses Selenium to load the video page.
- Automatically presses the play button and waits for it to vanish, indicating that the video has started playing.
- Captures a screenshot of the video element (or the full page if not found) every N seconds.
- Saves images to ~/Desktop/video_images.
- Calls a hook function after each image capture for further processing.

Requirements:
- Selenium (`pip install selenium`)
- ChromeDriver (or an equivalent driver) installed and in your PATH.
- Google Chrome installed (if using ChromeDriver).

Usage:
    python video_scraper_selenium.py --interval 5
"""

import os
import sys
import time
import datetime
import logging
import argparse
import re

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


from PIL import Image
import pytesseract

def slugify(text):
    """
    Create a safe slug from the given video URL.
    The slug is derived from a part of the URL.
    """
    parts = text.split("/")
    if len(parts) > 3:
        return parts[3]
    return "unknown"

def image_hook(image_path):
    """
    Hook function that is called after a new image is captured.
    Customize this function to process the image (e.g., compute embeddings,
    push to a database, etc.).
    """
    logging.info(f"Hook: New image saved at {image_path}")
    # Insert your custom processing code here.
    # For example:
    #   embedding = compute_embedding(image_path)
    #   push_to_db(embedding)

def init_driver():
    """
    Initializes and returns a Selenium WebDriver instance with headless Chrome.
    Make sure ChromeDriver is installed and in your PATH.
    """
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1280,720")

    driver = webdriver.Chrome(options=chrome_options)
    return driver

def capture_screenshot(driver, output_path):
    """
    Attempts to capture a screenshot of the video element.
    If the video element cannot be found, captures a full page screenshot.
    Saves the screenshot to output_path.
    """
    try:
        video = driver.find_element(By.TAG_NAME, "video")
        video.screenshot(output_path)
    except Exception:
        logging.info("Video element not found or screenshot failed; capturing full page.")
        driver.save_screenshot(output_path)

def press_play_button(driver, timeout=30):
    """
    Automatically finds and clicks the play button on the video.
    Assumes the play button has the class 'ytp-large-play-button'.
    Waits until the button is visible, clicks it, and then waits until it disappears.
    """
    try:
        logging.info("Looking for play button to press...")
        play_button = WebDriverWait(driver, timeout).until(
            EC.visibility_of_element_located((By.CLASS_NAME, "ytp-large-play-button"))
        )
        logging.info("Play button found. Clicking play...")
        play_button.click()
        WebDriverWait(driver, timeout).until(
            EC.invisibility_of_element_located((By.CLASS_NAME, "ytp-large-play-button"))
        )
        logging.info("Play button disappeared, video is now playing.")
    except Exception as e:
        logging.warning(f"Could not press play button: {e}. Continuing anyway.")

def extract_timestamp_from_image(image_path, crop_box=None):
    """
    Extracts the video timestamp from the given screenshot image using OCR.

    Args:
        image_path (str): Path to the screenshot image.
        crop_box (tuple, optional): A tuple (left, upper, right, lower) defining
                                    the region of the image to OCR. If None, the entire image is used.
                                    Adjust this as needed based on where the timestamp is displayed.

    Returns:
        str or None: The extracted timestamp in format HH:MM:SS or MM:SS, or None if not found.
    """
    try:
        image = Image.open(image_path)
        if crop_box is not None:
            image = image.crop(crop_box)
        # Convert image to grayscale to improve OCR accuracy.
        gray = image.convert('L')
        ocr_result = pytesseract.image_to_string(gray)
        logging.debug(f"OCR result: {ocr_result}")
        # Look for a timestamp pattern, e.g., 00:01:23 or 01:23.
        pattern = r"(\d{1,2}:\d{2}(?::\d{2})?)"
        matches = re.findall(pattern, ocr_result)
        if matches:
            return matches[0].replace(" ", "")
    except Exception as e:
        logging.error(f"Error extracting timestamp: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Normal Video Scraper using Selenium")
    parser.add_argument("--interval", type=int, default=5, help="Interval between scrapes in seconds (default: 5)")
    args = parser.parse_args()

    # Directory to store images.
    home = os.path.expanduser("~")
    image_dir = os.path.join(home, "Desktop", "video_images")
    os.makedirs(image_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting Normal Video Scraper (Selenium-based)")

    # List of videos.
    videos = [
        {
            "id": slugify("https://www.youtube.com/watch?v=aAD8eV06OcU&ab_channel=PeacefulRelaxation"),
            "url": "https://www.youtube.com/watch?v=aAD8eV06OcU&ab_channel=PeacefulRelaxation"
        },
        # You can add more videos here.
    ]

    # Initialize a driver for each video.
    drivers = {}
    for video in videos:
        vid_id = video.get("id", "unknown")
        logging.info(f"Initializing driver for video {vid_id}")
        try:
            driver = init_driver()
            driver.get(video.get("url"))
            # press play for me pls
            press_play_button(driver, timeout=30)
            drivers[vid_id] = driver
        except Exception as e:
            logging.error(f"Failed to initialize driver for {vid_id}: {e}")

    try:
        while True:
            start_time = time.time()
            for video in videos:
                vid_id = video.get("id", "unknown")
                logging.info(f"Processing video {vid_id}")
                driver = drivers.get(vid_id)
                if driver is None:
                    logging.error(f"No driver for video {vid_id}, skipping.")
                    continue

                try:
                    time.sleep(5)  # 5 mimimimimis
                    #base file?
                    base_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    slug = slugify(video["url"])
                    temp_filename = f"{base_timestamp}_{slug}_temp.png"
                    temp_path = os.path.join(image_dir, temp_filename)

                    # make temp filee
                    capture_screenshot(driver, temp_path)
                    logging.info(f"Captured temporary screenshot for {vid_id} at {temp_path}")

                    # Use OCR to extract the timestamp????
                    video_timestamp = extract_timestamp_from_image()
                    if video_timestamp:
                        new_filename = f"{base_timestamp}_{video_timestamp}_{slug}.png"
                    else:
                        new_filename = f"{base_timestamp}_{slug}.png"
                    new_path = os.path.join(image_dir, new_filename)

                    os.rename(temp_path, new_path)
                    logging.info(f"Renamed screenshot to {new_path}")

                    capture_screenshot(driver, new_path)
                    logging.info(f"Captured image for {vid_id} saved to {new_path}")
                    image_hook(new_path)
                except Exception as e:
                    logging.error(f"Error processing video {vid_id}: {e}")

            elapsed = time.time() - start_time
            sleep_time = max(0, args.interval - elapsed)
            logging.info(f"Sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        logging.info("Exiting scraper...")
    finally:
        # Close all driver instances.
        for vid_id, driver in drivers.items():
            driver.quit()

if __name__ == "__main__":
    main()
