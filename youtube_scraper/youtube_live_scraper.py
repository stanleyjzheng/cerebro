#!/usr/bin/env python3
"""
wildlife_scraper_selenium.py

A Selenium-based scraper for live wildlife camera streams.

Features:
- Opens a list of wildlife camera URLs (assumed to be live streams).
- Uses Selenium to load the embed page.
- Checks if the stream is live by looking for a live badge element.
- Captures a screenshot of the video element (or the full page if not found) every N seconds.
- Saves images to ~/Desktop/wildlife_images.
- Calls a hook function after each image capture for further processing.

Requirements:
- Selenium (`pip install selenium`)
- ChromeDriver (or an equivalent driver) installed and in your PATH.
- Google Chrome installed (if using ChromeDriver).

Usage:
    python wildlife_scraper_selenium.py --interval 5
"""

import os
import sys
import time
import datetime
import logging
import argparse

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

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

def is_live(driver, timeout=10):
    """
    Checks if the YouTube embed page indicates a live stream.
    Returns True if a live badge element is found, else False.
    Note: YouTube embed pages typically include a ".ytp-live-badge" element when live.
    """
    try:
        wait = WebDriverWait(driver, timeout)
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".ytp-live-badge")))
        return True
    except Exception:
        logging.info("Live badge not found; stream may not be live.")
        return False

def capture_screenshot(driver, output_path):
    """
    Attempts to capture a screenshot of the video element.
    If the video element cannot be found, captures a full page screenshot.
    Saves the screenshot to output_path.
    """
    try:
        # Try to locate the video element.
        video = driver.find_element(By.TAG_NAME, "video")
        video.screenshot(output_path)
    except Exception:
        logging.info("Video element not found or screenshot failed; capturing full page.")
        driver.save_screenshot(output_path)

def main():
    parser = argparse.ArgumentParser(description="Wildlife Camera Scraper using Selenium")
    parser.add_argument("--interval", type=int, default=5, help="Interval between scrapes in seconds (default: 5)")
    args = parser.parse_args()

    # Directory to store images.
    home = os.path.expanduser("~")
    image_dir = os.path.join(home, "Desktop", "wildlife_images")
    os.makedirs(image_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting Wildlife Camera Scraper (Selenium-based)")

    # List of wildlife cameras.
    cameras = [
        {
            "id": "camera1",
            "url": "https://www.youtube.com/watch?v=WW-Rs9lnZNM&ab_channel=ExploreLiveNatureCams"
        },
        # You can add more cameras here.
    ]

    # Initialize a driver for each camera.
    drivers = {}
    for camera in cameras:
        cam_id = camera.get("id", "unknown")
        logging.info(f"Initializing driver for camera {cam_id}")
        try:
            driver = init_driver()
            driver.get(camera.get("url"))
            drivers[cam_id] = driver
        except Exception as e:
            logging.error(f"Failed to initialize driver for {cam_id}: {e}")

    try:
        while True:
            start_time = time.time()
            for camera in cameras:
                cam_id = camera.get("id", "unknown")
                cam_url = camera.get("url")
                logging.info(f"Processing camera {cam_id}")
                driver = drivers.get(cam_id)
                if driver is None:
                    logging.error(f"No driver for camera {cam_id}, skipping.")
                    continue

                try:
                    # Refresh the page to update the live status.
                    driver.refresh()
                    # Wait a bit for the page to load after refresh.
                    time.sleep(5)  # Adjust as needed for your network/setup.

                    if not is_live(driver):
                        logging.info(f"Camera {cam_id} is not live; skipping screenshot.")
                        continue

                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"{cam_id}_{timestamp}.png"
                    output_path = os.path.join(image_dir, output_filename)

                    capture_screenshot(driver, output_path)
                    logging.info(f"Captured image for {cam_id} saved to {output_path}")
                    image_hook(output_path)
                except Exception as e:
                    logging.error(f"Error processing camera {cam_id}: {e}")

            elapsed = time.time() - start_time
            sleep_time = max(0, args.interval - elapsed)
            logging.info(f"Sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        logging.info("Exiting scraper...")
    finally:
        # Close all driver instances.
        for cam_id, driver in drivers.items():
            driver.quit()

if __name__ == "__main__":
    main()
