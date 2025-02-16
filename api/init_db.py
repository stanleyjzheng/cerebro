#!/usr/bin/env python3
"""
This script initializes the SQLite database (`embeddings.db`) with the correct schema.

It ensures a table exists to store embeddings along with metadata:
- `file_path` (TEXT): The original file path of the image or video.
- `youtube_video_id` (TEXT, optional): If the image is from a YouTube video, stores the video ID.
- `timestamp` (REAL, optional): If the image is a video frame, stores the timestamp in seconds.
- `media_type` (TEXT): Either "image" or "video_frame".
- `embedding` (BLOB): The vectorized embedding stored as a pickled NumPy array.

Usage:
    python init_db.py
"""

import sqlite3
import os

DB_PATH = "embeddings.db"

def init_db():
    """Creates `embeddings.db` with the required schema if it does not already exist."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT UNIQUE,
            youtube_video_id TEXT,
            timestamp REAL,
            media_type TEXT CHECK(media_type IN ('image', 'video_frame')),
            embedding BLOB
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database `{DB_PATH}` initialized successfully.")

if __name__ == "__main__":
    init_db()
