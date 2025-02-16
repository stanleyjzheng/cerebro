"""
This script creates an API (using FastAPI) that serves two endpoints:
1. A GET endpoint (/search/text) that accepts a text query and returns the top N images/frames 
   (with an optional filter for images, YouTube videos, or both) based on cosine similarity.
2. A POST endpoint (/search/image) that accepts an image upload and returns the top N similar images/frames.

Embeddings are loaded from the SQLite database and indexed using FAISS for fast similarity search.
"""

import sqlite3
import pickle
import io
from typing import List, Optional

import numpy as np
import torch
import faiss
import open_clip
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from PIL import Image

app = FastAPI()
DB_PATH = "embeddings.db"

def get_db_connection():
    return sqlite3.connect(DB_PATH)

# --- Device and Model Setup ---
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"


model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None)
# Load checkpoint
checkpoint_path = "./epoch_12.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Load checkpoint

# Check if the checkpoint contains a state_dict key (some PyTorch checkpoints do)
if "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

# Load the checkpoint into the model
model.load_state_dict(checkpoint, strict=False)  # strict=False allows missing keys

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)
model.eval()

# For text tokenization.
import open_clip.tokenizer as tokenizer

# --- Global Embeddings and FAISS Index ---
# These globals are built at startup from the DB.
global_metadata = []         # List of dicts: each record's metadata.
global_embeddings_matrix = None  # Numpy array (n, d) of normalized embeddings.
global_faiss_index = None      # FAISS index built over all embeddings.

def load_all_embeddings_from_db():
    """
    Load all embeddings from the DB.
    Returns a tuple (metadata_list, embeddings_matrix).
    Each metadata record is a dict and the embedding is converted to a numpy array.
    """
    conn = get_db_connection()
    c = conn.cursor()
    query = "SELECT id, file_path, youtube_video_id, timestamp, media_type, embedding FROM embeddings"
    c.execute(query)
    rows = c.fetchall()
    conn.close()

    metadata = []
    vectors = []
    for row in rows:
        id_, file_path, youtube_video_id, timestamp, media_type, emb_blob = row
        emb_array = pickle.loads(emb_blob)
        vec = np.array(emb_array, dtype=np.float32)
        metadata.append({
            "id": id_,
            "file_path": file_path,
            "youtube_video_id": youtube_video_id,
            "timestamp": timestamp,
            "media_type": media_type
        })
        vectors.append(vec)
    if len(vectors) == 0:
        return metadata, None

    vectors = np.stack(vectors)
    # Normalize each vector for cosine similarity.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.maximum(norms, 1e-10)
    return metadata, vectors

def build_faiss_index(vectors: np.ndarray):
    """
    Build a FAISS index (IndexFlatIP) for cosine similarity on normalized vectors.
    """
    if vectors is None or vectors.shape[0] == 0:
        return None
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)  # Inner product is equivalent to cosine similarity if vectors are normalized.
    index.add(vectors)
    return index

def get_filtered_faiss_index(filter_media: Optional[str]):
    """
    If a filter is provided (e.g., "image" or "youtube"), filter the global metadata
    and embeddings, then build a temporary FAISS index.
    Returns (index, filtered_metadata).
    """
    if global_metadata is None or global_embeddings_matrix is None:
        return None, []
    filtered_indices = []
    filtered_metadata = []
    for i, record in enumerate(global_metadata):
        if filter_media == "image" and record["media_type"] != "image":
            continue
        if filter_media == "youtube" and record["youtube_video_id"] is None:
            continue
        filtered_indices.append(i)
        filtered_metadata.append(record)
    if len(filtered_indices) == 0:
        return None, []
    filtered_vectors = global_embeddings_matrix[filtered_indices]
    index = build_faiss_index(filtered_vectors)
    return index, filtered_metadata

# Build the global index at startup.
global_metadata, global_embeddings_matrix = load_all_embeddings_from_db()
global_faiss_index = build_faiss_index(global_embeddings_matrix)
print(f"Loaded {len(global_metadata)} embeddings into FAISS index.")

# --- Utility Functions ---
def perform_faiss_search(query_vector: np.ndarray, index, metadata, top_n: int):
    """
    Search the FAISS index using the query_vector.
    Returns a list of tuples (score, metadata_record).
    """
    if index is None:
        return []
    query_vector = query_vector.astype(np.float32)
    # FAISS expects queries as 2D arrays.
    query_vector = np.expand_dims(query_vector, axis=0)
    scores, indices = index.search(query_vector, top_n)
    scores = scores[0]
    indices = indices[0]
    results = []
    for score, idx in zip(scores, indices):
        if idx < len(metadata):
            results.append((score, metadata[idx]))
    return results

# --- Response Model ---
class SearchResponseItem(BaseModel):
    file_path: str
    youtube_video_id: Optional[str]
    timestamp: Optional[float]
    media_type: str
    score: float

# --- API Endpoints ---
@app.get("/search/text", response_model=List[SearchResponseItem])
def search_text(query: str, top_n: int = 5, filter_media: Optional[str] = None):
    """
    Search by a text query.
    - **query**: the text prompt.
    - **top_n**: number of top results to return.
    - **filter_media**: "image", "youtube", or omitted for all.
    """
    # Tokenize and encode the text query.
    text_tokens = open_clip.tokenize([query]).to(device)
    with torch.no_grad():
        query_embedding = model.encode_text(text_tokens)
    # Convert to numpy and normalize.
    query_vector = query_embedding.cpu().numpy()[0]
    query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-10)

    # Choose global or filtered index.
    if filter_media is None:
        index = global_faiss_index
        metadata = global_metadata
    else:
        index, metadata = get_filtered_faiss_index(filter_media)
        if index is None:
            return []

    results = perform_faiss_search(query_vector, index, metadata, top_n)
    # Sort results in descending order of score.
    results.sort(key=lambda x: x[0], reverse=True)
    response = []
    for score, record in results:
        response.append(SearchResponseItem(
            file_path=record["file_path"],
            youtube_video_id=record["youtube_video_id"],
            timestamp=record["timestamp"],
            media_type=record["media_type"],
            score=float(score)
        ))
    return response

@app.post("/search/image", response_model=List[SearchResponseItem])
async def search_image(
    file: UploadFile = File(...),
    top_n: int = Form(5),
    filter_media: Optional[str] = Form(None)
):
    """
    Search by uploading an image file.
    - **file**: the image file.
    - **top_n**: number of top results to return.
    - **filter_media**: "image", "youtube", or omitted for all.
    """
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        query_embedding = model.encode_image(input_tensor)
    query_vector = query_embedding.cpu().numpy()[0]
    query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-10)

    # Choose global or filtered index.
    if filter_media is None:
        index = global_faiss_index
        metadata = global_metadata
    else:
        index, metadata = get_filtered_faiss_index(filter_media)
        if index is None:
            return []

    results = perform_faiss_search(query_vector, index, metadata, top_n)
    results.sort(key=lambda x: x[0], reverse=True)
    response = []
    for score, record in results:
        response.append(SearchResponseItem(
            file_path=record["file_path"],
            youtube_video_id=record["youtube_video_id"],
            timestamp=record["timestamp"],
            media_type=record["media_type"],
            score=float(score)
        ))
    return response
