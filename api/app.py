#!/usr/bin/env python3
"""
API Server for FAISS-based Similarity Search

This API loads a FAISS index and metadata created by your revised indexing script.
It uses your custom CLIP model checkpoint (./epoch_12.pt) to encode text and image queries.
Endpoints:
  - GET /search/text: Searches using a text query.
  - POST /search/image: Searches using an uploaded image.
Optional filtering by media type is supported (e.g. filter_media=image).

Usage:
    uvicorn app:app --reload
"""

import os
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
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths for the FAISS index and metadata produced by your indexing script.
INDEX_PATH = "faiss_index.index"
METADATA_PATH = "faiss_metadata.pkl"

def load_faiss_index_and_metadata():
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        raise RuntimeError("FAISS index or metadata not found. Please run your indexing script first.")
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# Load the global FAISS index and metadata.
global_index, global_metadata = load_faiss_index_and_metadata()
print(f"Loaded FAISS index with {global_index.ntotal} vectors.")

# Set device: prefer CUDA if available, then MPS (for M1), else CPU.
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
print(f"Using device: {device}")

# Load the CLIP model and preprocessing transforms.
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None)
# Load checkpoint
checkpoint_path = "./epoch_12.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Load checkpoint

# Check if the checkpoint contains a state_dict key (some PyTorch checkpoints do)
if "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

# Load the checkpoint into the model
model.load_state_dict(checkpoint, strict=False)  # strict=False allows missing keys

model = model.to(device)
model.eval()

# If using multiple GPUs, wrap the model in DataParallel.
if device == "cuda" and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for inference")
    model = torch.nn.DataParallel(model)

def encode_image(x: torch.Tensor) -> torch.Tensor:
    """Helper to call encode_image correctly whether or not model is wrapped in DataParallel."""
    if hasattr(model, "module"):
        return model.module.encode_image(x)
    return model.encode_image(x)

# For text tokenization.
import open_clip.tokenizer as tokenizer

def get_filtered_faiss_index(filter_media: Optional[str]):
    """
    If filter_media is provided ("image" or "youtube"), rebuild a FAISS index using only
    those entries whose metadata matches the filter. Returns (new_index, filtered_metadata).
    """
    if global_index is None or global_metadata is None:
        return None, []
    filtered_indices = []
    for i, record in enumerate(global_metadata):
        if filter_media == "image" and record["media_type"] != "image":
            continue
        if filter_media == "youtube" and record["youtube_video_id"] is None:
            continue
        filtered_indices.append(i)
    if len(filtered_indices) == 0:
        return None, []
    # Reconstruct vectors for each index entry.
    all_vectors = np.array([global_index.reconstruct(i) for i in range(global_index.ntotal)], dtype=np.float32)
    filtered_vectors = all_vectors[filtered_indices]
    d = filtered_vectors.shape[1]
    new_index = faiss.IndexFlatIP(d)
    new_index.add(filtered_vectors)
    filtered_metadata = [global_metadata[i] for i in filtered_indices]
    return new_index, filtered_metadata

def perform_faiss_search(query_vector: np.ndarray, index, metadata, top_n: int):
    """Search the given FAISS index with the normalized query_vector."""
    query_vector = query_vector.astype(np.float32).reshape(1, -1)
    scores, indices = index.search(query_vector, top_n)
    scores = scores[0]
    indices = indices[0]
    results = []
    for score, idx in zip(scores, indices):
        if idx < len(metadata):
            results.append((score, metadata[idx]))
    return results

# Response model.
class SearchResponseItem(BaseModel):
    file_path: str
    youtube_video_id: Optional[str]
    timestamp: Optional[float]
    media_type: str
    score: float

@app.get("/search/text", response_model=List[SearchResponseItem])
def search_text(query: str, top_n: int = 5, filter_media: Optional[str] = None):
    """
    Search using a text query.
      - query: the text prompt.
      - top_n: number of top results.
      - filter_media: "image", "youtube", or omitted for all.
    """
    text_tokens = open_clip.tokenize([query]).to(device)
    with torch.no_grad():
        query_embedding = model.encode_text(text_tokens)
    query_vector = query_embedding.cpu().numpy()[0]
    query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-10)

    if filter_media is None:
        index = global_index
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
    Search using an uploaded image.
      - file: the image file.
      - top_n: number of top results.
      - filter_media: "image", "youtube", or omitted for all.
    """
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        query_embedding = encode_image(input_tensor)
    query_vector = query_embedding.cpu().numpy()[0]
    query_vector = query_vector / (np.linalg.norm(query_vector) + 1e-10)

    if filter_media is None:
        index = global_index
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
