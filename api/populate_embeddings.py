"""
Indexer Script (Iteration 2 with Fix): Processes images and videos in a directory to compute embeddings 
using our ViT‑B‑32 model and pushes them to a FAISS index. This version uses batched processing for speed,
supports multi‑GPU (via DataParallel), and fixes the attribute error when calling encode_image.
Each embedding is associated with metadata (file_path, youtube_video_id, timestamp, media_type).

Usage:
    uv run populate_embeddings.py /path/to/directory
"""

import os
import argparse
import pickle
import cv2
import numpy as np
import torch
from PIL import Image
import faiss
import open_clip
from tqdm import tqdm  # progress bar

# Paths for saving FAISS index and metadata.
INDEX_PATH = "faiss_index.index"
METADATA_PATH = "faiss_metadata.pkl"

def load_existing_index_and_metadata():
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        print("Loading existing FAISS index and metadata...")
        index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
    else:
        index = None
        metadata = []
    return index, metadata

def save_index_and_metadata(index, metadata):
    print("Saving FAISS index and metadata...")
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

def record_exists(metadata, file_path, timestamp):
    for record in metadata:
        if record["file_path"] == file_path and record.get("timestamp") == timestamp:
            return True
    return False

# Set device: use GPU (CUDA) if available, or M1 Metal (mps), else CPU.
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

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

model = model.to(device)
model.eval()

# --- Multi-GPU support ---
if device == "cuda" and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)
# ---------------------------

# Helper function to handle encode_image for both single GPU and DataParallel setups.
def encode_image(x):
    if hasattr(model, "module"):
        return model.module.encode_image(x)
    return model.encode_image(x)

def add_embedding(index, embedding_np):
    if index is None:
        d = embedding_np.shape[1]
        index = faiss.IndexFlatIP(d)
    index.add(embedding_np)
    return index

def process_images_batch(image_paths, index, metadata, batch_size=320):
    unindexed_paths = [p for p in image_paths if not record_exists(metadata, p, None)]
    if not unindexed_paths:
        return index, metadata

    for i in tqdm(range(0, len(unindexed_paths), batch_size), desc="Indexing images"):
        batch_paths = unindexed_paths[i : i + batch_size]
        images = []
        valid_paths = []
        for file_path in batch_paths:
            try:
                image = Image.open(file_path).convert("RGB")
                images.append(preprocess(image))
                valid_paths.append(file_path)
            except Exception as e:
                print(f"Failed to open image {file_path}: {e}")
        if not images:
            continue
        batch_tensor = torch.stack(images).to(device)
        with torch.no_grad():
            embeddings = encode_image(batch_tensor)
        embeddings_np = embeddings.cpu().detach().numpy()
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        embeddings_np = embeddings_np / np.maximum(norms, 1e-10)
        index = add_embedding(index, embeddings_np)
        for file_path in valid_paths:
            metadata.append({
                "file_path": file_path,
                "youtube_video_id": None,
                "timestamp": None,
                "media_type": "image"
            })
            print(f"Indexed image: {file_path}")
    return index, metadata

def process_video(file_path, index, metadata, frame_interval=1.0, youtube_video_id=None, batch_size=320):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Failed to open video {file_path}")
        return index, metadata

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    steps = int(np.floor(duration / frame_interval)) + 1

    frames = []
    timestamps = []
    current_time = 0.0
    pbar = tqdm(total=steps, desc=f"Processing video {os.path.basename(file_path)}", leave=False)

    while current_time < duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
        ret, frame = cap.read()
        if not ret:
            break

        if record_exists(metadata, file_path, current_time):
            print(f"Skipping video frame {file_path} at {current_time}s (already indexed)")
            current_time += frame_interval
            pbar.update(1)
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        frames.append(preprocess(image))
        timestamps.append(current_time)

        if len(frames) == batch_size:
            batch_tensor = torch.stack(frames).to(device)
            with torch.no_grad():
                embeddings = encode_image(batch_tensor)
            embeddings_np = embeddings.cpu().detach().numpy()
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            embeddings_np = embeddings_np / np.maximum(norms, 1e-10)
            index = add_embedding(index, embeddings_np)
            for ts in timestamps:
                metadata.append({
                    "file_path": file_path,
                    "youtube_video_id": youtube_video_id,
                    "timestamp": ts,
                    "media_type": "video_frame"
                })
                print(f"Indexed video frame from {file_path} at {ts}s")
            frames = []
            timestamps = []

        current_time += frame_interval
        pbar.update(1)

    if frames:
        batch_tensor = torch.stack(frames).to(device)
        with torch.no_grad():
            embeddings = encode_image(batch_tensor)
        embeddings_np = embeddings.cpu().detach().numpy()
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        embeddings_np = embeddings_np / np.maximum(norms, 1e-10)
        index = add_embedding(index, embeddings_np)
        for ts in timestamps:
            metadata.append({
                "file_path": file_path,
                "youtube_video_id": youtube_video_id,
                "timestamp": ts,
                "media_type": "video_frame"
            })
            print(f"Indexed video frame from {file_path} at {ts}s")
    pbar.close()
    cap.release()
    return index, metadata

def process_directory(directory, index, metadata):
    image_paths = []
    video_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            ext = os.path.splitext(file)[1].lower()
            if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
                image_paths.append(full_path)
            elif ext in [".mp4", ".mov", ".avi", ".mkv"]:
                video_paths.append(full_path)
    
    index, metadata = process_images_batch(image_paths, index, metadata)
    for video_path in tqdm(video_paths, desc="Indexing videos"):
        index, metadata = process_video(video_path, index, metadata, frame_interval=1.0, youtube_video_id=None)
    return index, metadata

def main():
    parser = argparse.ArgumentParser(
        description="Index images and videos into a FAISS index for similarity search (batched & multi-GPU)"
    )
    parser.add_argument("directory", type=str, help="Directory containing images/videos")
    args = parser.parse_args()

    index, metadata = load_existing_index_and_metadata()
    index, metadata = process_directory(args.directory, index, metadata)
    save_index_and_metadata(index, metadata)
    print("Indexing complete.")

if __name__ == "__main__":
    main()
