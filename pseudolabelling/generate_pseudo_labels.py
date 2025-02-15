import os
import glob
import json
import torch
import open_clip
from PIL import Image
import webdataset as wds
from io import BytesIO
from tqdm import tqdm
import concurrent.futures

# --- Configuration ---
IMAGE_PATTERN = "animals/animals/**/*.jpg"  
PSEUDO_LABELS_FILE = "pseudo_labels.txt"    # ~700 label strings (one per line
OUTPUT_TAR = "pseudo_labeled_dataset.tar"
BATCH_SIZE = 32
NUM_WORKERS = 8

# --- Device Selection ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Load Label Strings ---
with open(PSEUDO_LABELS_FILE, "r") as f:
    labels = [line.strip() for line in f if line.strip()]
num_labels = len(labels)
print(f"Loaded {num_labels} labels.")

# --- Load Pretrained CLIP Model for Pseudo Labeling ---
# TODO: swap in our own model here
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
model.eval()

# --- Precompute Text Embeddings ---
with torch.no_grad():
    text_tokens = open_clip.tokenize(labels).to(device)
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.cpu()

# --- Prepare WebDataset Writer ---
writer = wds.TarWriter(OUTPUT_TAR)
image_paths = glob.glob(IMAGE_PATTERN, recursive=True)
print(f"Found {len(image_paths)} images.")

# --- Utility Function: Load & Preprocess an Image ---
def load_and_preprocess(path):
    try:
        with open(path, "rb") as f:
            img_bytes = f.read()
        # Use BytesIO so that we don’t need to re-read the file from disk
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        preprocessed = preprocess(img)  # Returns a tensor
        key = os.path.splitext(os.path.basename(path))[0]
        return key, img_bytes, preprocessed
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

# --- Process Images in Batches with a Thread Pool & tqdm Progress Bar ---
batch_data = []
pbar = tqdm(total=len(image_paths), desc="Processing Images", unit="img")

with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(load_and_preprocess, path): path for path in image_paths}
    for future in concurrent.futures.as_completed(futures):
        result = future.result()
        pbar.update(1)
        if result is None:
            continue
        key, img_bytes, preprocessed = result
        batch_data.append((key, img_bytes, preprocessed))
        if len(batch_data) >= BATCH_SIZE:
            # Stack preprocessed tensors to form a batch
            keys, img_bytes_list, preprocessed_list = zip(*batch_data)
            batch_tensor = torch.stack(preprocessed_list).to(device)
            with torch.no_grad():
                image_features = model.encode_image(batch_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu()
            # Compute similarities (batch_size x num_labels)
            similarities = image_features @ text_features.t()
            best_idxs = similarities.argmax(dim=-1).tolist()
            best_scores = similarities.max(dim=-1).values.tolist()
            for key, img_bytes, best_idx, score in zip(keys, img_bytes_list, best_idxs, best_scores):
                best_label = labels[best_idx]
                sample = {
                    "__key__": key,
                    "jpg": img_bytes,
                    "cls": best_label,
                    "json": json.dumps({"pseudo_label": best_label, "score": score}),
                }
                writer.write(sample)
            batch_data = []

# Process any remaining images
if batch_data:
    keys, img_bytes_list, preprocessed_list = zip(*batch_data)
    batch_tensor = torch.stack(preprocessed_list).to(device)
    with torch.no_grad():
        image_features = model.encode_image(batch_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu()
    similarities = image_features @ text_features.t()
    best_idxs = similarities.argmax(dim=-1).tolist()
    best_scores = similarities.max(dim=-1).values.tolist()
    for key, img_bytes, best_idx, score in zip(keys, img_bytes_list, best_idxs, best_scores):
        best_label = labels[best_idx]
        sample = {
            "__key__": key,
            "jpg": img_bytes,
            "cls": best_label,
            "json": json.dumps({"pseudo_label": best_label, "score": score}),
        }
        writer.write(sample)

pbar.close()
writer.close()
print(f"Pseudo-labeled dataset saved to {OUTPUT_TAR}.")
