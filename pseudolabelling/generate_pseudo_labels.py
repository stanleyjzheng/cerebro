import os
import glob
import json
import torch
import torch.nn as nn
import open_clip
from PIL import Image
import webdataset as wds
from io import BytesIO
from tqdm import tqdm
import concurrent.futures

# --- Configuration ---
IMAGE_PATTERN = "./dataset/"  
PSEUDO_LABELS_FILE = "pseudo_labels.txt" 
OUTPUT_TAR = "pseudo_labeled_dataset.tar"
BATCH_SIZE = 1920
NUM_WORKERS = 22

# --- Getting all image files ---
# Define image file extensions
image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp", "*.tiff", "*.webp")

# Use glob to find all images recursively
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(f'{IMAGE_PATTERN}/**/{ext}', recursive=True))

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
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')

# Load checkpoint
checkpoint_path = "/workspace/open_clip/src/logs/2025_02_15-22_49_52-model_ViT-B-32-lr_0.0005-b_480-j_4-p_amp/checkpoints/epoch_12.pt"
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

# --- Wrap the model to enable DataParallel ---
# DataParallel splits the batch and calls the forward() method on each sub-batch.
class CLIPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, images):
        # Compute image features and normalize
        feats = self.model.encode_image(images)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

wrapper = CLIPWrapper(model)
# If you have 4 GPUs, use them
if torch.cuda.device_count() >= 4:
    device_ids = [0, 1, 2, 3]
    wrapper = nn.DataParallel(wrapper, device_ids=device_ids)
    print(f"Using DataParallel on GPUs: {device_ids}")
else:
    print("Not enough GPUs for DataParallel, running on a single device.")

wrapper = wrapper.to(device)

# --- Precompute Text Embeddings ---
with torch.no_grad():
    text_tokens = open_clip.tokenize(labels).to(device)
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.cpu()

# --- Prepare WebDataset Writer ---
writer = wds.TarWriter(OUTPUT_TAR)
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
                # The wrapper now distributes the batch over 4 GPUs
                image_features = wrapper(batch_tensor)
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
        image_features = wrapper(batch_tensor)
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
