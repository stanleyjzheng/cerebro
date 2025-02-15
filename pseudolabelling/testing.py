import os
import glob
import json
import torch
import open_clip
from PIL import Image
import webdataset as wds

# --- Configuration ---
# Path pattern for your images (adjust as needed)
IMAGE_PATTERN = "path/to/images/**/*.jpg"  
# Path to your pseudo label list (one label per line)
PSEUDO_LABELS_FILE = "pseudo_labels.txt"  
# Output tar file for the WebDataset
OUTPUT_TAR = "pseudo_labeled_dataset.tar"

# --- Device selection ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Load label strings ---
with open(PSEUDO_LABELS_FILE, "r") as f:
    labels = [line.strip() for line in f if line.strip()]
num_labels = len(labels)
print(f"Loaded {num_labels} labels.")

# --- Load a pretrained CLIP model for pseudo labeling ---
# (using OpenCLIP’s "ViT-B-32" teacher model here)
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
model = model.to(device)
model.eval()

# --- Precompute text embeddings for the 700 labels ---
with torch.no_grad():
    text_tokens = open_clip.tokenize(labels).to(device)
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.cpu()

# --- Prepare WebDataset writer ---
writer = wds.TarWriter(OUTPUT_TAR)

# --- Process each image ---
image_paths = glob.glob(IMAGE_PATTERN, recursive=True)
print(f"Found {len(image_paths)} images.")

for image_path in image_paths:
    try:
        # Open and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)

        # Compute image features
        with torch.no_grad():
            image_feature = model.encode_image(image_input)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)
        image_feature = image_feature.cpu()

        # Compute similarity scores against all label text embeddings
        similarity = (image_feature @ text_features.t()).squeeze(0)  # (num_labels,)
        best_idx = similarity.argmax().item()
        best_label = labels[best_idx]
        best_score = similarity[best_idx].item()

        # Read raw image bytes (to save the original file)
        with open(image_path, "rb") as img_f:
            img_bytes = img_f.read()

        # Create a record with a unique key
        key = os.path.splitext(os.path.basename(image_path))[0]
        sample = {
            "__key__": key,
            "jpg": img_bytes,
            "cls": best_label,  # you could also store the index if you prefer
            "json": json.dumps({"pseudo_label": best_label, "score": best_score}),
        }
        writer.write(sample)
    except Exception as e:
        print(f"Failed to process {image_path}: {e}")

writer.close()
print(f"Pseudo-labeled dataset saved to {OUTPUT_TAR}.")
