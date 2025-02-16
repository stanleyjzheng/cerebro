import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import open_clip
import webdataset as wds
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Device & Distributed Setup ---
if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
    torch.distributed.init_process_group(backend="nccl")
    rank = torch.distributed.get_rank()
    device = torch.device("cuda", rank)
else:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
print(f"Training on device: {device}")

# --- Load Pseudo Label Strings ---
PSEUDO_LABELS_FILE = "pseudo_labels.txt"
with open(PSEUDO_LABELS_FILE, "r") as f:
    labels = [line.strip() for line in f if line.strip()]
num_labels = len(labels)
print(f"Found {num_labels} pseudo labels.")

# --- Precompute Fixed Text Embeddings ---
_, text_encoder, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None)

text_encoder = text_encoder.to(device)
with torch.no_grad():
    text_tokens = open_clip.tokenize(labels).to(device)
    text_features = text_encoder(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
# Freeze text embeddings for training
text_features = text_features.detach()

# --- Create the Image Encoder (Trainable) ---
image_encoder, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=None)
# Load checkpoint
checkpoint_path = "/workspace/open_clip/src/logs/2025_02_15-22_49_52-model_ViT-B-32-lr_0.0005-b_480-j_4-p_amp/checkpoints/epoch_12.pt"
checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Load checkpoint

# Check if the checkpoint contains a state_dict key (some PyTorch checkpoints do)
if "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

# Load the checkpoint into the model
image_encoder.load_state_dict(checkpoint, strict=False)  # strict=False allows missing keys

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

image_encoder = image_encoder.to(device)

optimizer = optim.Adam(image_encoder.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

# --- Prepare the WebDataset ---
DATASET_TAR = "pseudo_labeled_dataset.tar"
dataset = (
    wds.WebDataset(DATASET_TAR)
    .decode("pil")
    .to_tuple("jpg", "cls")
)

# Build label-to-index mapping
label_to_idx = {label: idx for idx, label in enumerate(labels)}

def collate_fn(batch):
    images, class_strs = zip(*batch)
    targets = [label_to_idx[c] for c in class_strs]
    # Preprocess each PIL image to a tensor
    images = torch.stack([preprocess(img) for img in images])
    targets = torch.tensor(targets, dtype=torch.long)
    return images, targets

dataloader = DataLoader(
    dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4
)

# --- Training Loop with Progress Bars ---
num_epochs = 10
for epoch in range(num_epochs):
    image_encoder.train()
    running_loss = 0.0
    batch_count = 0
    epoch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    for images, targets in epoch_pbar:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        img_features = image_encoder.encode_image(images)
        img_features /= img_features.norm(dim=-1, keepdim=True)
        # Compute logits as similarity between image features & fixed text embeddings
        logits = img_features @ text_features.t()
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        batch_count += 1
        epoch_pbar.set_postfix(loss=f"{loss.item():.4f}")
    avg_loss = running_loss / batch_count
    print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

# Save the trained image encoder
torch.save(image_encoder.state_dict(), "trained_clip_image_encoder.pth")
print("Training complete and model saved.")
