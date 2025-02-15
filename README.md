# Cerebro

## How it works
TL;DR: We trained a CLIP-style ViT model on a large dataset of 3M image-text pairs and then specialized wildlife photos. We use its embeddings for cosine similarity between the user's text input and the wildlife photos.

## Training
We trained on 4x Nvidia RTX4090 with a VIT-B-32 model (~151m params). 

Our training scheme is as follows:
```mermaid
graph TD
    A["Initialize Model <br> (Random Weights, No Pretraining)"] --> B["Train on CC3M Dataset <br> (generalized dataset, 3M Image-Text Pairs)"]
    B --> C["Infer Pseudo Labels <br> on Wildlife Images (from 8 datasets)"]
    C --> D["Fine-Tune Model on <br> Wildlife Images with Pseudo Labels"]
    D -.->|Iterate 3x: Generate New Pseudo Labels & Fine-Tune Again| C
```

For more details on pseudo-labelling, I gave [a talk](https://youtu.be/c8uWUOSGYUI?si=6LILuVIdwS-cxBMJ&t=193) on it 4 years ago.

CC3m was downloaded from [img2dataset](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md) and took ~1 hour. Initial CC3M training took ~N hours on the 4x RTX4090 GPU's. Then, each pseudo label scheme took ~30min.

We used no externally pretrained models. Our model was a random weights initialization which we trained from scratch on CC3M, then fine tuned on wildlife images.