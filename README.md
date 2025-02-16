# Cerebro

## How it works
TL;DR: We trained our own [CLIP](https://openai.com/index/clip/)-style ViT model on a large dataset of 3M image-text pairs and then 153k specialized wildlife photos (scheme in "Training" below). We use its embeddings for cosine similarity between the user's text input and the wildlife photos. 

An embedding is a high-dimensional vector representation of something (a text or an image); the closer two vectors are, the more semantically similar they are. Eg. if you subtract the embedding of "mother" from the embedding of "father", you get a vector that is similar to the vector of "parents". This means that we can use cosine similarity to determine the similarity between two embeddings.

CLIP has two neural networks: one for images (in this case, a vision transformer) and one for text (a transformer, like GPT).

Given an image, the image encoder produces an embedding representing its visual features. Given a text string, the text encoder generates a corresponding embedding that captures its semantic meaning.

We scrape wildlife images, and for each image, we precompute its embedding and store it (scheme in "Similarity Search" below). When a user inputs a text query, it is converted into an embedding using the text encoder. Then, we calculate the cosine similarity between the text embedding and each image embedding.

The image with the highest cosine similarity (i.e., the smallest angle between the vectors in embedding space) is selected as the best match. This shared embedding space allows CLIP to generalize well to unseen image-text pairs, making it highly effective for zero-shot learning and retrieval tasks like ours.

## Similarity Search Scheme

What is similarity search?

Given a set of vectors $x_i$ in dimension d, Faiss builds a data structure in RAM
from it. After the structure is constructed, when given a new vector x in
dimension d it performs efficiently the operation:

$$j = argmin_i||x-x_i||$$

where $||·||$ is the Euclidean distance ($L^2$).

In Faiss terms, the data structure is an index, an object that has an add method
to add $x_i$ vectors. Note that the $x_i$'s are assumed to be fixed.

Computing the argmin is the search operation on the index.

This is all what Faiss is about. We use it to return the k-th nearest neighbour in a large dataset of 152982 images and tens of thousands of video stills in real-time.

## Training Scheme
We trained on 4x Nvidia RTX4090 with a VIT-B-32 model (~151m params). Everything is fairly meticulously optimized to use data parallelism and split across multiple GPU's. It's capable of scaling across multiple nodes for larger models (but which would take much more than our 24hr time limit to train).

Our training scheme is as follows:
```mermaid
graph TD
    A["Initialize Model <br> (Random Weights, No Pretraining)"] --> B["Train on CC3M Dataset <br> (generalized dataset, 3M Image-Text Pairs)"]
    B --> C["Infer Pseudo Labels <br> on Wildlife Images (from 8 datasets)"]
    C --> D["Fine-Tune Model on <br> 153k Wildlife Images with Pseudo Labels"]
    D -.->|Iterate 3x: Generate New Pseudo Labels & Fine-Tune Again| C
```

For more details on pseudo-labelling, I gave [a talk](https://youtu.be/c8uWUOSGYUI?si=6LILuVIdwS-cxBMJ&t=193) on it 4 years ago.

CC3m was downloaded from [img2dataset](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md) and took ~1 hour. Initial CC3M training took ~6 hours on the 4x RTX4090 GPU's. Then, each pseudo label scheme took ~45min for 152982 images.

We used no externally pretrained models. Our model was a random weights initialization which we trained from scratch on CC3M, then fine tuned on wildlife images.

## How to run the frontend
It's deployed at [sjz.ca](https://sjz.ca) but you can also run it locally on your own data.

1. Install requirements with `uv pip install -r pyproject.toml`
2. Have a directory of images in any format, and run `uv run populate_embeddings.py ../dataset` to generate embeddings and insert them into the FAISS database.
3. Run the API with `uv run uvicorn app:app --reload`
4. In a separate terminal, run the interface with `uv run streamlit run interface.py`