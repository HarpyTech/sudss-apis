from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import numpy as np
import faiss

app = FastAPI()

# Load model and processor
model = None
processor = None

def load_model():
    global model, processor
    model_name = "google/medsiglip-448"  # Replace with the actual model name from Hugging Face
    if not os.path.exists(model_name):
        print(f"Model not found locally. Downloading from Hugging Face: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, dtype=torch.float32)
    model.eval()

class EmbeddingRequest(BaseModel):
    image_paths: list
    texts: list

@app.on_event("startup")
def startup_event():
    load_model()

@app.post("/embed/")
async def embed(request: EmbeddingRequest):
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    image_embs = embed_images_batch(request.image_paths)
    text_embs = embed_texts_batch(request.texts)

    return {
        "image_embeddings": image_embs.tolist(),
        "text_embeddings": text_embs.tolist()
    }

def embed_images_batch(image_paths, batch_size=2):
    embeddings = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to("cpu")
        with torch.no_grad():
            img_emb = model.get_image_features(**inputs)
        embeddings.append(img_emb.cpu())
    return torch.cat(embeddings, dim=0).numpy().astype("float32")

def embed_texts_batch(texts, batch_size=4):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to("cpu")
        with torch.no_grad():
            text_emb = model.get_text_features(**inputs)
        embeddings.append(text_emb.cpu())
    return torch.cat(embeddings, dim=0).numpy().astype("float32")