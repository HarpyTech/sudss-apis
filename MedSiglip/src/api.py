from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import numpy as np
import faiss

app = FastAPI()

class EmbeddingRequest(BaseModel):
    image_paths: list[str]
    texts: list[str]

model = None
processor = None

def load_model():
    global model, processor
    model_name = "google/medsiglip-448"  # Replace with the actual model name from Hugging Face
    if not os.path.exists(model_name):
        print(f"Downloading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, dtype=torch.float32)
    model.eval()

@app.on_event("startup")
def startup_event():
    load_model()

@app.post("/embed")
def embed(request: EmbeddingRequest):
    if not model or not processor:
        raise HTTPException(status_code=500, detail="Model not loaded")

    image_embs = []
    text_embs = []

    for image_path in request.image_paths:
        if os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt", padding=True)
            with torch.no_grad():
                img_emb = model.get_image_features(**inputs)
            image_embs.append(img_emb.cpu().numpy())
        else:
            raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")

    for text in request.texts:
        inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_emb = model.get_text_features(**inputs)
        text_embs.append(text_emb.cpu().numpy())

    return {
        "image_embeddings": np.array(image_embs).tolist(),
        "text_embeddings": np.array(text_embs).tolist()
    }