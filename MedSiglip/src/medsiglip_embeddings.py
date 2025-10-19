import os
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import pandas as pd

class MedSigLipEmbedder:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = torch.device("cpu")  # Use CPU for safety

    def load_model(self):
        model_name = "google/medsiglip-448"  # Replace with the actual model name from Hugging Face
        print(f"📦 Loading MedSigCLIP model: {model_name}")

        # Load processor
        print("🔹 Loading processor...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("✅ Processor loaded!")

        # Load model
        print("🔹 Loading model (this can take a minute)...")
        self.model = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        self.model.to(self.device)
        print("✅ Model loaded successfully!")

    def embed_images_batch(self, image_paths, batch_size=2):
        embeddings = []
        self.model.eval()
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=batch_imgs, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                img_emb = self.model.get_image_features(**inputs)
            embeddings.append(img_emb.cpu())
        return torch.cat(embeddings, dim=0)

    def embed_texts_batch(self, texts, batch_size=4):
        embeddings = []
        self.model.eval()
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                text_emb = self.model.get_text_features(**inputs)
            embeddings.append(text_emb.cpu())
        return torch.cat(embeddings, dim=0)

    def embed_data(self, image_paths, texts):
        if self.model is None or self.processor is None:
            self.load_model()

        image_embs = self.embed_images_batch(image_paths)
        text_embs = self.embed_texts_batch(texts)

        return image_embs, text_embs
    
    def read_data_from_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        image_paths = df['image_path'].tolist()
        texts = df['report_text'].tolist()
        return image_paths, texts