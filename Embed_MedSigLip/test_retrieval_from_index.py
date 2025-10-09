import torch
import faiss
import pandas as pd
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

# ==========================================
# CONFIG
# ==========================================
model_path = r"C:\PES_MTECH_APR-24-2024\SEM-3\Capstone Project\PROJECT OUTPUT DELIVERABLES TO PES_27-8-2025\INDIANA UNIVERSITY DATASET\Models\Embed_MedSigLip"
test_image_path = r"C:\PES_MTECH_APR-24-2024\SEM-3\Capstone Project\PROJECT OUTPUT DELIVERABLES TO PES_27-8-2025\INDIANA UNIVERSITY DATASET\test_new_image\10_IM-0002-1001.dcm.png"  # 👈 Replace with your test image
faiss_index_path = r"C:\PES_MTECH_APR-24-2024\SEM-3\Capstone Project\PROJECT OUTPUT DELIVERABLES TO PES_27-8-2025\INDIANA UNIVERSITY DATASET\faiss_index_saved\faiss_image_embeddings.index"
metadata_csv_path = r"C:\PES_MTECH_APR-24-2024\SEM-3\Capstone Project\PROJECT OUTPUT DELIVERABLES TO PES_27-8-2025\INDIANA UNIVERSITY DATASET\test_meta_data\test_indiana_metadata.csv"
device = torch.device("cpu")

# ==========================================
# LOAD MODEL AND PROCESSOR
# ==========================================
print("🔹 Loading model and processor...")
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
model.to(device)
model.eval()
print("✅ Model loaded!")

# ==========================================
# LOAD FAISS INDEX + METADATA
# ==========================================
print("🔹 Loading FAISS index and metadata...")
index = faiss.read_index(faiss_index_path)
metadata_df = pd.read_csv(metadata_csv_path, index_col="faiss_id")
print(f"✅ Loaded index with {index.ntotal} entries")

# ==========================================
# EMBED TEST IMAGE
# ==========================================
print(f"🔹 Embedding test image: {test_image_path}")
img = Image.open(test_image_path).convert("RGB")
inputs = processor(images=img, return_tensors="pt").to(device)

with torch.no_grad():
    query_emb = model.get_image_features(**inputs).cpu().numpy().astype("float32")

print("✅ Image embedding complete.")

# ==========================================
# SEARCH TOP-K SIMILAR CASES
# ==========================================
k = 5
D, I = index.search(query_emb, k)

print(f"\n🔍 Top-{k} similar cases:")
for rank, idx in enumerate(I[0]):
    row = metadata_df.loc[idx]
    print(f"\n--- Result #{rank + 1} ---")
    print(f"📁 Image Path   : {row['image_path']}")
    print(f"🧠 Findings     : {row['findings']}")
    print(f"📝 Impression   : {row['impression']}")
