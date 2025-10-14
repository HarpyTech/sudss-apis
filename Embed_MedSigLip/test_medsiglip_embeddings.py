import os
import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import pandas as pd


# ==========================================================
# CONFIG
# ==========================================================
model_path = r"C:\PES_MTECH_APR-24-2024\SEM-3\Capstone Project\PROJECT OUTPUT DELIVERABLES TO PES_27-8-2025\INDIANA UNIVERSITY DATASET\Models\Embed_MedSigLip"
device = torch.device("cpu")  # ⚠️ Force CPU (safer for low memory)

# ==========================================================
# SAFE BATCH FUNCTIONS
# ==========================================================
def embed_images_batch(image_paths, model, processor, batch_size=2):
    """Embed multiple images in smaller batches to avoid OOM errors."""
    embeddings = []
    model.eval()
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            img_emb = model.get_image_features(**inputs)
        embeddings.append(img_emb.cpu())
    return torch.cat(embeddings, dim=0)


def embed_texts_batch(texts, model, processor, batch_size=4):
    """Embed multiple texts in smaller batches."""
    embeddings = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_emb = model.get_text_features(**inputs)
        embeddings.append(text_emb.cpu())
    return torch.cat(embeddings, dim=0)


# ==========================================================
# MAIN EXECUTION
# ==========================================================
try:
    print(f"📦 Loading MedSigCLIP model from: {model_path}")

    # Step 1: Load processor
    print("🔹 Step 1: Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path)
    print("✅ Processor loaded!")

    # Step 2: Load model
    print("🔹 Step 2: Loading model (this can take a minute)...")
    model = AutoModel.from_pretrained(model_path, dtype=torch.float32)
    model.to(device)
    print("✅ Model loaded successfully!")

    # ======================================================
# Step 3 & 4: Load metadata CSV, prepare paired batch embeddings
# ======================================================

    csv_path = r"C:\PES_MTECH_APR-24-2024\SEM-3\Capstone Project\PROJECT OUTPUT DELIVERABLES TO PES_27-8-2025\INDIANA UNIVERSITY DATASET\test_meta_data\test_indiana_metadata.csv"

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"🔹 Loading metadata from CSV: {csv_path}")
    df = pd.read_csv(csv_path)

# Drop rows with missing image path or text fields
    df = df.dropna(subset=["image_path", "findings", "impression"])

# Prepare paired image paths and text inputs
    paired_image_paths = []
    paired_texts = []

    for idx, row in df.iterrows():
        image_path = row["image_path"]
        if not os.path.exists(image_path):
            continue  # Skip missing image files
    
    # Concatenate relevant metadata fields
        text = f"{row.get('findings', '')} {row.get('impression', '')}".strip()
    
        paired_image_paths.append(image_path)
        paired_texts.append(text)

    if not paired_image_paths:
        raise ValueError("⚠️ No valid image-text pairs found for embedding.")

    print(f"🔹 Found {len(paired_image_paths)} valid image-text pairs.")

# Step 3: Embed images in batches
    print(f"🔹 Step 3: Embedding {len(paired_image_paths)} images (batch mode)...")
    image_embs = embed_images_batch(paired_image_paths, model, processor, batch_size=2)
    print("✅ Image embeddings generated successfully!")
    print("Image embedding shape:", image_embs.shape)

# Step 4: Embed paired texts in batches
    print(f"🔹 Step 4: Embedding {len(paired_texts)} paired texts (batch mode)...")
    text_embs = embed_texts_batch(paired_texts, model, processor, batch_size=2)
    print("✅ Text embeddings generated successfully!")
    print("Text embedding shape:", text_embs.shape)

# ======================================================
# Step 5: Save FAISS index + aligned metadata CSV
# ======================================================
    import faiss
    import numpy as np
    import pandas as pd

# ⚠️ Ensure image and text embeddings are same length as valid metadata
    assert len(image_embs) == len(text_embs), "Embedding count mismatch"
    assert len(image_embs) == len(paired_image_paths), "Image paths and embeddings mismatch"

# Convert to NumPy (FAISS requires float32)
    image_embs_np = image_embs.numpy().astype("float32")

# Build FAISS index
    faiss_index = faiss.IndexFlatL2(image_embs_np.shape[1])  # L2 distance
    faiss_index.add(image_embs_np)

# Save FAISS index to file
    faiss_index_path = r"C:\PES_MTECH_APR-24-2024\SEM-3\Capstone Project\PROJECT OUTPUT DELIVERABLES TO PES_27-8-2025\INDIANA UNIVERSITY DATASET\faiss_index_saved\faiss_image_embeddings.index"
    faiss.write_index(faiss_index, faiss_index_path)
    print(f"✅ FAISS index saved to: {faiss_index_path}")

# Create aligned metadata (only for valid rows)
    metadata_rows = []

# Re-read metadata from the original file to extract corresponding rows
    full_df = pd.read_csv(csv_path)

# Only include rows whose image paths were valid and used in embeddings
    valid_image_paths_set = set(paired_image_paths)
    for _, row in full_df.iterrows():
        if row["image_path"] in valid_image_paths_set:
            metadata_rows.append({
                "uid": row["uid"],
                "filename": row["filename"],
                "image_path": row["image_path"],
                "projection": row.get("projection", ""),
                "findings": row.get("findings", ""),
                "impression": row.get("impression", ""),
            })

# Ensure the count matches
    assert len(metadata_rows) == len(image_embs), "Metadata and embeddings count mismatch"

# Save metadata to CSV
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_csv_path = "image_metadata.csv"
    metadata_df.to_csv(metadata_csv_path, index_label="faiss_id")
    print(f"✅ Metadata saved to: {metadata_csv_path}")
except Exception as e:
    print(f"❌ Error occurred during processing: {e}")

# ======================================================
# Step 5: Save Separate FAISS indices for Image & Text
# ======================================================
import faiss
import numpy as np
import pandas as pd

# Convert to numpy arrays
image_embs_np = image_embs.numpy().astype("float32")
text_embs_np = text_embs.numpy().astype("float32")

# Build FAISS indices
faiss_index_img = faiss.IndexFlatL2(image_embs_np.shape[1])
faiss_index_txt = faiss.IndexFlatL2(text_embs_np.shape[1])

faiss_index_img.add(image_embs_np)
faiss_index_txt.add(text_embs_np)

# Save indices
faiss_img_path = r"C:\PES_MTECH_APR-24-2024\SEM-3\Capstone Project\PROJECT OUTPUT DELIVERABLES TO PES_27-8-2025\INDIANA UNIVERSITY DATASET\faiss_index_saved\faiss_image_embeddings.index"
faiss_txt_path = r"C:\PES_MTECH_APR-24-2024\SEM-3\Capstone Project\PROJECT OUTPUT DELIVERABLES TO PES_27-8-2025\INDIANA UNIVERSITY DATASET\faiss_index_saved\faiss_text_embeddings.index"

faiss.write_index(faiss_index_img, faiss_img_path)
faiss.write_index(faiss_index_txt, faiss_txt_path)
print(f"✅ Saved FAISS indices for image & text.")

# Save metadata for later retrieval
metadata = pd.DataFrame({
    "uid": df["uid"],
    "filename": df["filename"],
    "image_path": df["image_path"],
    "projection": df["projection"],
    "findings": df["findings"],
    "impression": df["impression"]
})
metadata.to_csv("image_metadata.csv", index_label="faiss_id")
print("✅ Saved metadata file.")

# Save embeddings as .npy for hybrid retrieval
np.save(r"C:\PES_MTECH_APR-24-2024\SEM-3\Capstone Project\PROJECT OUTPUT DELIVERABLES TO PES_27-8-2025\INDIANA UNIVERSITY DATASET\faiss_index_saved\image_embeddings.npy", image_embs.numpy())
np.save(r"C:\PES_MTECH_APR-24-2024\SEM-3\Capstone Project\PROJECT OUTPUT DELIVERABLES TO PES_27-8-2025\INDIANA UNIVERSITY DATASET\faiss_index_saved\text_embeddings.npy", text_embs.numpy())
print("✅ Saved embeddings as .npy")

