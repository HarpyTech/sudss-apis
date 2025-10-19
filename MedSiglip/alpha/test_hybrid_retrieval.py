import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoProcessor, AutoModel
from PIL import Image
from fetch_ehr import fetch_ehr_for_uid as fetch_ehr_for_patient
from sklearn.preprocessing import normalize
import faiss

# ============================================================
# CONFIG
# ============================================================
model_path = r"C:\PES_MTECH_APR-24-2024\SEM-3\Capstone Project\PROJECT OUTPUT DELIVERABLES TO PES_27-8-2025\INDIANA UNIVERSITY DATASET\Models\Embed_MedSigLip"
metadata_csv = r"C:\PES_MTECH_APR-24-2024\SEM-3\Capstone Project\PROJECT OUTPUT DELIVERABLES TO PES_27-8-2025\INDIANA UNIVERSITY DATASET\test_meta_data\test_indiana_metadata.csv"
ehr_csv = r"C:\PES_MTECH_APR-24-2024\SEM-3\Capstone Project\PROJECT OUTPUT DELIVERABLES TO PES_27-8-2025\INDIANA UNIVERSITY DATASET\EHR_data\EHR_History.csv"
test_image_path = r"C:\PES_MTECH_APR-24-2024\SEM-3\Capstone Project\PROJECT OUTPUT DELIVERABLES TO PES_27-8-2025\INDIANA UNIVERSITY DATASET\test_new_image\14_IM-0256-1001.dcm.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# LOAD MODEL + METADATA
# ============================================================
print("🔹 Loading model and processor...")
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)
print("✅ Model loaded!")

print("🔹 Loading metadata...")
df_meta = pd.read_csv(metadata_csv)
print(f"✅ Loaded {len(df_meta)} metadata entries")

# ============================================================
# EMBED IMAGE FUNCTION
# ============================================================
def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs)
    return emb.cpu().numpy().flatten()

# ============================================================
# HYBRID RETRIEVAL FUNCTION (Cosine + Visual + Semantic + Threshold)
# ============================================================
def compute_hybrid_similarity_faiss(new_img_emb, new_txt_emb, meta_df,
                                    image_embeddings, text_embeddings,
                                    faiss_img_index, faiss_txt_index,
                                    top_k=5, w_cos=0.5, w_sem=0.3, w_vis=0.2, threshold=0.6):

    new_img_emb = normalize(new_img_emb).astype('float32')
    new_txt_emb = normalize(new_txt_emb).astype('float32')

    D_img, I_img = faiss_img_index.search(new_img_emb, top_k * 5)
    vis_sim = 1 - D_img[0]

    D_txt, I_txt = faiss_txt_index.search(new_txt_emb, top_k * 5)
    sem_sim = 1 - D_txt[0]

    cos_sim = np.dot(image_embeddings, new_img_emb.T).squeeze()

    final_score = np.zeros(len(meta_df))
    for i, idx in enumerate(I_img[0]):
        final_score[idx] += w_vis * vis_sim[i]
    for i, idx in enumerate(I_txt[0]):
        final_score[idx] += w_sem * sem_sim[i]
    final_score += w_cos * cos_sim

    mask = final_score > threshold
    top_indices = np.argsort(final_score[mask])[::-1][:top_k]
    top_records = meta_df.iloc[np.where(mask)[0][top_indices]].copy()
    top_records.loc[:, 'similarity'] = final_score[mask][top_indices]

    return top_records

# ============================================================
# MEDGEMMA REPORT GENERATION (SIMULATION)
# ============================================================
def medgemma_generate_report(ehr, top5):
    findings_list = top5['findings'].dropna().tolist()
    impression_list = top5['impression'].dropna().tolist()

    combined_findings = " ".join(findings_list)
    combined_impression = " ".join(impression_list)

    ehr_section = f"\nEHR Notes: {ehr}" if ehr and "No EHR found" not in ehr else ""

    risk_keywords = ['lesion', 'nodule', 'mass', 'consolidation', 'opacity', 'collapse']
    low_risk_keywords = ['normal', 'no abnormality', 'clear', 'unremarkable']

    combined_text = combined_findings.lower() + " " + combined_impression.lower()

    if any(kw in combined_text for kw in risk_keywords):
        overall_status = "Potential abnormality detected – further evaluation advised."
    elif any(kw in combined_text for kw in low_risk_keywords):
        overall_status = "No significant abnormalities – patient appears stable."
    else:
        overall_status = "Inconclusive – additional tests may be required."

    return {
        "Findings": combined_findings.strip(),
        "Impression": combined_impression.strip(),
        "Overall Status": overall_status + ehr_section
    }

# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    print("🔹 Computing image embeddings from image_path column...")
    image_embeddings_list = [embed_image(path) for path in df_meta['image_path']]
    image_embeddings = np.vstack(image_embeddings_list)
    image_embeddings = normalize(image_embeddings).astype('float32')
    print(f"✅ Image embeddings shape: {image_embeddings.shape}")

    print("🔹 Computing text embeddings from findings + impression columns...")
    text_data = (df_meta['findings'].fillna('') + " " + df_meta['impression'].fillna('')).tolist()
    text_embeddings_list = []
    for text in text_data:
        inputs_txt = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            txt_emb = model.get_text_features(**inputs_txt)
        text_embeddings_list.append(txt_emb.cpu().numpy().flatten())
    text_embeddings = np.vstack(text_embeddings_list)
    text_embeddings = normalize(text_embeddings).astype('float32')
    print(f"✅ Text embeddings shape: {text_embeddings.shape}")

    faiss_img_index = faiss.IndexFlatL2(image_embeddings.shape[1])
    faiss_img_index.add(image_embeddings)

    faiss_txt_index = faiss.IndexFlatL2(text_embeddings.shape[1])
    faiss_txt_index.add(text_embeddings)

    print(f"✅ FAISS image index built with {faiss_img_index.ntotal} vectors")
    print(f"✅ FAISS text index built with {faiss_txt_index.ntotal} vectors")

    test_emb = embed_image(test_image_path).reshape(1, -1)

    inputs_txt_test = processor(text="", return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        test_txt_emb = model.get_text_features(**inputs_txt_test).cpu().numpy().reshape(1, -1)
    test_txt_emb = normalize(test_txt_emb).astype('float32')

    uid = os.path.basename(test_image_path).split("_")[0]
    ehr_history = fetch_ehr_for_patient(uid)
    print(f"✅ EHR fetched for patient {uid}")

    top5 = compute_hybrid_similarity_faiss(
        new_img_emb=test_emb,
        new_txt_emb=test_txt_emb,
        meta_df=df_meta,
        image_embeddings=image_embeddings,
        text_embeddings=text_embeddings,
        faiss_img_index=faiss_img_index,
        faiss_txt_index=faiss_txt_index,
        top_k=5,
        w_cos=0.5,
        w_sem=0.3,
        w_vis=0.2,
        threshold=0.6
    )

    print("✅ Retrieved top 5 similar cases:")
    print(top5[["uid", "filename", "similarity"]])

    report = medgemma_generate_report(ehr_history, top5)

    print("\n================ FINAL REPORT ================")
    #print(top5[["uid", "filename", "similarity"]])
    print("Findings:", report["Findings"])
    print()
    print("Impression:", report["Impression"])
    print()
    print("Overall Status:", report["Overall Status"])
    print("==============================================")
