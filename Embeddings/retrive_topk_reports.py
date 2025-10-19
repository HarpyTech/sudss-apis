"""
retrieve_topk_reports.py

Usage:
    python retrieve_topk_reports.py --image_path /path/to/query.png \
        --index_path ./embeddings_out/images.index \
        --meta_path ./embeddings_out/metadata.jsonl \
        --model_name google/medsiglip-448 \
        --k 5

Requirements:
    pip install torch torchvision transformers pillow faiss-cpu numpy tqdm

Notes:
 - This script assumes the FAISS index was built with L2-normalized vectors (IndexFlatIP).
 - The model should be the same one used to build the index (e.g., google/medsiglip-448).
 - Metadata JSONL must have one JSON object per vector in the same order as embeddings
   (uid, image_path, findings, impression, etc).
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
import faiss
from tqdm.auto import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_path", required=True, help="Path to the query image")
    p.add_argument("--index_path", default="./embeddings_out/images.index", help="Path to FAISS index file")
    p.add_argument("--meta_path", default="./embeddings_out/metadata.jsonl", help="Path to metadata jsonl")
    p.add_argument("--model_name", default="google/medsiglip-448", help="HF model used for embeddings")
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detect if not provided)")
    p.add_argument("--k", type=int, default=5, help="Top-k results to return")
    return p.parse_args()


def get_device(args_device: str = None) -> torch.device:
    if args_device:
        return torch.device(args_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QueryEmbedder:
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        # load processor + model (image encoder)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

    def embed_image(self, image_path: str) -> np.ndarray:
        # load image
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            # try model helper then fallbacks
            try:
                feats = self.model.get_image_features(**inputs)
            except Exception:
                out = self.model(**inputs)
                if hasattr(out, "image_embeds"):
                    feats = out.image_embeds
                elif hasattr(out, "pooler_output"):
                    feats = out.pooler_output
                elif hasattr(out, "last_hidden_state"):
                    feats = out.last_hidden_state[:, 0, :]
                else:
                    raise RuntimeError("Could not find image embedding in model output")
            # ensure tensor
            if not isinstance(feats, torch.Tensor):
                feats = torch.tensor(np.asarray(feats)).to(self.device)
            # normalize (L2)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            emb = feats.cpu().numpy().astype("float32")
        return emb  # shape (1, D)


def load_metadata_jsonl(path: str) -> List[Dict[str, Any]]:
    meta = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta.append(json.loads(line))
    return meta


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    # check files
    idx_path = Path(args.index_path)
    meta_path = Path(args.meta_path)
    if not idx_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {idx_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata JSONL not found: {meta_path}")

    # load FAISS index
    print("Loading FAISS index...", idx_path)
    index = faiss.read_index(str(idx_path))

    # load metadata
    print("Loading metadata...", meta_path)
    metadata = load_metadata_jsonl(str(meta_path))
    n_meta = len(metadata)
    print(f"Loaded {n_meta} metadata records")

    # embed query image
    print("Loading model and embedding query image...")
    embedder = QueryEmbedder(args.model_name, device)
    q_emb = embedder.embed_image(args.image_path)  # shape (1, D)

    # normalize again to be safe (faiss IndexFlatIP expects normalized vectors if you want cosine)
    # faiss.normalize_L2(q_emb)  # q_emb already normalized by embedder

    k = args.k
    # handle k > n
    k_search = min(k, index.ntotal)
    if k_search == 0:
        print("Index has no vectors.")
        return

    # search
    print(f"Searching top {k_search}...")
    scores, ids = index.search(q_emb, k_search)  # scores shape (1, k), ids shape (1, k)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    results = []
    for rank, (idx, score) in enumerate(zip(ids, scores), start=1):
        if idx < 0:
            continue
        if idx >= len(metadata):
            # metadata length mismatch — show a warning and skip
            print(f"Warning: idx {idx} >= metadata length {len(metadata)}. Skipping.")
            continue
        m = metadata[idx].copy()
        m["_score"] = float(score)
        m["_rank"] = rank
        results.append(m)

    # print nicely
    print("\nTop results:")
    for r in results:
        print(f"Rank {r.get('_rank')} | uid: {r.get('uid')} | score: {r.get('_score'):.4f}")
        print(f"  image_path: {r.get('image_path')}")
        print(f"  findings: {r.get('findings')}")
        print(f"  impression: {r.get('impression')}")
        print("-" * 60)

    # Optionally return JSON
    out = {"query_image": args.image_path, "k": k_search, "results": results}
    # write json output file next to query (optional)
    out_path = Path(args.image_path).with_suffix(".retrieval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved retrieval JSON to: {out_path}")


if __name__ == "__main__":
    main()
