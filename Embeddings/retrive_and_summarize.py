"""
retrieve_and_summarize.py

Retrieves top-k reports using FAISS + MedSigLIP embeddings and then uses retrieved
reports as context to generate Findings + Impression for a query image using MedGemma (or another LLM).

Usage:
    python retrieve_and_summarize.py \
       --query_image /path/to/q.png \
       --index_path ./embeddings_out/images.index \
       --meta_path ./embeddings_out/metadata.jsonl \
       --medsiglip_model google/medsiglip-448 \
       --gen_model_name <your-gen-model> \
       --device cuda \
       --k 5

Requirements:
    pip install torch torchvision transformers pillow faiss-cpu numpy tqdm

Notes:
 - This script assumes your FAISS index was built using normalized embeddings (IndexFlatIP).
 - Replace placeholders for MedGemma inference with your actual endpoint or model repo as needed.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, AutoModelForCausalLM
import faiss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--query_image", required=True, help="Path to query image")
    p.add_argument("--index_path", default="./embeddings_out/images.index")
    p.add_argument("--meta_path", default="./embeddings_out/metadata.jsonl")
    p.add_argument("--medsiglip_model", default="google/medsiglip-448",
                   help="Model used to produce image embeddings (must match index)")
    p.add_argument("--gen_model_name", default="gpt2", help="Text generation model (MedGemma or other LLM)")
    p.add_argument("--device", default=None, help="cuda or cpu (auto-detect if not supplied)")
    p.add_argument("--k", type=int, default=5, help="Top-k retrieval")
    p.add_argument("--max_context_reports", type=int, default=5, help="How many retrieved reports to include in prompt")
    p.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate")
    return p.parse_args()


def get_device(args_device: str = None) -> torch.device:
    if args_device:
        return torch.device(args_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_metadata_jsonl(path: str) -> List[Dict[str, Any]]:
    meta = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta.append(json.loads(line))
    return meta


class MedSigLIPEmbedder:
    """Small helper to produce an image embedding for a query image using MedSigLIP-like model."""
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        # load the model that produces image features; trust_remote_code may be required for med models
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

    def embed_image(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
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
                    raise RuntimeError("Could not extract image features from model output")
            if not isinstance(feats, torch.Tensor):
                feats = torch.tensor(np.asarray(feats)).to(self.device)
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
            emb = feats.cpu().numpy().astype("float32")  # shape (1, D)
        return emb


def retrieve_topk(index_path: str, q_emb: np.ndarray, k: int):
    idx = faiss.read_index(index_path)
    k_search = min(k, idx.ntotal)
    if k_search == 0:
        return [], []
    scores, ids = idx.search(q_emb, k_search)
    return scores[0].tolist(), ids[0].tolist()


def assemble_prompt_from_reports(reports: List[Dict[str, Any]], query_projection: str = None) -> str:
    """
    Build a single prompt string that concatenates the retrieved reports as context,
    then asks the model to generate Findings and Impression for the query image.
    """
    ctx_lines = []
    ctx_lines.append("You are a radiology assistant. Use the example reports below as reference.")
    ctx_lines.append("")  # blank line
    for i, r in enumerate(reports, start=1):
        header = f"--- Example {i} | uid: {r.get('uid')} | projection: {r.get('projection')} | score: {r.get('_score'):.4f} ---"
        ctx_lines.append(header)
        # keep Findings and Impression short but present
        f = r.get("findings") or ""
        im = r.get("impression") or ""
        # truncate very long fields to avoid overly long context
        max_len = 800
        if len(f) > max_len:
            f = f[:max_len] + " ... [truncated]"
        if len(im) > max_len:
            im = im[:max_len] + " ... [truncated]"
        ctx_lines.append("Findings: " + f)
        ctx_lines.append("Impression: " + im)
        ctx_lines.append("")  # blank line between examples

    ctx_lines.append("Now given the query chest x-ray image (use the same style as the examples),")
    if query_projection:
        ctx_lines.append(f"Projection: {query_projection}")
    ctx_lines.append("Write two sections clearly labeled 'Findings:' and 'Impression:' — be concise and mention uncertainty where appropriate.")
    ctx_lines.append("")  # final blank
    prompt = "\n".join(ctx_lines)
    return prompt


def generate_with_text_model(prompt: str, model_name: str, device: torch.device, max_new_tokens: int = 256) -> str:
    """
    Simple example using an AutoModelForCausalLM-compatible text model.
    Replace this with your MedGemma text-serving code or remote API if needed.
    """
    # NOTE: If using a very large model, you may want to call a remote endpoint instead.
    print(f"Loading generation model {model_name} on {device} (this may take time)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated


def main():
    args = parse_args()
    device = get_device(args.device)
    print("Using device:", device)

    # basic checks
    idx_p = Path(args.index_path)
    meta_p = Path(args.meta_path)
    if not idx_p.exists():
        raise FileNotFoundError("Index not found: " + str(idx_p))
    if not meta_p.exists():
        raise FileNotFoundError("Metadata not found: " + str(meta_p))

    # load metadata
    metadata = load_metadata_jsonl(str(meta_p))
    print(f"Loaded {len(metadata)} metadata records")

    # embed query image
    embder = MedSigLIPEmbedder(args.medsiglip_model, device)
    q_emb = embder.embed_image(args.query_image)  # shape (1, D)

    # retrieve top-k
    scores, ids = retrieve_topk(str(idx_p), q_emb, args.k)
    print("Retrieved ids:", ids)
    # build results list by mapping ids to metadata
    results = []
    for rank, (idx, score) in enumerate(zip(ids, scores), start=1):
        if idx < 0 or idx >= len(metadata):
            continue
        m = metadata[idx].copy()
        m["_score"] = float(score)
        m["_rank"] = rank
        results.append(m)

    # Optionally filter by projection similarity (uncomment if you want)
    query_proj = None
    # If your metadata keeps projection, you can set query_proj from query or pass it in
    # query_proj = "AP"

    # Compose prompt using top-N retrieved reports
    top_n = min(args.max_context_reports, len(results))
    context_reports = results[:top_n]
    prompt = assemble_prompt_from_reports(context_reports, query_projection=query_proj)

    print("\n--- Prompt (truncated 1000 chars) ---\n")
    print(prompt)
    print("...\n--- end prompt preview ---\n")

    # === Option A: Text-only generation (recommended if you only have text LLM endpoint) ===
    # We feed the prompt (retrieved examples + instruction) into a text generation model.
    # generated_text = generate_with_text_model(prompt, args.gen_model_name, device, max_new_tokens=args.max_new_tokens)
    # print("\n=== Generated findings/impression ===\n")
    # print(generated_text)

    # === Option B: Multimodal MedGemma approach (if you have a multimodal model that accepts the image + textual context) ===
    # PSEUDOCODE / placeholder: adapt to your MedGemma API or repo
    #
    # Example (pseudo):
    #   multimodal_model.generate(image=query_image_bytes, context_text=prompt, ...)
    #
    # If your MedGemma repo provides a helper (e.g., `medgemma.generate_with_image_and_context(image, context)`),
    # call that instead of the text-only `generate_with_text_model()` above.
    #
    # Be careful: multimodal servers often require the *image tensor* or base64 and a specific prompt format.

    # Save results JSON
    # out = {
    #     "query_image": str(args.query_image),
    #     "k": len(results),
    #     "retrieved": results,
    #     "prompt": prompt,
    #     "generated": generated_text,
    # }
    # out_path = Path(args.query_image).with_suffix(".retrieval_and_gen.json")
    # with open(out_path, "w", encoding="utf-8") as f:
    #     json.dump(out, f, ensure_ascii=False, indent=2)
    # print("Saved output to:", out_path)


if __name__ == "__main__":
    main()
