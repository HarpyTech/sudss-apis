#!/usr/bin/env python3
"""
medgemma_local.py
Run an image-text-to-text pipeline for MedGemma locally with robust device/quantization handling.
"""

import os
import sys
import argparse
import requests
from io import BytesIO
from PIL import Image
import torch
from huggingface_hub import login
from transformers import pipeline, BitsAndBytesConfig, logging

# reduce transformer logs unless you want them
logging.set_verbosity_error()


def download_image(url: str) -> Image.Image:
    resp = requests.get(
        "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
    )
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def load_image(path_or_url: str, is_url: bool):
    if is_url:
        return download_image(path_or_url)
    else:
        return Image.open(path_or_url).convert("RGB")


def main(args):
    # --- 1) Get HF token from environment ---
    hf_token = os.environ.get("HF_TOKEN")
    if args.hf_token:
        hf_token = args.hf_token
    if not hf_token:
        print(
            "ERROR: Set HF_TOKEN environment variable with your Hugging Face token.",
            file=sys.stderr,
        )
        sys.exit(1)

    login(hf_token)
    # --- 2) Prepare model / device config ---
    model_variant = args.model_variant or "4b-it"
    model_id = f"google/medgemma-{model_variant}"

    # device selection
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = 0  # pipeline accepts device int for GPU
        device_map = "auto"
        print("CUDA available -> will attempt GPU (device_map='auto').")
    else:
        device = "cpu"
        device_map = "cpu"
        print("No CUDA -> using CPU.")

    # quantization config (only valid with bitsandbytes + GPU)
    quant_config = None
    if args.use_quant:
        if not use_cuda:
            print(
                "WARNING: 4-bit quantization requested but no CUDA available. Ignoring quantization."
            )
        else:
            quant_config = BitsAndBytesConfig(load_in_4bit=True)
            print("4-bit quantization enabled (bitsandbytes).")

    # model kwargs
    model_kwargs = dict(
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        device_map=device_map,
    )
    if quant_config:
        model_kwargs["quantization_config"] = quant_config

    # --- 3) Load image ---
    if args.image_file:
        image = load_image(args.image_file, is_url=False)
    else:
        image = load_image(
            "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png",
            is_url=True,
        )

    # --- 4) Create pipeline ---
    # Keep the same task name that worked in your original environment; adjust if necessary
    task_name = "image-text-to-text"  # if this errors, try "image-to-text" or "vision-encoder-decoder"
    try:
        pipe = pipeline(
            task_name,
            model=model_id,
            model_kwargs=model_kwargs,
            # use_auth_token=hf_token,
        )
    except Exception as e:
        # try fallback without model_kwargs (some local setups require simpler load)
        print(
            "Model load failed with model_kwargs — retrying with simpler config...",
            file=sys.stderr,
        )
        try:
            pipe = pipeline(task_name, model=model_id)
        except Exception as e2:
            print("Failed to load pipeline. Exception follows:", file=sys.stderr)
            raise e2 from e

    # optional: disable sampling for deterministic behaviour
    try:
        pipe.model.generation_config.do_sample = False
    except Exception:
        pass

    # --- 5) Build messages (same structure you used) ---
    role_instruction = (
        args.role_instruction
        or "You are an expert radiologist. Explain how the summarization was defined with respect to the given X-ray."
    )
    user_prompt = input("Enter your prompt (or leave blank for default): ").strip()

    prompt = (
        user_prompt
        or "Describe this X-ray, And Provide the summarization by explaining how the Findings has been captured"
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": role_instruction}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image},
            ],
        },
    ]

    # --- 6) Run pipeline ---
    print("\nRunning model inference... (this may take a while depending on device)\n")
    max_new_tokens = args.max_new_tokens or 300
    output = pipe(text=messages, max_new_tokens=max_new_tokens)

    # The return structure may vary by model/pipeline version.
    # We try to safely extract generated text:
    generated_text = None
    try:
        # older style: output is list with dicts that have "generated_text"
        if (
            isinstance(output, list)
            and len(output) > 0
            and "generated_text" in output[0]
        ):
            generated_text = output[0]["generated_text"]
        # another style (your original code): nested structure
        elif (
            isinstance(output, list)
            and len(output) > 0
            and isinstance(output[0].get("generated_text"), list)
        ):
            # attempt to extract last text segment
            generated_text = output[0]["generated_text"][-1].get("content")
        else:
            # fallback: convert whole output to string
            generated_text = str(output)
    except Exception:
        generated_text = str(output)

    # --- 7) Present results ---
    print("=== PROMPT ===")
    print(prompt)
    print("\n=== MODEL OUTPUT ===")
    print(generated_text)
    print("\n=== RAW PIPE OUTPUT (for debugging) ===")
    print(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image-url", type=str, help="URL of image to download and feed to model."
    )
    group.add_argument(
        "--image-file", type=str, help="Local path to image file to feed to model."
    )
    p.add_argument(
        "--model-variant",
        type=str,
        default="4b-it",
        help="MedGemma variant (default: 4b-it)",
    )
    p.add_argument(
        "--use-quant",
        action="store_true",
        default=True,
        help="Try to use 4-bit quantization (requires CUDA + bitsandbytes).",
    )
    p.add_argument(
        "--hf-token", type=str, help="Hugging Face token (or set HF_TOKEN env var)."
    )
    p.add_argument("--max-new-tokens", type=int, default=300)
    p.add_argument("--prompt", type=str, help="Text prompt to send with image.")
    p.add_argument("--role-instruction", type=str, help="System role instruction.")
    args = p.parse_args()
    main(args)
