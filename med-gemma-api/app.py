# app/app.py
from __future__ import annotations

import io
import logging
import os
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import torch
from transformers import BitsAndBytesConfig, pipeline

# -----------------------
# Configuration (env)
# -----------------------
MODEL_VARIANT = os.environ.get(
    "MEDGEMMA_VARIANT", "4b-it"
)  # e.g. "4b-it" or "27b-text-it"
MODEL_ID = os.environ.get("MEDGEMMA_ID", f"google/medgemma-{MODEL_VARIANT}")
USE_QUANTIZATION = os.environ.get("USE_QUANTIZATION", "true").lower() in (
    "1",
    "true",
    "yes",
)
DEFAULT_PROMPT = os.environ.get("DEFAULT_PROMPT", "Describe this X-ray")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")


IMAGE_PATH = "./Chest_Xray_PA_3-8-2010.png"
# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("medgemma_api")

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="MedGemma Image-Text API")

# Pipeline singleton holder
PIPELINE = {"pipe": None}


class AnalyzeResponse(BaseModel):
    prompt: str
    generated_text: str
    model_id: str


# -----------------------
# Startup / shutdown
# -----------------------
@app.on_event("startup")
def load_pipeline():
    """
    Load the MedGemma pipeline once on startup.
    """
    try:
        logger.info(
            "Loading MedGemma pipeline (model=%s, quant=%s)...",
            MODEL_ID,
            USE_QUANTIZATION,
        )
        model_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
        if USE_QUANTIZATION:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        # Create pipeline "image-text-to-text" as used in your example
        pipe = pipeline("image-text-to-text", model=MODEL_ID, model_kwargs=model_kwargs)

        # Disable sampling for deterministic outputs (as in your snippet)
        try:
            pipe.model.generation_config.do_sample = False
        except Exception:
            # Some generations configs may not expose this attribute; ignore if so.
            logger.debug("Could not set generation_config.do_sample; continuing.")

        PIPELINE["pipe"] = pipe
        logger.info("MedGemma pipeline loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load MedGemma pipeline: %s", e)
        # Do not crash the server — keep PIPELINE['pipe'] as None and return errors on requests.
        PIPELINE["pipe"] = None


@app.on_event("shutdown")
def cleanup():
    # If needed, free resources. Not strictly necessary in many cases.
    logger.info("Shutting down MedGemma API.")


# -----------------------
# Helper
# -----------------------
def read_image_from_uploadfile(upload_file: UploadFile) -> Image.Image:
    """
    Read a PIL Image from a FastAPI UploadFile without writing to disk.
    """
    try:
        contents = upload_file.file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except Exception as e:
        logger.exception("Failed to read image from UploadFile: %s", e)
        raise


# -----------------------
# API endpoints
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": PIPELINE["pipe"] is not None}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...), prompt: Optional[str] = None):
    """
    Analyze an uploaded image.

    - file: multipart file (image/png, image/jpg, etc.)
    - prompt: optional textual prompt/question. If omitted, a default prompt is used.

    Returns JSON:
    {
      "prompt": "<used prompt>",
      "generated_text": "<model output>",
      "model_id": "<model id>"
    }
    """
    if PIPELINE["pipe"] is None:
        logger.error("Pipeline not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Image file is required"
        )

    used_prompt = (prompt or DEFAULT_PROMPT).strip()
    logger.info(
        "Received analyze request. Prompt: %s, filename: %s", used_prompt, file.filename
    )

    try:
        # Read image from upload
        image = read_image_from_uploadfile(file)

        # Build messages structure similar to your example
        system_instruction = "You are an expert radiologist."
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_instruction}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": used_prompt},
                    {"type": "image", "image": image},
                ],
            },
        ]

        # Call the pipeline
        pipe = PIPELINE["pipe"]

        # pipeline expects text=..., return like in your snippet
        logger.debug("Calling pipeline with messages; model=%s", MODEL_ID)
        # Use CPU/GPU device mapping as handled by pipeline/model_kwargs
        output = pipe(text=messages, max_new_tokens=300)

        # The output structure in your example had nested content; handle common shapes safely
        try:
            # Try to extract similar to your snippet: output[0]["generated_text"][-1]["content"]
            generated_text_field = ""
            if isinstance(output, (list, tuple)) and len(output) > 0:
                first = output[0]
                # case: {'generated_text': [{'content': '...'}]} or nested list
                if isinstance(first, dict) and "generated_text" in first:
                    gen = first["generated_text"]
                    # gen might be a list of blocks or a single block
                    if isinstance(gen, list) and len(gen) > 0:
                        # the snippet used [-1]["content"], be conservative and scan for text content
                        last_block = gen[-1]
                        if isinstance(last_block, dict) and "content" in last_block:
                            generated_text_field = last_block["content"]
                        else:
                            # try string form
                            generated_text_field = str(gen)
                    else:
                        generated_text_field = str(gen)
                else:
                    # fallback: try to stringify first result
                    generated_text_field = str(first)
            else:
                generated_text_field = str(output)
        except Exception as e:
            logger.exception("Failed to parse pipeline output: %s", e)
            generated_text_field = str(output)

        logger.info(
            "Model generation completed (len=%d chars).", len(generated_text_field)
        )

        return JSONResponse(
            status_code=200,
            content={
                "prompt": used_prompt,
                "generated_text": generated_text_field,
                "model_id": MODEL_ID,
            },
        )
    except Exception as e:
        logger.exception("Error during analyze: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
