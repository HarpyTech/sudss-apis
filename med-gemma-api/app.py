# app/app.py
from __future__ import annotations

import io
import logging
import os
from typing import Optional, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import torch
from transformers import BitsAndBytesConfig, pipeline
import httpx

# -----------------------
# Configuration (env)
# -----------------------
MODEL_VARIANT = os.environ.get("MEDGEMMA_VARIANT", "4b-it")
MODEL_ID = os.environ.get("MEDGEMMA_ID", f"google/medgemma-{MODEL_VARIANT}")
USE_QUANTIZATION = os.environ.get("USE_QUANTIZATION", "true").lower() in (
    "1",
    "true",
    "yes",
)
DEFAULT_PROMPT = os.environ.get("DEFAULT_PROMPT", "Describe this X-ray")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
HTTPX_TIMEOUT = float(os.environ.get("HTTPX_TIMEOUT", "10.0"))

# -----------------------
# Predefined test images (update paths as needed)
# -----------------------
# Keys are identifiers clients will pass; values are filesystem paths inside container.
ALLOWED_TEST_IMAGES: Dict[str, str] = {
    # Example: "key": "/app/test_images/filename.png"
    "xray": "./chest_xray.png",
}

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
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


class ImageUrlRequest(BaseModel):
    image_url: str
    prompt: Optional[str] = None


class TestRequest(BaseModel):
    image_path: str  # must be one of ALLOWED_TEST_IMAGES keys
    prompt: Optional[str] = None


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

        pipe = pipeline("image-text-to-text", model=MODEL_ID, model_kwargs=model_kwargs)

        try:
            pipe.model.generation_config.do_sample = False
        except Exception:
            logger.debug("Could not set generation_config.do_sample; continuing.")

        PIPELINE["pipe"] = pipe
        logger.info("MedGemma pipeline loaded successfully.")
    except Exception as e:
        logger.exception("Failed to load MedGemma pipeline: %s", e)
        PIPELINE["pipe"] = None


@app.on_event("shutdown")
def cleanup():
    logger.info("Shutting down MedGemma API.")


# -----------------------
# Helpers
# -----------------------
def read_image_from_uploadfile(upload_file: UploadFile) -> Image.Image:
    try:
        contents = upload_file.file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except Exception as e:
        logger.exception("Failed to read image from UploadFile: %s", e)
        raise


def download_image_from_url(url: str) -> Image.Image:
    try:
        logger.info("Downloading image from URL: %s", url)
        headers = {"User-Agent": "medgemma-api/1.0"}
        with httpx.Client(
            timeout=HTTPX_TIMEOUT, headers=headers, follow_redirects=True
        ) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                logger.error("Failed to download image. Status: %s", resp.status_code)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Image download failed: {resp.status_code}",
                )
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return image
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error while downloading image from URL: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download/parse image: {e}",
        )


def load_test_image_by_key(key: str) -> Image.Image:
    """
    Load a server-side predefined test image by key.
    """
    if key not in ALLOWED_TEST_IMAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown test image key: {key}",
        )
    path = ALLOWED_TEST_IMAGES[key]
    if not os.path.exists(path):
        logger.error("Test image path not found: %s", path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Test image not found on server: {path}",
        )
    try:
        image = Image.open(path).convert("RGB")
        return image
    except Exception as e:
        logger.exception("Failed to load test image %s: %s", path, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load test image: {e}",
        )


def parse_pipeline_output(output: object) -> str:
    try:
        generated_text_field = ""
        if isinstance(output, (list, tuple)) and len(output) > 0:
            first = output[0]
            if isinstance(first, dict) and "generated_text" in first:
                gen = first["generated_text"]
                if isinstance(gen, list) and len(gen) > 0:
                    last_block = gen[-1]
                    if isinstance(last_block, dict) and "content" in last_block:
                        generated_text_field = last_block["content"]
                    else:
                        generated_text_field = str(gen)
                else:
                    generated_text_field = str(gen)
            else:
                generated_text_field = str(first)
        else:
            generated_text_field = str(output)
        return generated_text_field
    except Exception as e:
        logger.exception("Failed to parse pipeline output: %s", e)
        return str(output)


# -----------------------
# API endpoints
# -----------------------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": PIPELINE["pipe"] is not None}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(file: UploadFile = File(...), prompt: Optional[str] = None):
    if PIPELINE["pipe"] is None:
        logger.error("Pipeline not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    used_prompt = (prompt or DEFAULT_PROMPT).strip()
    logger.info(
        "Received analyze request. Prompt: %s, filename: %s", used_prompt, file.filename
    )

    try:
        image = read_image_from_uploadfile(file)

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

        pipe = PIPELINE["pipe"]
        logger.debug("Calling pipeline with messages; model=%s", MODEL_ID)
        output = pipe(text=messages, max_new_tokens=300)

        generated_text_field = parse_pipeline_output(output)
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
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during analyze: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post("/analyze_url", response_model=AnalyzeResponse)
async def analyze_url(request: ImageUrlRequest):
    if PIPELINE["pipe"] is None:
        logger.error("Pipeline not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    image_url = request.image_url
    used_prompt = (request.prompt or DEFAULT_PROMPT).strip()
    logger.info(
        "Received analyze_url request. URL: %s Prompt: %s", image_url, used_prompt
    )

    try:
        image = download_image_from_url(image_url)

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

        pipe = PIPELINE["pipe"]
        logger.debug("Calling pipeline with messages (URL); model=%s", MODEL_ID)
        output = pipe(text=messages, max_new_tokens=300)

        generated_text_field = parse_pipeline_output(output)
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
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during analyze_url: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post("/test", response_model=AnalyzeResponse)
async def test_endpoint(request: TestRequest):
    """
    Test endpoint that uses server-side predefined images.
    Request JSON:
      {"image_path": "<key from ALLOWED_TEST_IMAGES>", "prompt": "<optional prompt>"}
    """
    if PIPELINE["pipe"] is None:
        logger.error("Pipeline not loaded.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    key = request.image_path
    used_prompt = (request.prompt or DEFAULT_PROMPT).strip()
    logger.info("Received test request. test_key=%s prompt=%s", key, used_prompt)

    try:
        image = load_test_image_by_key(key)

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

        pipe = PIPELINE["pipe"]
        logger.debug(
            "Calling pipeline with messages (test); model=%s test_key=%s", MODEL_ID, key
        )
        output = pipe(text=messages, max_new_tokens=300)

        generated_text_field = parse_pipeline_output(output)
        logger.info(
            "Test generation completed (len=%d chars).", len(generated_text_field)
        )

        return JSONResponse(
            status_code=200,
            content={
                "prompt": used_prompt,
                "generated_text": generated_text_field,
                "model_id": MODEL_ID,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during test endpoint: %s", e)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
