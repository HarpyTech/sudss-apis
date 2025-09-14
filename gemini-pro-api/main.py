from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    logging.error("GOOGLE_API_KEY not found in environment variables")
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI(title="Gemini API's", version="1.0.0")
logging.info("FastAPI app initialized")


class GenerateRequest(BaseModel):
    prompt: str
    model: str
    media_urls: Optional[List[str]] = None  # Optional list of image URLs


SUPPORTED_MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]


def get_timestamp():
    return datetime.utcnow().isoformat()


@app.post("/generate")
async def generate_text(request: GenerateRequest):
    logging.info(
        f"Received generate request: model={request.model},"
        f"prompt={request.prompt}, media_urls={request.media_urls}"
    )

    if request.model not in SUPPORTED_MODELS:
        logging.warning(f"Unsupported model requested: {request.model}")
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{request.model}' not supported. "
                f"Use one of: {SUPPORTED_MODELS}"
            ),
        )

    try:
        logging.info(f"Using model: {request.model}")
        model = genai.GenerativeModel(request.model)
        if request.media_urls:
            logging.info(f"Media URLs provided: {request.media_urls}")
            contents = [{"role": "user", "parts": [{"text": request.prompt}]}]
            for url in request.media_urls:
                contents[0]["parts"].append({"image_url": url})
            response = model.generate_content(contents)
        else:
            logging.info("No media URLs provided, generating text only")
            response = model.generate_content(request.prompt)

        if hasattr(response, "text"):
            response_text = response.text
        else:
            response_text = str(response)

        logging.info("Response generated successfully")
        return {
            "model": request.model,
            "prompt": request.prompt,
            "response": response_text,
            "timestamp": get_timestamp(),
        }

    except Exception as e:
        logging.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
