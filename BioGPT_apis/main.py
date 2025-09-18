from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Model name (BioGPT)
MODEL_NAME = "microsoft/BioGPT-Large"   # You can change to "microsoft/BioGPT" (smaller) if RAM is low

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model once at startup
logging.info(f"Loading BioGPT model: {MODEL_NAME} to device {device}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)
logging.info("Model loaded successfully!")

# Create FastAPI app
app = FastAPI(title="BioGPT API", version="1.0.0")

# Request schema
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128

# Helper function for timestamp
def get_timestamp():
    return datetime.utcnow().isoformat()

# Endpoint
@app.post("/generate")
async def generate_text(request: GenerateRequest):
    try:
        logging.info(f"Received request: {request.prompt}")
        
        inputs = tokenizer(request.prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "model": MODEL_NAME,
            "prompt": request.prompt,
            "response": response_text,
            "timestamp": get_timestamp(),
        }

    except Exception as e:
        logging.error(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
