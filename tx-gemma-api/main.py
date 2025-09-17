# tx-gemma-api/main.py
import os, time, torch
from typing import List, Literal, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from common_utils.helpers import get_timestamp
except Exception:
    def get_timestamp(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

from transformers import AutoTokenizer, AutoModelForCausalLM

TX_GEMMA_MODEL = os.getenv("TX_GEMMA_MODEL", "google/gemma-2-2b-it")  # small & CPU-friendly
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
TEMPERATURE    = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P          = float(os.getenv("TOP_P", "0.95"))
TOP_K          = int(os.getenv("TOP_K", "40"))

_tok = None
_lm = None

def load_model():
    global _tok, _lm
    if _tok is not None and _lm is not None:
        return _tok, _lm

    _tok = AutoTokenizer.from_pretrained(TX_GEMMA_MODEL)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    _lm = AutoModelForCausalLM.from_pretrained(
        TX_GEMMA_MODEL,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
    )
    _lm.eval()
    return _tok, _lm

app = FastAPI(title="SUDSS – Tx-Gemma API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Plain prompt text")
    temperature: float = TEMPERATURE
    max_new_tokens: int = MAX_NEW_TOKENS
    top_p: float = TOP_P
    top_k: int = TOP_K

class Message(BaseModel):
    role: Literal["system", "user", "assistant"] = "user"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: float = TEMPERATURE
    max_new_tokens: int = MAX_NEW_TOKENS
    top_p: float = TOP_P
    top_k: int = TOP_K

@app.get("/health")
def health():
    return {"status": "ok", "service": "tx-gemma-api", "model": TX_GEMMA_MODEL, "time": get_timestamp()}

@app.post("/v1/generate")
def generate(req: GenerateRequest):
    try:
        tok, lm = load_model()
        inputs = tok(req.prompt, return_tensors="pt")
        if torch.cuda.is_available(): inputs = {k: v.to(lm.device) for k,v in inputs.items()}
        with torch.no_grad():
            out = lm.generate(
                **inputs,
                do_sample=True,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                max_new_tokens=req.max_new_tokens,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        return {"model": TX_GEMMA_MODEL, "created": int(time.time()), "text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat")
def chat(req: ChatRequest):
    try:
        tok, lm = load_model()
        # Use Gemma chat template
        chat_msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        prompt = tok.apply_chat_template(chat_msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt")
        if torch.cuda.is_available(): inputs = {k: v.to(lm.device) for k,v in inputs.items()}
        with torch.no_grad():
            out = lm.generate(
                **inputs,
                do_sample=True,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                max_new_tokens=req.max_new_tokens,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id
            )
        full = tok.decode(out[0], skip_special_tokens=True)
        # Return only model reply after the last user turn
        reply = full.split(chat_msgs[-1]["content"])[-1].strip()
        return {"model": TX_GEMMA_MODEL, "created": int(time.time()), "text": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
