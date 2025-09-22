from fastapi import FastAPI, Query
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, os
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI("BioGPT API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or use ["http://localhost:8000"] for stricter control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = None
model = None

@app.get("/")
def home():
    return {"message": "BioGPT API is running. Use /ask?question=your_query"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ask")
def ask(question: str = Query(..., description="Biomedical question")):
    global tokenizer, model

    if tokenizer is None or model is None:
        try:
            print("🔄 Lazy-loading BioGPT model...")
            MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/BioGPT-Large-PubMedQA")
            HF_TOKEN = os.getenv("HF_TOKEN")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)
            print("✅ Model loaded successfully.")
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return {"error": "Model failed to load. Check logs or HF_TOKEN."}

    inputs = tokenizer(question, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=200,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"question": question, "answer": answer}
