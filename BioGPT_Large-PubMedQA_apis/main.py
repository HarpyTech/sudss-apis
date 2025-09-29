from fastapi import FastAPI, Query
from transformers import pipeline
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise EnvironmentError("HF_TOKEN not found. Please set it in your .env file or environment.")

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load BioGPT model ONCE at startup using pipeline
app = FastAPI(title="BioGPT API with Auto-Tuned Prompts")

qa_pipeline = pipeline(
    "text-generation",
    model="microsoft/BioGPT-Large-PubMedQA",
    token=HF_TOKEN
)

# --- Prompt Tuning Logic ---
def classify_question_type(question: str) -> str:
    q = question.lower()
    if any(kw in q for kw in ["define", "what is", "meaning", "describe"]):
        return "definition"
    elif any(kw in q for kw in ["treatment", "therapy", "medication", "manage"]):
        return "treatment"
    elif any(kw in q for kw in ["cause", "mechanism", "pathophysiology", "why"]):
        return "mechanism"
    else:
        return "general"

def generate_prompt(question: str) -> str:
    q_type = classify_question_type(question)
    if q_type == "definition":
        return f"Provide a clear biomedical definition:\nQuestion: {question}"
    elif q_type == "treatment":
        return f"Summarize the current medical treatments:\nQuestion: {question}"
    elif q_type == "mechanism":
        return f"Explain the biomedical mechanisms involved:\nQuestion: {question}"
    else:
        return f"Answer the following biomedical question clearly:\nQuestion: {question}"

# --- API Endpoints ---
@app.get("/")
def root():
    return {"message": "BioGPT API with automated prompt tuning is ready. Use /ask?question=your_query"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ask")
def ask(question: str = Query(..., description="Biomedical question")):
    """
    Example:
    GET /ask?question=What is the treatment for diabetes?
    """
    prompt = generate_prompt(question)
    logging.info(f"Classified prompt: {prompt}")
    result = qa_pipeline(prompt, max_length=400, do_sample=True, top_k=50, top_p=0.95, temperature=0.7)
    answer = result[0]["generated_text"]
    logging.info(f"Answer: {answer}")
    return {"question": question, "prompt": prompt, "answer": answer}
