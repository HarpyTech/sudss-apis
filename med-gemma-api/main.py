from fastapi import FastAPI
from common_utils.helpers import get_timestamp

app = FastAPI(title="Med-Gemma API (Service B)")

@app.get("/")
async def root():
    return {"message": "Service B (Med-Gemma) is running", "timestamp": get_timestamp()}

@app.get("/users/{user_name}")
async def read_user(user_name: str):
    return {"user_name": user_name, "timestamp": get_timestamp()}
