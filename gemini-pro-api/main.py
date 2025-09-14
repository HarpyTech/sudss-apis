from fastapi import FastAPI
from common_utils.helpers import get_timestamp

app = FastAPI(title="Gemini Pro API (Service A)")

@app.get("/")
async def root():
    return {"message": "Service A (Gemini-pro) is running", "timestamp": get_timestamp()}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id, "timestamp": get_timestamp()}
