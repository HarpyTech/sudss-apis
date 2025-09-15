from fastapi import FastAPI
from common_utils.helpers import get_timestamp
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'med_gemma_api_{datetime.now().strftime("%Y%m%d")}.log'),
    ],
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Med-Gemma API (Service B)")


@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    response = {
        "message": "Service B (Med-Gemma) is running",
        "timestamp": get_timestamp(),
    }
    logger.debug(f"Returning response: {response}")
    return response


@app.get("/users/{user_name}")
async def read_user(user_name: str):
    logger.info(f"User endpoint accessed with username: {user_name}")
    try:
        response = {"user_name": user_name, "timestamp": get_timestamp()}
        logger.debug(f"Returning response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error processing request for user {user_name}: {str(e)}")
        raise


# Add startup and shutdown event handlers
@app.on_event("startup")
async def startup_event():
    logger.info("Med-Gemma API starting up")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Med-Gemma API shutting down")
