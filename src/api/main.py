from fastapi import FastAPI
from pydantic import BaseModel
from src.services.churn_service import ChurnService
from src.utils.logger import get_logger

app = FastAPI(
    title="Churn Prediction API",
    version="2.0",
    description="Production-ready churn prediction service"
)

logger = get_logger()
service = ChurnService()


# 🧠 INPUT SCHEMA (IMPORTANT PRO UPGRADE)
class ChurnRequest(BaseModel):
    age: float
    last_interaction: float
    usage_frequency: float
    support_calls: float
    payment_delay: float
    tenure: float


# 🚀 PREDICTION ENDPOINT
@app.post("/classify")
def classify(data: ChurnRequest):

    logger.info("API request received")

    result = service.predict(data.dict())

    return result


# ❤️ HEALTH CHECK (production standard)
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "churn-api",
        "version": "2.0"
    }


# 🏠 ROOT ENDPOINT
@app.get("/")
def home():
    return {
        "message": "Churn API is running",
        "status": "ok",
        "docs": "/docs"
    }