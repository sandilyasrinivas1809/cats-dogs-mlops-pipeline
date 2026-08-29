"""FastAPI inference service: liveness check, image classification, metrics."""

import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.inference.predict import load_model, predict
from src.monitoring.logging_config import (
    LoggingMiddleware,
    configure_logging,
    metrics_response,
    record_prediction,
    set_model_loaded,
)

configure_logging()
logger = logging.getLogger(__name__)

model_holder: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")
    model_holder["model"] = load_model()
    set_model_loaded(True)
    logger.info("Model loaded.")
    yield
    model_holder.clear()
    set_model_loaded(False)


app = FastAPI(title="Cats vs Dogs Classifier", lifespan=lifespan)
app.add_middleware(LoggingMiddleware)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in model_holder}


@app.get("/metrics")
def metrics():
    return metrics_response()


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> dict:
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    result = predict(model_holder["model"], image)
    record_prediction(result["label"])
    return result
