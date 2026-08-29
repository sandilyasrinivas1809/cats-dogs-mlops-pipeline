"""FastAPI inference service: liveness check + image classification."""

import io
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.inference.predict import load_model, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

model_holder: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")
    model_holder["model"] = load_model()
    logger.info("Model loaded.")
    yield
    model_holder.clear()


app = FastAPI(title="Cats vs Dogs Classifier", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "model" in model_holder}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)) -> dict:
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc

    return predict(model_holder["model"], image)
