"""Load the trained model and run inference on a single image."""

from pathlib import Path

import torch
from PIL import Image

from src.data.preprocess import CLASSES, preprocess_pil_image
from src.models.model import SimpleCNN

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "model.pt"


def load_model(model_path: Path = DEFAULT_MODEL_PATH, device: str = "cpu") -> SimpleCNN:
    model = SimpleCNN()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict(model: SimpleCNN, image: Image.Image, device: str = "cpu") -> dict:
    """Return `{"label": "Cat"|"Dog", "probabilities": {"Cat": p, "Dog": p}}` for one image."""
    array = preprocess_pil_image(image)
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor)
        prob_dog = torch.sigmoid(logit).item()

    probabilities = {"Cat": 1.0 - prob_dog, "Dog": prob_dog}
    label = CLASSES[1] if prob_dog >= 0.5 else CLASSES[0]
    return {"label": label, "probabilities": probabilities}
