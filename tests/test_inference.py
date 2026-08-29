import pytest
from PIL import Image

from src.inference.predict import DEFAULT_MODEL_PATH, load_model, predict


@pytest.fixture(scope="module")
def model():
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip(f"Trained model artifact not found: {DEFAULT_MODEL_PATH}")
    return load_model()


def test_predict_returns_valid_label_and_probabilities(model):
    image = Image.new("RGB", (300, 300), color=(120, 90, 60))
    result = predict(model, image)

    assert result["label"] in ("Cat", "Dog")

    probabilities = result["probabilities"]
    assert set(probabilities) == {"Cat", "Dog"}
    assert all(0.0 <= p <= 1.0 for p in probabilities.values())
    assert abs(sum(probabilities.values()) - 1.0) < 1e-6
