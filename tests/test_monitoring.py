import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.inference.app import app
from src.inference.predict import DEFAULT_MODEL_PATH


@pytest.fixture(scope="module")
def client():
    if not DEFAULT_MODEL_PATH.exists():
        pytest.skip(f"Trained model artifact not found: {DEFAULT_MODEL_PATH}")
    with TestClient(app) as test_client:
        yield test_client


def sample_image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), color=(120, 90, 60)).save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


def test_metrics_endpoint_exposes_prometheus_format(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    body = response.text
    assert "http_requests_total" in body
    assert "http_request_latency_seconds" in body
    assert "model_loaded" in body


def test_metrics_track_requests_and_predictions(client):
    before = client.get("/metrics").text

    client.get("/health")
    client.post("/predict", files={"file": ("sample.jpg", sample_image_bytes(), "image/jpeg")})

    after = client.get("/metrics").text

    # The /health counter should have advanced, and a prediction recorded.
    assert 'endpoint="/health"' in after
    assert "predictions_total" in after
    assert after != before


def test_invalid_image_still_records_a_4xx_metric(client):
    client.post("/predict", files={"file": ("bad.jpg", io.BytesIO(b"not an image"), "image/jpeg")})

    body = client.get("/metrics").text
    assert 'status="400"' in body
