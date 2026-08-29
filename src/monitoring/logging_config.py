"""Request/response logging and Prometheus metrics for the inference API.

Exposes a `LoggingMiddleware` that logs one line per request (endpoint,
status, latency - never the raw image bytes) and records the same data
into Prometheus counters/histograms, plus `metrics_response()` to serve
them on `/metrics`.
"""

import logging
import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s %(message)s")


REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests.",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency in seconds.",
    ["method", "endpoint"],
)
PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total successful predictions, by predicted class.",
    ["label"],
)
MODEL_LOADED = Gauge(
    "model_loaded",
    "1 if the model is loaded and ready to serve, else 0.",
)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log and record metrics for every request.

    Uses the matched route template (e.g. `/predict`) rather than the raw
    path as the `endpoint` label, so that path parameters can't blow up
    Prometheus label cardinality if parameterized routes are added later.
    """

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            latency = time.perf_counter() - start
            endpoint = _endpoint_label(request)
            REQUEST_COUNT.labels(request.method, endpoint, "500").inc()
            REQUEST_LATENCY.labels(request.method, endpoint).observe(latency)
            logger.exception(
                "%s %s -> 500 in %.1fms (unhandled exception)",
                request.method, endpoint, latency * 1000,
            )
            raise

        latency = time.perf_counter() - start
        endpoint = _endpoint_label(request)
        REQUEST_COUNT.labels(request.method, endpoint, str(response.status_code)).inc()
        REQUEST_LATENCY.labels(request.method, endpoint).observe(latency)
        logger.info(
            "%s %s -> %d in %.1fms",
            request.method, endpoint, response.status_code, latency * 1000,
        )
        return response


def _endpoint_label(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", None) or request.url.path


def record_prediction(label: str) -> None:
    PREDICTION_COUNT.labels(label).inc()


def set_model_loaded(loaded: bool) -> None:
    MODEL_LOADED.set(1 if loaded else 0)


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
