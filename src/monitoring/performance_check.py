"""Post-deployment performance check against a running inference API.

Sends a batch of held-out test images (with known labels, from the
`data/processed/test.csv` manifest) to the deployed `/predict` endpoint,
compares predictions against ground truth, and logs accuracy plus
per-request latency.

Runs locally against a locally-running container/API: the raw labeled
images live only on this machine (not in git), so ground truth isn't
available to the ephemeral CD runner.

Usage:
    python -m src.monitoring.performance_check --host http://localhost:8000 --limit 50
"""

import argparse
import logging
import time
from pathlib import Path

import httpx
import pandas as pd

from src.data.preprocess import DEFAULT_PROCESSED_DIR, DEFAULT_RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# httpx logs one INFO line per request, which drowns out the summary.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def run_check(
    host: str,
    raw_dir: Path,
    processed_dir: Path,
    limit: int,
    seed: int,
    timeout: float,
) -> dict:
    df = pd.read_csv(processed_dir / "test.csv")
    if limit < len(df):
        df = df.sample(n=limit, random_state=seed)

    correct, failed, latencies = 0, 0, []

    with httpx.Client(base_url=host, timeout=timeout) as client:
        for row in df.itertuples(index=False):
            image_path = raw_dir / row.filepath
            if not image_path.exists():
                logger.warning("Missing local image, skipping: %s", image_path)
                failed += 1
                continue

            try:
                with open(image_path, "rb") as fh:
                    start = time.perf_counter()
                    response = client.post("/predict", files={"file": (image_path.name, fh, "image/jpeg")})
                    latencies.append(time.perf_counter() - start)
                response.raise_for_status()
            except (httpx.HTTPError, OSError) as exc:
                logger.warning("Request failed for %s: %s", row.filepath, exc)
                failed += 1
                continue

            if response.json()["label"] == row.class_name:
                correct += 1

    scored = len(df) - failed
    accuracy = correct / scored if scored else 0.0
    mean_latency_ms = (sum(latencies) / len(latencies) * 1000) if latencies else 0.0
    p95_latency_ms = (
        sorted(latencies)[int(len(latencies) * 0.95) - 1] * 1000 if len(latencies) >= 20 else None
    )

    logger.info("Sent %d requests (%d scored, %d failed)", len(df), scored, failed)
    logger.info("Accuracy: %.4f (%d/%d correct)", accuracy, correct, scored)
    logger.info("Latency: mean=%.1fms%s", mean_latency_ms,
                f" p95={p95_latency_ms:.1f}ms" if p95_latency_ms else "")

    return {
        "requests": len(df),
        "scored": scored,
        "failed": failed,
        "correct": correct,
        "accuracy": accuracy,
        "mean_latency_ms": mean_latency_ms,
        "p95_latency_ms": p95_latency_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--limit", type=int, default=50, help="Number of test images to send")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    result = run_check(
        args.host, args.raw_dir, args.processed_dir, args.limit, args.seed, args.timeout
    )
    if result["scored"] == 0:
        raise SystemExit("No requests succeeded - is the service running?")


if __name__ == "__main__":
    main()
