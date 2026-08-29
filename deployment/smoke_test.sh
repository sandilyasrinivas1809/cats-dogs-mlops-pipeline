#!/usr/bin/env bash
# Post-deploy smoke test: health check, then a real prediction call.
# Usage: smoke_test.sh [host] [sample_image]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${1:-http://localhost:8000}"
SAMPLE_IMAGE="${2:-$SCRIPT_DIR/sample.jpg}"

echo "Waiting for $HOST/health..."
for i in $(seq 1 30); do
  if curl -sf "$HOST/health" > /dev/null; then
    echo "Service is healthy."
    break
  fi
  if [ "$i" -eq 30 ]; then
    echo "Service did not become healthy in time." >&2
    exit 1
  fi
  sleep 2
done

echo "Checking POST /predict with $SAMPLE_IMAGE..."
curl -sf -X POST "$HOST/predict" -F "file=@${SAMPLE_IMAGE}" > /dev/null

echo "Smoke tests passed."
