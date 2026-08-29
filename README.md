# cats-dogs-mlops-pipeline

End-to-end MLOps pipeline for Cats vs Dogs image classification — MLflow experiment tracking, FastAPI inference service, Docker containerization, GitHub Actions CI/CD, and Docker Compose deployment with monitoring.

BITS MLOps (S1-25_AIMLCZG523) — Assignment 2.

## Stack

Git + Git LFS · PyTorch · MLflow · FastAPI · Docker · GitHub Actions · Docker Hub · Docker Compose · Prometheus client

## Layout

```
src/
├── data/preprocess.py       # validate images, stratified 80/10/10 split, manifest generation
├── data/dataset.py          # PyTorch Dataset/DataLoader over the manifests
├── models/model.py          # SimpleCNN (Conv/BatchNorm/Pool -> dense -> logit)
├── models/train.py          # training loop + MLflow tracking
├── inference/predict.py     # load model, run inference
├── inference/app.py         # FastAPI: /health, /predict, /metrics
└── monitoring/
    ├── logging_config.py    # request logging middleware + Prometheus metrics
    └── performance_check.py # post-deployment accuracy/latency check
tests/                       # pytest suite (11 tests)
deployment/                  # docker-compose.yml, smoke_test.sh, sample.jpg
.github/workflows/           # ci.yml, cd.yml
```

## Data

Raw dataset (`dog-and-cat-classification-dataset/versions/1/PetImages/{Cat,Dog}`, ~864MB / 25k images) is
downloaded via `kagglehub` and is **not** version-controlled — it would exhaust GitHub's free Git LFS quota.
Only the lightweight split manifests (`data/processed/*.csv`, ~4MB) and the trained model
(`models/model.pt`) are LFS-tracked. Resize/normalize/augmentation happen on-the-fly at load time.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Build the split manifests** (requires the raw dataset locally):

```bash
python -m src.data.preprocess
```

**Train** (CPU-only; the full dataset takes ~25-30 min/epoch, so use the limit flags for a quick run):

```bash
# quick run on a subsample
python -m src.models.train --train-limit 4000 --val-limit 500 --test-limit 500 --epochs 5

# full dataset (~2-2.5 hrs for 5 epochs on CPU)
python -m src.models.train --epochs 5 --batch-size 32
```

**View experiments in MLflow:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

**Run the API locally:**

```bash
python -m uvicorn src.inference.app:app --host 127.0.0.1 --port 8000
# or containerized:
docker build -t cats-dogs-classifier:latest .
docker run -d --name cats-dogs-api -p 8000:8000 cats-dogs-classifier:latest
```

**Call it:**

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics
curl -X POST http://localhost:8000/predict \
  -F "file=@dog-and-cat-classification-dataset/versions/1/PetImages/Cat/1941.jpg"
```

Interactive docs: <http://localhost:8000/docs>

**Run the tests:**

```bash
python -m pytest -v
```

**Deploy via Docker Compose + smoke test:**

```bash
export DOCKERHUB_USERNAME=<your-dockerhub-username>
docker compose -f deployment/docker-compose.yml up -d
deployment/smoke_test.sh http://localhost:8000
```

**Post-deployment performance check** (sends held-out labeled test images at a running API and reports
accuracy + latency; run locally, since the ground-truth images aren't available to CI):

```bash
python -m src.monitoring.performance_check --host http://localhost:8000 --limit 50
```

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness/readiness — reports whether the model is loaded |
| `/predict` | POST | Multipart image upload → predicted label + class probabilities |
| `/metrics` | GET | Prometheus metrics (request counts, latency histograms, predictions by class, model-loaded gauge) |

## CI/CD

- **`ci.yml`** (push/PR to `main`): checkout with LFS → install deps → `pytest` → `docker build`; on push to
  `main`, also pushes the image to Docker Hub tagged `latest` and by commit SHA.
  Requires `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` repo secrets.
- **`cd.yml`** (on CI success on `main`): `compose pull` → `up -d` → `smoke_test.sh` → `down`.
  Deploys on the ephemeral GitHub-hosted runner, so no persistent infrastructure (or self-hosted runner
  with access to a personal machine) is required.

## Model notes

Baseline `SimpleCNN`: four Conv/BatchNorm/ReLU/MaxPool blocks → dense(128) → dropout → single logit,
trained with `BCEWithLogitsLoss`. BatchNorm and logit output are both load-bearing: without them, on
`[0,1]`-normalized inputs the sigmoid+`BCELoss` combination produces no usable gradient and training
stalls at `ln(2)` loss indefinitely.

The committed checkpoint was trained on a 4000-image subsample for 5 epochs (CPU-only), reaching ~59%
test accuracy with high recall / low precision (it over-predicts "Dog"). It demonstrates the pipeline
rather than a tuned model — rerun the full-dataset training command above for better numbers.
