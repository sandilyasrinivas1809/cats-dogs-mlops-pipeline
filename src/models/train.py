"""Train the baseline CNN with MLflow experiment tracking.

Logs hyperparameters, per-epoch metrics (loss/accuracy/precision/recall),
and artifacts (confusion matrix, loss curve, serialized model) to MLflow.
"""

import argparse
import logging
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from torch.utils.data import Subset

from src.data.dataset import get_dataloader
from src.data.preprocess import DEFAULT_PROCESSED_DIR, DEFAULT_RAW_DIR
from src.models.model import SimpleCNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    """Run one pass over `loader`. Trains if `optimizer` is given, else evaluates."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs.detach())
            all_preds.extend((probs.cpu().numpy() >= 0.5).astype(int).tolist())
            all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    return avg_loss, accuracy, precision, recall, all_labels, all_preds


def plot_loss_curve(train_losses: list[float], val_losses: list[float], out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(labels: list[int], preds: list[int], out_path: Path) -> None:
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(4, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Cat", "Dog"], yticklabels=["Cat", "Dog"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (test set)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def subsample(loader_dataset, limit: int | None):
    if limit is None or limit >= len(loader_dataset):
        return loader_dataset
    return Subset(loader_dataset, range(limit))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=None, help="Cap train split size (debug/quick runs)")
    parser.add_argument("--val-limit", type=int, default=None, help="Cap val split size (debug/quick runs)")
    parser.add_argument("--test-limit", type=int, default=None, help="Cap test split size (debug/quick runs)")
    parser.add_argument("--experiment-name", type=str, default="cats-dogs-baseline-cnn")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    split_limits = {"train": args.train_limit, "val": args.val_limit, "test": args.test_limit}
    loaders = {}
    for split in ("train", "val", "test"):
        loader = get_dataloader(
            split, raw_dir=args.raw_dir, processed_dir=args.processed_dir,
            batch_size=args.batch_size, num_workers=args.num_workers,
        )
        limit = split_limits[split]
        if limit is not None:
            dataset = subsample(loader.dataset, limit)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size,
                shuffle=(split == "train"), num_workers=args.num_workers,
            )
        loaders[split] = loader
        logger.info("%s: %d samples", split, len(loader.dataset))

    model = SimpleCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # MLflow 3.x's plain filesystem backend is in maintenance mode and refuses to
    # start; use an explicit local sqlite store instead. Both the tracking URI and
    # the experiment's artifact location are built as absolute paths ourselves,
    # since mlflow's own default-URI resolution mis-handles spaces in the cwd
    # (e.g. "Assignment 2") and would otherwise write outside the repo.
    mlflow.set_tracking_uri(f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(args.experiment_name) is None:
        mlflow.create_experiment(args.experiment_name, artifact_location=(REPO_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "img_size": 224,
                "model": "SimpleCNN",
                "train_samples": len(loaders["train"].dataset),
                "val_samples": len(loaders["val"].dataset),
                "test_samples": len(loaders["test"].dataset),
            }
        )

        train_losses, val_losses = [], []
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc, train_prec, train_rec, _, _ = run_epoch(
                model, loaders["train"], criterion, optimizer, device
            )
            val_loss, val_acc, val_prec, val_rec, _, _ = run_epoch(
                model, loaders["val"], criterion, None, device
            )
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "train_precision": train_prec,
                    "train_recall": train_rec,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_precision": val_prec,
                    "val_recall": val_rec,
                },
                step=epoch,
            )
            logger.info(
                "Epoch %d/%d - train_loss=%.4f train_acc=%.4f - val_loss=%.4f val_acc=%.4f",
                epoch, args.epochs, train_loss, train_acc, val_loss, val_acc,
            )

        test_loss, test_acc, test_prec, test_rec, test_labels, test_preds = run_epoch(
            model, loaders["test"], criterion, None, device
        )
        mlflow.log_metrics(
            {
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "test_precision": test_prec,
                "test_recall": test_rec,
            }
        )
        logger.info(
            "Test - loss=%.4f acc=%.4f precision=%.4f recall=%.4f",
            test_loss, test_acc, test_prec, test_rec,
        )

        model_path = MODEL_DIR / "model.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path))

        with tempfile.TemporaryDirectory() as tmp_dir:
            loss_curve_path = Path(tmp_dir) / "loss_curve.png"
            cm_path = Path(tmp_dir) / "confusion_matrix.png"
            plot_loss_curve(train_losses, val_losses, loss_curve_path)
            plot_confusion_matrix(test_labels, test_preds, cm_path)
            mlflow.log_artifact(str(loss_curve_path))
            mlflow.log_artifact(str(cm_path))

    logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    main()
