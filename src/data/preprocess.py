"""Build train/val/test manifests for the Cats vs Dogs dataset.

Raw images are kept on disk (not versioned); this script validates them,
performs a stratified 80/10/10 split, and writes lightweight CSV manifests
(filepath, label, class_name) to `data/processed/`. Resize/normalize/
augmentation are applied on-the-fly at load time (see `dataset.py`) rather
than persisted, so `data/processed/` stays small enough for Git LFS.
"""

import argparse
import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, UnidentifiedImageError
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IMG_SIZE = 224
CLASSES = ["Cat", "Dog"]
LABEL_MAP = {name: idx for idx, name in enumerate(CLASSES)}
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1
SEED = 42

DEFAULT_RAW_DIR = (
    Path(__file__).resolve().parents[2]
    / "dog-and-cat-classification-dataset"
    / "versions"
    / "1"
    / "PetImages"
)
DEFAULT_PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


@dataclass(frozen=True)
class Record:
    filepath: str
    label: int
    class_name: str


def is_valid_image(path: Path) -> bool:
    """Return True if `path` can be opened and fully decoded as an image.

    PIL's `verify()` only checks file structure, not pixel data, so a
    truncated/corrupt image can pass it but raise (or warn) on the actual
    decode below. Warnings (e.g. "Truncated File Read") are promoted to
    errors here so such images are treated as invalid rather than silently
    loaded with incomplete pixel data.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with Image.open(path) as img:
                img.convert("RGB")
        return True
    except (UnidentifiedImageError, OSError, Warning):
        return False


def list_valid_images(raw_dir: Path) -> list[Record]:
    """Scan `raw_dir/{Cat,Dog}` and return records for decodable images only."""
    records: list[Record] = []
    for class_name in CLASSES:
        class_dir = raw_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class directory not found: {class_dir}")

        label = LABEL_MAP[class_name]
        valid, skipped = 0, 0
        for path in sorted(class_dir.iterdir()):
            if not path.is_file():
                continue
            if is_valid_image(path):
                rel_path = path.relative_to(raw_dir).as_posix()
                records.append(Record(filepath=rel_path, label=label, class_name=class_name))
                valid += 1
            else:
                skipped += 1
                logger.warning("Skipping unreadable image: %s", path)
        logger.info("%s: %d valid, %d skipped", class_name, valid, skipped)
    return records


def stratified_split(
    records: list[Record],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = SEED,
) -> dict[str, list[Record]]:
    """Split records into train/val/test, stratified by class label."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    labels = [r.label for r in records]
    train, rest = train_test_split(
        records, train_size=train_ratio, random_state=seed, stratify=labels
    )
    rest_labels = [r.label for r in rest]
    val_size = val_ratio / (val_ratio + test_ratio)
    val, test = train_test_split(
        rest, train_size=val_size, random_state=seed, stratify=rest_labels
    )
    return {"train": train, "val": val, "test": test}


def load_and_preprocess_image(path: str | Path, img_size: int = IMG_SIZE) -> np.ndarray:
    """Load an image, convert to RGB, resize to (img_size, img_size), normalize to [0, 1]."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize((img_size, img_size), Image.BILINEAR)
        array = np.asarray(img, dtype=np.float32) / 255.0
    return array


def augment_image(img: Image.Image, seed: int | None = None) -> Image.Image:
    """Apply random flip/rotation/color-jitter augmentation. Training split only."""
    rng = np.random.default_rng(seed)

    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    angle = float(rng.uniform(-20, 20))
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = float(rng.uniform(0.8, 1.2))
        img = enhancer_cls(img).enhance(factor)

    return img


def write_manifests(splits: dict[str, list[Record]], processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in splits.items():
        df = pd.DataFrame(
            {
                "filepath": [r.filepath for r in records],
                "label": [r.label for r in records],
                "class_name": [r.class_name for r in records],
                "augment": split_name == "train",
            }
        )
        out_path = processed_dir / f"{split_name}.csv"
        df.to_csv(out_path, index=False)
        logger.info("Wrote %d rows to %s", len(df), out_path)


def build_manifest(
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    seed: int = SEED,
) -> dict[str, list[Record]]:
    records = list_valid_images(raw_dir)
    splits = stratified_split(records, seed=seed)
    write_manifests(splits, processed_dir)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    splits = build_manifest(args.raw_dir, args.processed_dir, args.seed)
    for name, records in splits.items():
        n_cat = sum(1 for r in records if r.class_name == "Cat")
        n_dog = sum(1 for r in records if r.class_name == "Dog")
        logger.info("%s split: %d total (%d Cat / %d Dog)", name, len(records), n_cat, n_dog)


if __name__ == "__main__":
    main()
