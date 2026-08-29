"""PyTorch Dataset/DataLoader wrapper around the preprocessing manifests.

Reads `data/processed/{split}.csv` (see `preprocess.py`), resolves each
relative filepath against the raw data directory, and applies augmentation
(train split only, per the manifest's `augment` flag) + resize/normalize
on-the-fly.
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.preprocess import (
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    IMG_SIZE,
    augment_image,
    load_and_preprocess_image,
)


class CatsDogsDataset(Dataset):
    def __init__(self, split: str, raw_dir=DEFAULT_RAW_DIR, processed_dir=DEFAULT_PROCESSED_DIR, img_size: int = IMG_SIZE):
        manifest_path = processed_dir / f"{split}.csv"
        self.df = pd.read_csv(manifest_path)
        self.raw_dir = raw_dir
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        path = self.raw_dir / row["filepath"]

        if row["augment"]:
            with Image.open(path) as img:
                img = augment_image(img.convert("RGB"))
                img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
                array = np.asarray(img, dtype=np.float32) / 255.0
        else:
            array = load_and_preprocess_image(path, self.img_size)

        image = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        label = torch.tensor(row["label"], dtype=torch.float32)
        return image, label


def get_dataloader(
    split: str,
    raw_dir=DEFAULT_RAW_DIR,
    processed_dir=DEFAULT_PROCESSED_DIR,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DataLoader:
    dataset = CatsDogsDataset(split, raw_dir=raw_dir, processed_dir=processed_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
    )
