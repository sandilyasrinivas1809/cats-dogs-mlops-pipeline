import numpy as np
import pytest
from PIL import Image

from src.data.preprocess import (
    IMG_SIZE,
    Record,
    augment_image,
    is_valid_image,
    preprocess_pil_image,
    stratified_split,
)


def make_image(size=(300, 200), color=(255, 0, 0)):
    return Image.new("RGB", size, color)


def test_preprocess_pil_image_shape_and_dtype():
    array = preprocess_pil_image(make_image(), IMG_SIZE)
    assert array.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert array.dtype == np.float32


def test_preprocess_pil_image_normalizes_to_unit_range():
    array = preprocess_pil_image(make_image(size=(10, 10), color=(128, 64, 0)), img_size=10)
    assert array.min() >= 0.0
    assert array.max() <= 1.0
    assert np.allclose(array[0, 0], [128 / 255.0, 64 / 255.0, 0.0], atol=1e-6)


def test_augment_image_preserves_rgb_mode():
    augmented = augment_image(make_image(size=(50, 50)))
    assert augmented.mode == "RGB"


def test_is_valid_image_accepts_real_image(tmp_path):
    path = tmp_path / "good.jpg"
    make_image().save(path)
    assert is_valid_image(path) is True


def test_is_valid_image_rejects_corrupt_file(tmp_path):
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"not an image")
    assert is_valid_image(path) is False


def test_stratified_split_respects_ratios_and_balance():
    records = [
        Record(filepath=f"{i}.jpg", label=i % 2, class_name="Cat" if i % 2 == 0 else "Dog")
        for i in range(100)
    ]
    splits = stratified_split(records, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=0)

    assert len(splits["train"]) == 80
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 10

    for split_records in splits.values():
        labels = [r.label for r in split_records]
        assert sum(labels) == len(labels) // 2


def test_stratified_split_rejects_bad_ratios():
    records = [Record(filepath="0.jpg", label=0, class_name="Cat")]
    with pytest.raises(ValueError):
        stratified_split(records, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)
