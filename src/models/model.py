"""Baseline CNN for Cats vs Dogs binary classification."""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Conv/BatchNorm/Pool blocks -> dense, for 224x224x3 RGB input.

    Outputs a raw logit (no final sigmoid): pair with `nn.BCEWithLogitsLoss`
    for training and `torch.sigmoid(logit)` to get a probability at
    inference time. Without BatchNorm, the [0, 1]-normalized pixel inputs
    keep this network's pre-activations in too narrow a range for the
    sigmoid+BCELoss combo to produce any usable gradient signal - loss
    stays pinned at ln(2) indefinitely. BatchNorm fixes that, but then
    saturated sigmoid outputs make plain BCELoss numerically unstable
    (loss can spike into the tens); BCEWithLogitsLoss avoids this by
    combining the sigmoid and the loss in one numerically stable op.
    """

    def __init__(self, img_size: int = 224):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14
        )
        flattened_size = 128 * (img_size // 16) ** 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x).squeeze(1)
