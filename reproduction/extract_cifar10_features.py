#!/usr/bin/env python3
"""One-off preprocessing: extract pretrained ResNet-18 (ImageNet) features for
CIFAR10, exactly matching the paper's real-data protocol (official cifar10.py).

Deterministic configuration (frozen so the committed features are reproducible):
  * ResNet-18 with ImageNet weights, final FC replaced by Identity (512-dim).
  * CIFAR10 images resized to 224x224, ImageNet normalization.
  * n_train = n_test = 2000 sub-sample, seed 2026 (matches official cifar10.py).
  * Multi-output one-hot targets (K=10).

The output data/cifar10_resnet18_features.npz stores X_train, y_train,
X_test, y_test as float32/int64 arrays. The ridge self-distillation experiment
(claim 6) consumes this file directly, so the runtime environment stays
torch-free; rerunning this script regenerates the features bit-for-bit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms


SEED = 2026
N_TRAIN = 2000
N_TEST = 2000
K = 10
OUT = Path(__file__).resolve().parents[1] / "data" / "cifar10_resnet18_features.npz"


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_full = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_full = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Identity()
    model = model.to(device).eval()

    rng = np.random.default_rng(SEED)
    train_idx = rng.choice(len(train_full), N_TRAIN, replace=False)
    test_idx = rng.choice(len(test_full), N_TEST, replace=False)

    def extract(dataset, indices):
        feats, labels = [], []
        with torch.no_grad():
            for i in range(0, len(indices), 32):
                batch = torch.stack([dataset[int(j)][0] for j in indices[i:i + 32]])
                out = model(batch.to(device)).cpu().numpy()
                feats.append(out)
                labels.append(np.array([dataset[int(j)][1] for j in indices[i:i + 32]]))
        return np.vstack(feats).astype(np.float32), np.hstack(labels).astype(np.int64)

    X_train, y_train = extract(train_full, train_idx)
    X_test, y_test = extract(test_full, test_idx)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"Saved {OUT}  shapes X_train={X_train.shape} X_test={X_test.shape}")


if __name__ == "__main__":
    main()
