"""CSCI E-89 Assignment 03, Problem 4 — Pornjira Pongsuwan.

Step 2: Define and verify the Fashion-MNIST classifier, following the
"Building an Image Classifier with PyTorch" section of the reference notebook.
"""

from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
import torchvision.transforms.v2 as T


SEED = 42
BATCH_SIZE = 32
DATA_DIR = Path(__file__).resolve().parent / "datasets"


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 300),
            nn.ReLU(),
            nn.Linear(300, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, images):
        # CrossEntropyLoss expects raw logits, so do not apply softmax.
        return self.mlp(images)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Reproduce Step 1's data setup without modifying that script.
    to_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    train_and_valid_data = FashionMNIST(
        root=DATA_DIR, train=True, download=True, transform=to_tensor
    )
    test_data = FashionMNIST(
        root=DATA_DIR, train=False, download=True, transform=to_tensor
    )
    train_data, valid_data = random_split(
        train_and_valid_data,
        [55_000, 5_000],
        generator=torch.Generator().manual_seed(SEED),
    )
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    model = ImageClassifier().to(device)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Device: {device}")
    print(f"Dataset sizes: train={len(train_loader.dataset):,}, "
          f"validation={len(valid_loader.dataset):,}, test={len(test_loader.dataset):,}")
    print("Model architecture:")
    print(model)
    print(f"Total trainable parameters: {trainable_parameters:,}")

    batch_images, _ = next(iter(train_loader))
    batch_images = batch_images.to(device)
    model.eval()
    with torch.no_grad():
        logits = model(batch_images)

    expected_shape = (BATCH_SIZE, 10)
    assert tuple(logits.shape) == expected_shape, (
        f"Expected {expected_shape}, got {tuple(logits.shape)}"
    )
    print(f"Input batch shape: {tuple(batch_images.shape)}")
    print(f"Output shape: {tuple(logits.shape)} (verified)")
    print("Output contains raw logits; no softmax applied.")


if __name__ == "__main__":
    main()
