"""CSCI E-89 Assignment 03, Problem 4 — Pornjira Pongsuwan.

Step 1: Fashion-MNIST data setup, following the "Building an Image
Classifier with PyTorch" section of 10_neural_nets_with_pytorch.ipynb.
"""

from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FashionMNIST
import torchvision.transforms.v2 as T


SEED = 42
BATCH_SIZE = 32
DATA_DIR = Path(__file__).resolve().parent / "datasets"


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

    # Convert uint8 pixels to float32 and scale from [0, 255] to [0, 1].
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

    image, label = train_data[0]
    batch_images, batch_labels = next(iter(train_loader))

    print(f"Device: {device}")
    print(f"Training samples: {len(train_loader.dataset):,}")
    print(f"Validation samples: {len(valid_loader.dataset):,}")
    print(f"Test samples: {len(test_loader.dataset):,}")
    print(f"Image shape (channels, height, width): {tuple(image.shape)}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image pixel range: [{image.min().item():.1f}, {image.max().item():.1f}]")
    print(f"Label: {label} ({train_and_valid_data.classes[label]})")
    print(f"Batch image shape: {tuple(batch_images.shape)}")
    print(f"Batch label shape: {tuple(batch_labels.shape)}")


if __name__ == "__main__":
    main()
