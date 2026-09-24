"""CSCI E-89 Assignment 03, Problem 4 — Pornjira Pongsuwan.

Step 3: Train the Fashion-MNIST classifier for 20 epochs, following the
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
EPOCHS = 20
LEARNING_RATE = 0.1
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
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    history = {"train_loss": [], "valid_loss": []}

    print(f"Device: {device}", flush=True)
    print(f"Dataset sizes: train={len(train_loader.dataset):,}, "
          f"validation={len(valid_loader.dataset):,}, test={len(test_loader.dataset):,}",
          flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Weight batch means by sample count, including the smaller last batch.
            batch_size = labels.size(0)
            train_loss_sum += loss.item() * batch_size
            train_samples += batch_size

        model.eval()
        valid_loss_sum = 0.0
        valid_samples = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                batch_size = labels.size(0)
                valid_loss_sum += loss.item() * batch_size
                valid_samples += batch_size

        train_loss = train_loss_sum / train_samples
        valid_loss = valid_loss_sum / valid_samples
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        print(f"Epoch {epoch:02d}/{EPOCHS} | Training loss: {train_loss:.6f} | "
              f"Validation loss: {valid_loss:.6f}", flush=True)

    print(f"Final training loss: {history['train_loss'][-1]:.6f}", flush=True)
    print(f"Final validation loss: {history['valid_loss'][-1]:.6f}", flush=True)
    return history


if __name__ == "__main__":
    main()
