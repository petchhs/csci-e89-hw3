"""CSCI E-89 Assignment 03, Problem 4 — Pornjira Pongsuwan.

Step 4: Track loss and accuracy and plot learning curves, following the
"Building an Image Classifier with PyTorch" section of the reference notebook.
"""

from pathlib import Path
import random

import matplotlib

matplotlib.use("Agg")  # Save plots without opening a GUI window.
import matplotlib.pyplot as plt
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
    history = {"train_loss": [], "valid_loss": [],
               "train_accuracy": [], "valid_accuracy": []}

    print(f"Device: {device}", flush=True)
    print(f"Dataset sizes: train={len(train_loader.dataset):,}, "
          f"validation={len(valid_loader.dataset):,}, test={len(test_loader.dataset):,}",
          flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        train_correct = 0
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
            # Accuracy uses the same pre-update logits as the training loss.
            train_correct += (logits.detach().argmax(dim=1) == labels).sum().item()

        model.eval()
        valid_loss_sum = 0.0
        valid_samples = 0
        valid_correct = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                batch_size = labels.size(0)
                valid_loss_sum += loss.item() * batch_size
                valid_samples += batch_size
                valid_correct += (logits.argmax(dim=1) == labels).sum().item()

        train_loss = train_loss_sum / train_samples
        valid_loss = valid_loss_sum / valid_samples
        train_accuracy = train_correct / train_samples
        valid_accuracy = valid_correct / valid_samples
        history["train_accuracy"].append(train_accuracy)
        history["valid_accuracy"].append(valid_accuracy)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        print(f"Epoch {epoch:02d}/{EPOCHS} | Training loss: {train_loss:.6f} | "
              f"Validation loss: {valid_loss:.6f} | "
              f"Training accuracy: {train_accuracy:.2%} | "
              f"Validation accuracy: {valid_accuracy:.2%}", flush=True)

    print(f"Final training loss: {history['train_loss'][-1]:.6f}", flush=True)
    print(f"Final validation loss: {history['valid_loss'][-1]:.6f}", flush=True)
    print(f"Final training accuracy: {history['train_accuracy'][-1]:.2%}", flush=True)
    print(f"Final validation accuracy: {history['valid_accuracy'][-1]:.2%}", flush=True)
    print(f"Best validation accuracy: {max(history['valid_accuracy']):.2%}", flush=True)
    save_plots(history)
    return history


def save_plots(history):
    """Save separate loss and accuracy figures beside this script."""
    epochs = range(1, EPOCHS + 1)
    output_dir = Path(__file__).resolve().parent
    for metric, ylabel, title, filename, scale in [
        ("loss", "Average cross-entropy loss", "Fashion-MNIST: Training and Validation Loss",
         "codex_loss_curves.png", 1),
        ("accuracy", "Accuracy (%)", "Fashion-MNIST: Training and Validation Accuracy",
         "codex_accuracy_curves.png", 100),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        ax.plot(epochs, [value * scale for value in history[f"train_{metric}"]],
                marker="o", label="Training")
        ax.plot(epochs, [value * scale for value in history[f"valid_{metric}"]],
                marker="s", label="Validation")
        ax.set(xlabel="Epoch", ylabel=ylabel, title=title)
        ax.set_xticks(list(epochs))
        ax.grid(True, alpha=0.3)
        ax.legend()
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot: {output_path}", flush=True)


if __name__ == "__main__":
    main()
