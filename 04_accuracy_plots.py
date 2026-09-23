"""Step 4: Track accuracy and plot learning curves.

Extends the training loop from 03_train.py to also record training and
validation accuracy at each epoch, then plots loss and accuracy vs. epoch.
Same data, model, loss (CrossEntropyLoss), optimizer (SGD, lr=0.1) and
number of epochs (20) as before.
"""

import importlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# The step modules' names start with a digit, so a plain import won't work
data_setup = importlib.import_module("01_data_setup")
model_module = importlib.import_module("02_model")

train_loader = data_setup.train_loader
valid_loader = data_setup.valid_loader
ImageClassifier = model_module.ImageClassifier
device = model_module.device


def evaluate(model, data_loader, criterion):
    """Return (mean loss, accuracy) over the whole data set."""
    model.eval()
    total_loss, n_correct = 0., 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            total_loss += criterion(y_pred, y_batch).item() * len(X_batch)
            n_correct += (y_pred.argmax(dim=1) == y_batch).sum().item()
    n_samples = len(data_loader.dataset)
    return total_loss / n_samples, n_correct / n_samples


def train(model, optimizer, criterion, train_loader, valid_loader, n_epochs):
    history = {"train_losses": [], "valid_losses": [],
               "train_accuracies": [], "valid_accuracies": []}
    for epoch in range(n_epochs):
        model.train()
        total_loss, n_correct = 0., 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item() * len(X_batch)
            n_correct += (y_pred.argmax(dim=1) == y_batch).sum().item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        n_samples = len(train_loader.dataset)
        history["train_losses"].append(total_loss / n_samples)
        history["train_accuracies"].append(n_correct / n_samples)
        valid_loss, valid_accuracy = evaluate(model, valid_loader, criterion)
        history["valid_losses"].append(valid_loss)
        history["valid_accuracies"].append(valid_accuracy)
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"train acc: {history['train_accuracies'][-1]:.4f}, "
              f"valid loss: {history['valid_losses'][-1]:.4f}, "
              f"valid acc: {history['valid_accuracies'][-1]:.4f}")
    return history


def plot_curves(history, train_key, valid_key, ylabel, title, filename):
    n_epochs = len(history[train_key])
    epochs = np.arange(1, n_epochs + 1)
    plt.figure(figsize=(8, 5))
    # Training values are averaged while the weights change during the epoch,
    # so (as in the notebook) they're plotted half an epoch earlier
    plt.plot(epochs - 0.5, history[train_key], "o--", color="tab:blue",
             label="Training")
    plt.plot(epochs, history[valid_key], "s-", color="tab:orange",
             label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(range(0, n_epochs + 1, 2))
    plt.xlim(0, n_epochs + 0.5)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")


if __name__ == "__main__":
    n_epochs = 20
    torch.manual_seed(42)
    model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=300, n_hidden2=100,
                            n_classes=10).to(device)
    xentropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    print(f"Training on {device} for {n_epochs} epochs")
    history = train(model, optimizer, xentropy, train_loader, valid_loader,
                    n_epochs)

    plot_curves(history, "train_losses", "valid_losses", "Cross-entropy loss",
                "Training and validation loss", "loss_curves.png")
    plot_curves(history, "train_accuracies", "valid_accuracies", "Accuracy",
                "Training and validation accuracy", "accuracy_curves.png")
    plt.show()
