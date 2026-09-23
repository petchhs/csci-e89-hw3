"""Step 3: Train the Fashion MNIST image classifier.

Follows the training setup of the "Building an Image Classifier with PyTorch"
section in 10_neural_nets_with_pytorch.ipynb (CrossEntropyLoss, SGD with
lr=0.1), recording the training and validation loss at each epoch.
Accuracy and plots come in a later step.
"""

import importlib

import torch
import torch.nn as nn

# The step modules' names start with a digit, so a plain import won't work
data_setup = importlib.import_module("01_data_setup")
model_module = importlib.import_module("02_model")

train_loader = data_setup.train_loader
valid_loader = data_setup.valid_loader
ImageClassifier = model_module.ImageClassifier
device = model_module.device


def evaluate_loss(model, data_loader, criterion):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            total_loss += criterion(y_pred, y_batch).item() * len(X_batch)
    return total_loss / len(data_loader.dataset)


def train(model, optimizer, criterion, train_loader, valid_loader, n_epochs):
    history = {"train_losses": [], "valid_losses": []}
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item() * len(X_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        history["train_losses"].append(total_loss / len(train_loader.dataset))
        history["valid_losses"].append(
            evaluate_loss(model, valid_loader, criterion))
        print(f"Epoch {epoch + 1}/{n_epochs}, "
              f"train loss: {history['train_losses'][-1]:.4f}, "
              f"valid loss: {history['valid_losses'][-1]:.4f}")
    return history


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
