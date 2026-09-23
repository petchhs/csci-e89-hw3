"""Step 2: Model architecture for the Fashion MNIST image classifier.

Reproduces the "Building the Classifier" part of the
"Building an Image Classifier with PyTorch" section in
10_neural_nets_with_pytorch.ipynb. Training comes in a later step.
"""

import importlib

import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class ImageClassifier(nn.Module):
    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_classes):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_classes)
        )

    def forward(self, X):
        return self.mlp(X)


if __name__ == "__main__":
    torch.manual_seed(42)
    model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=300, n_hidden2=100,
                            n_classes=10).to(device)
    print(f"Device: {device}")
    print(model)

    n_params = sum([param.numel() for param in model.parameters()])
    print(f"Trainable parameters: {n_params:,}")

    # Forward pass on one real batch from step 1 (module name starts with a
    # digit, so it can't be imported with a plain import statement)
    data_setup = importlib.import_module("01_data_setup")
    X_batch, y_batch = next(iter(data_setup.train_loader))
    X_batch = X_batch.to(device)

    model.eval()
    with torch.no_grad():
        y_logits = model(X_batch)

    print(f"Input batch shape:  {tuple(X_batch.shape)}")
    print(f"Output logits shape: {tuple(y_logits.shape)}")
    assert y_logits.shape == (X_batch.shape[0], 10)
    print("Shape check passed: one logit per class for each image.")
