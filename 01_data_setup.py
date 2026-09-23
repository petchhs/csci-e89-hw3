"""Step 1: Data loading and preprocessing for the Fashion MNIST image classifier.

Reproduces the "Using TorchVision to Load the Dataset" part of the
"Building an Image Classifier with PyTorch" section in
10_neural_nets_with_pytorch.ipynb. The model and training loop come later.
"""

import torch
import torchvision
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

# Convert PIL images to float32 tensors with pixel values scaled to [0, 1]
toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

# Download Fashion MNIST (60,000 training images, 10,000 test images)
train_and_valid_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=True, download=True, transform=toTensor)
test_data = torchvision.datasets.FashionMNIST(
    root="datasets", train=False, download=True, transform=toTensor)

# Split the original training set into training and validation sets
torch.manual_seed(42)
train_data, valid_data = torch.utils.data.random_split(
    train_and_valid_data, [55_000, 5_000])

# Wrap the datasets in mini-batch loaders (only the training set is shuffled)
torch.manual_seed(42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)


if __name__ == "__main__":
    print(f"Training samples:   {len(train_data)}")
    print(f"Validation samples: {len(valid_data)}")
    print(f"Test samples:       {len(test_data)}")

    # Each entry is a tuple (image, target)
    X_sample, y_sample = train_data[0]
    print(f"Sample image shape: {X_sample.shape}")  # [channels, rows, columns]
    print(f"Sample image dtype: {X_sample.dtype}")
    print(f"Pixel value range:  [{X_sample.min():.2f}, {X_sample.max():.2f}]")
    print(f"Sample label:       {y_sample} "
          f"({train_and_valid_data.classes[y_sample]})")

    X_batch, y_batch = next(iter(train_loader))
    print(f"Batch shapes:       X={tuple(X_batch.shape)}, y={tuple(y_batch.shape)}")
    print(f"Classes:            {train_and_valid_data.classes}")
