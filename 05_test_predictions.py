"""Step 5: Evaluate on the test set and show sample predictions.

Trains the classifier with the same 20-epoch setup as the previous steps,
reports the test accuracy, then shows the predicted and actual class for
3 sample test images (as in the notebook's prediction cells).
"""

import importlib

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# The step modules' names start with a digit, so a plain import won't work
data_setup = importlib.import_module("01_data_setup")
model_module = importlib.import_module("02_model")
training = importlib.import_module("04_accuracy_plots")

train_loader = data_setup.train_loader
valid_loader = data_setup.valid_loader
test_loader = data_setup.test_loader
class_names = data_setup.train_and_valid_data.classes
ImageClassifier = model_module.ImageClassifier
device = model_module.device


if __name__ == "__main__":
    # 1. Train with the same setup as before
    n_epochs = 20
    torch.manual_seed(42)
    model = ImageClassifier(n_inputs=1 * 28 * 28, n_hidden1=300, n_hidden2=100,
                            n_classes=10).to(device)
    xentropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    print(f"Training on {device} for {n_epochs} epochs")
    training.train(model, optimizer, xentropy, train_loader, valid_loader,
                   n_epochs)

    # 2. Evaluate the final model on the test set
    test_loss, test_accuracy = training.evaluate(model, test_loader, xentropy)
    print(f"\nTest loss: {test_loss:.4f}, test accuracy: {test_accuracy:.4f} "
          f"({test_accuracy:.2%} of {len(test_loader.dataset):,} images)")

    # 3. Predict the first 3 test images
    model.eval()
    X_test, y_test = next(iter(test_loader))
    X_new, y_new = X_test[:3], y_test[:3]
    with torch.no_grad():
        y_pred_logits = model(X_new.to(device))
    y_proba = F.softmax(y_pred_logits, dim=1).cpu()
    y_pred = y_proba.argmax(dim=1)

    # 4. Show predicted vs. actual class names
    print("\nSample predictions:")
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.6))
    for i, ax in enumerate(axes):
        predicted = class_names[y_pred[i]]
        actual = class_names[y_new[i]]
        confidence = y_proba[i, y_pred[i]].item()
        correct = y_pred[i] == y_new[i]
        print(f"  Image {i + 1}: predicted = {predicted} ({confidence:.1%}), "
              f"actual = {actual} -> {'correct' if correct else 'WRONG'}")
        ax.imshow(X_new[i].squeeze(), cmap="binary")
        ax.set_title(f"Predicted: {predicted}\nActual: {actual}",
                     color="green" if correct else "red", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("test_predictions.png", dpi=150)
    print("Saved test_predictions.png")
    plt.show()
