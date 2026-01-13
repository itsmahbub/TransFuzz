import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights
import argparse
import os

from model_wrappers.resnet_wrapper import ResNetWrapper

# === Configuration ===
MODEL_PATH = "resnet50_unsafebench.pth"
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 2  # safe vs. unsafe
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# === Load and Prepare Data ===
def get_dataloaders():
    model_wrapper = ResNetWrapper()
    train_ds = model_wrapper.get_seeds("UnsafeBench", split="train", preprocessed=True)
    test_ds = model_wrapper.get_seeds("UnsafeBench", split="test", preprocessed=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    return train_loader, test_loader




# === Save Model ===
def save_model(model):
    torch.save(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


# === Load Model from File ===
def load_model():
    if os.path.exists(MODEL_PATH):
        model = torch.load(MODEL_PATH, map_location=DEVICE)
        print(f"Loaded model from {MODEL_PATH}")
    else:
        print(f"Model file not found at {MODEL_PATH}. Starting fresh.")
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        for param in model.parameters():
            param.requires_grad = False   # freeze backbone
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


# === Training Loop ===
def train(model, train_loader):
    model.to(DEVICE)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {total_loss:.4f} - Acc: {correct/total:.4f}")

    save_model(model)


# === Evaluation Loop ===
def test(model, test_loader):
    model.to(DEVICE)
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {correct/total:.4f}")


# === Main Entrypoint ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="train or test")
    args = parser.parse_args()

    train_loader, test_loader = get_dataloaders()
    model = load_model()

    if args.mode == "train":
        
        train(model, train_loader)

    if args.mode == "test":
        test(model, test_loader)


if __name__ == "__main__":
    main()
