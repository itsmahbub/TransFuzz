import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification

from model_wrappers.mobile_vit_wrapper import MobileViTWrapper

# === Configuration ===
MODEL_NAME = "apple/mobilevit-small"
MODEL_DIR = "mobilevit_unsafebench"
NUM_CLASSES = 2
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label2id = {"Safe": 0, "Unsafe": 1}
id2label = {0: "Safe", 1: "Unsafe"}


# === Model Save/Load ===
def save_model(model, processor, save_dir=MODEL_DIR):
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"Model and processor saved to {save_dir}")

def load_model(load_dir=MODEL_DIR):
    if os.path.exists(load_dir):
        print(f"Loading model from {load_dir}")
        model = AutoModelForImageClassification.from_pretrained(load_dir)
        processor = AutoImageProcessor.from_pretrained(load_dir)
    else:
        print(f"{load_dir} not found, loading base model {MODEL_NAME}")
        model = AutoModelForImageClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_CLASSES,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    return model, processor


# === Training Loop ===
def train(model, train_loader):
    model.train()
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(NUM_EPOCHS):
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(pixel_values=inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {total_loss:.4f} - Acc: {acc:.4f}")

# === Evaluation Loop ===
def test(model, test_loader):
    model.eval()
    model.to(DEVICE)
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(pixel_values=inputs)
            preds = outputs.logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {correct / total:.4f}")


# === Main Entrypoint ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], required=True)
    args = parser.parse_args()

    # Load model & processor
    model, processor = load_model()

    model_wrapper = MobileViTWrapper()
    train_ds = model_wrapper.get_seeds("UnsafeBench", split="train", preprocessed=True)
    test_ds = model_wrapper.get_seeds("UnsafeBench", split="test", preprocessed=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Run mode
    if args.mode == "train":
        train(model, train_loader)
        save_model(model, processor)

    elif args.mode == "test":
        test(model, test_loader)


if __name__ == "__main__":
    main()
