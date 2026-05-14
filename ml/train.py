import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from config import (
    TRAIN_DIR,
    VAL_DIR,
    MODELS_DIR,
    IMAGE_SIZE,
    NUM_CLASSES,
    BATCH_SIZE,
    NUM_WORKERS,
    RANDOM_SEED,
    LEARNING_RATE,
    EPOCHS,
    MODEL_NAME,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, val_transform


def build_dataloaders():
    train_transform, val_transform = build_transforms()

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=train_transform,
    )

    val_dataset = datasets.ImageFolder(
        root=VAL_DIR,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return train_dataset, val_dataset, train_loader, val_loader


def build_model():
    weights = models.EfficientNet_B0_Weights.DEFAULT

    model = models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        NUM_CLASSES,
    )

    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
        })

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def evaluate(model, val_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size

            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def save_checkpoint(model, class_to_idx, best_val_acc):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / MODEL_NAME
    temp_path = MODELS_DIR / f"{MODEL_NAME}.tmp"

    cpu_state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    checkpoint = {
        "model_state_dict": cpu_state_dict,
        "class_to_idx": class_to_idx,
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
        "best_val_acc": best_val_acc,
    }

    torch.save(
        checkpoint,
        temp_path,
        _use_new_zipfile_serialization=False,
    )

    temp_path.replace(model_path)

    print(f"Best model saved to: {model_path}")


def main():
    set_seed(RANDOM_SEED)

    device = get_device()
    print(f"Using device: {device}")

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders()

    print(f"Train images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")
    print(f"Classes: {len(train_dataset.classes)}")

    model = build_model()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model=model,
                class_to_idx=train_dataset.class_to_idx,
                best_val_acc=best_val_acc,
            )

    print("\nTraining finished.")
    print(f"Best Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
