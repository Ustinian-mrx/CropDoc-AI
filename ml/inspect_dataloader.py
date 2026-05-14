import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import (
    TRAIN_DIR,
    VAL_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
)


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
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


def main():
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

    print(f"Train directory: {TRAIN_DIR}")
    print(f"Val directory: {VAL_DIR}")
    print(f"Train images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")
    print(f"Classes: {len(train_dataset.classes)}")

    print("\nClass to index mapping:")
    for class_name, class_index in train_dataset.class_to_idx.items():
        print(f"{class_index:02d}: {class_name}")

    images, labels = next(iter(train_loader))

    print("\nOne training batch:")
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")
    print(f"Batch labels: {labels.tolist()}")

    val_images, val_labels = next(iter(val_loader))

    print("\nOne validation batch:")
    print(f"Val batch images shape: {val_images.shape}")
    print(f"Val batch labels shape: {val_labels.shape}")


if __name__ == "__main__":
    main()
