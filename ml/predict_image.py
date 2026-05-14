import argparse

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from config import (
    IMAGE_SIZE,
    MODELS_DIR,
    MODEL_NAME,
    NUM_CLASSES,
)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def build_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def build_model():
    model = models.efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(
        in_features,
        NUM_CLASSES,
    )

    return model


def load_checkpoint(model, device):
    model_path = MODELS_DIR / MODEL_NAME

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {
        index: class_name
        for class_name, index in class_to_idx.items()
    }

    return model, idx_to_class, checkpoint


def predict_image(image_path: str, top_k: int = 3):
    device = get_device()

    transform = build_transform()
    model = build_model()
    model, idx_to_class, checkpoint = load_checkpoint(model, device)

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(
            probabilities,
            k=top_k,
            dim=1,
        )

    results = []

    for prob, index in zip(top_probs[0], top_indices[0]):
        class_index = index.item()
        class_name = idx_to_class[class_index]
        confidence = prob.item()

        results.append({
            "class_name": class_name,
            "confidence": confidence,
        })

    return results, checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Predict crop disease from a single image.",
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to show.",
    )

    args = parser.parse_args()

    results, checkpoint = predict_image(
        image_path=args.image,
        top_k=args.top_k,
    )

    best_result = results[0]

    print(f"Image: {args.image}")
    print(f"Model best val acc: {checkpoint.get('best_val_acc'):.4f}")
    print()
    print(f"Predicted class: {best_result['class_name']}")
    print(f"Confidence: {best_result['confidence'] * 100:.2f}%")

    print()
    print(f"Top-{args.top_k} predictions:")

    for index, result in enumerate(results, start=1):
        confidence = result["confidence"] * 100
        print(f"{index}. {result['class_name']} - {confidence:.2f}%")


if __name__ == "__main__":
    main()
