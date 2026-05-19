from pathlib import Path
from typing import List, Dict

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
from app.services.label_utils import parse_class_name
from app.services.gradcam import GradCAM, save_gradcam_overlay



PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = PROJECT_ROOT / "models" / "efficientnet_b0_plantvillage.pth"

IMAGE_SIZE = 224
NUM_CLASSES = 38


class CropDiseasePredictor:
    def __init__(self):
        self.device = self._get_device()
        self.transform = self._build_transform()
        self.model = self._build_model()
        self.idx_to_class = {}
        self.gradcam = None


        self._load_checkpoint()

    def _get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    def _build_transform(self):
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _build_model(self):
        model = models.efficientnet_b0(weights=None)

        in_features = model.classifier[1].in_features

        model.classifier[1] = nn.Linear(
            in_features,
            NUM_CLASSES,
        )

        return model

    def _load_checkpoint(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        class_to_idx = checkpoint["class_to_idx"]

        self.idx_to_class = {
            index: class_name
            for class_name, index in class_to_idx.items()
        }
        target_layer = self.model.features[-1]
        
        self.gradcam = GradCAM(
            model=self.model,
            target_layer=target_layer,
        )


    def predict(self, image: Image.Image, top_k: int = 3) -> dict:
        image = image.convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(
                probabilities,
                k=top_k,
                dim=1,
            )

        results = []

        for prob, index in zip(top_probs[0], top_indices[0]):
            class_index = index.item()
            class_name = self.idx_to_class[class_index]
            confidence = prob.item()

            parsed_label = parse_class_name(class_name)

            results.append({
                **parsed_label,
                "confidence": confidence,
            })

        best_class_index = top_indices[0][0].item()

        cam = self.gradcam.generate(
            image_tensor=image_tensor,
            class_index=best_class_index,
        )

        heatmap_url = save_gradcam_overlay(
            image_tensor=image_tensor,
            cam=cam,
        )

        return {
            "top3": results,
            "heatmap_url": heatmap_url,
        }



predictor = CropDiseasePredictor()
