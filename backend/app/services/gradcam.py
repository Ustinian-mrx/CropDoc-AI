from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATIC_DIR = PROJECT_ROOT / "backend" / "static"
HEATMAP_DIR = STATIC_DIR / "heatmaps"


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.forward_hook = self.target_layer.register_forward_hook(
            self._save_activation,
        )
        self.backward_hook = self.target_layer.register_full_backward_hook(
            self._save_gradient,
        )

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor, class_index):
        self.model.zero_grad()

        outputs = self.model(image_tensor)
        score = outputs[:, class_index]
        score.backward(retain_graph=True)

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)

        cam = torch.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

        cam = cam - np.min(cam)

        max_value = np.max(cam)
        if max_value > 0:
            cam = cam / max_value

        return cam


def denormalize_image_tensor(image_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image_tensor = image_tensor.detach().cpu().squeeze(0)
    image_tensor = image_tensor * std + mean
    image_tensor = image_tensor.clamp(0, 1)

    image_array = image_tensor.permute(1, 2, 0).numpy()
    image_array = (image_array * 255).astype(np.uint8)

    return image_array


def save_gradcam_overlay(image_tensor, cam) -> str:
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    original_image = denormalize_image_tensor(image_tensor)

    height, width, _ = original_image.shape

    cam_resized = cv2.resize(cam, (width, height))
    heatmap = (cam_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(
        original_image,
        0.55,
        heatmap,
        0.45,
        0,
    )

    filename = f"{uuid4().hex}.jpg"
    output_path = HEATMAP_DIR / filename

    Image.fromarray(overlay).save(output_path, quality=95)

    return f"/static/heatmaps/{filename}"
