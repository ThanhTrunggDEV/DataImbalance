import os
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms
from typing import Optional


class GradCAM:
    """Grad-CAM for ResNet50: hooks into model.layer4 (last conv stage)."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.feature_maps: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self._backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.feature_maps = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, x: torch.Tensor, class_idx: Optional[int] = None) -> np.ndarray:
        self.model.zero_grad()
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(dim=1).item()
        target = out[0, class_idx]
        target.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.feature_maps).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def cleanup(self):
        self._forward_handle.remove()
        self._backward_handle.remove()


def overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Overlay a Grad-CAM heatmap on an image."""
    cam_img = Image.fromarray(np.uint8(plt.cm.jet(cam)[:, :, :3] * 255)).resize(image.size, Image.BICUBIC)
    blended = Image.blend(image.convert("RGB"), cam_img, alpha)
    return blended


def run_gradcam(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str,
    version_name: str = "model",
    num_per_class: int = 3,
    seed: int = 42,
):
    """Generate Grad-CAM visualisations for each class in test set."""
    os.makedirs(output_dir, exist_ok=True)
    layer = model.layer4
    cam_extractor = GradCAM(model, layer)

    fix, seen = torch.Generator().manual_seed(seed), {}
    all_images = list(test_loader.dataset.image_paths)
    all_labels = list(test_loader.dataset.labels)
    indices = torch.randperm(len(all_images), generator=fix).tolist()

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    rows = []
    for idx in indices:
        lbl = all_labels[idx]
        if lbl in seen and seen[lbl] >= num_per_class:
            continue
        seen.setdefault(lbl, 0)
        seen[lbl] += 1

        path = all_images[idx]
        pil_img = Image.open(path).convert("RGB")
        orig = pil_img.resize((224, 224))
        input_tensor = val_transform(pil_img).unsqueeze(0).to(device)

        with torch.set_grad_enabled(True):
            cam = cam_extractor.generate(input_tensor, class_idx=lbl)
        overlayed = overlay_heatmap(orig, cam, alpha=0.5)
        rows.append((orig, overlayed, cam, lbl, os.path.basename(path)))
    cam_extractor.cleanup()

    n = len(rows)
    cols = 4
    fig, axes = plt.subplots(n, cols, figsize=(4 * cols, 3 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    class_names = ["Grade 0\nHealthy", "Grade 1\nDoubtful", "Grade 2\nMinimal", "Grade 3\nModerate", "Grade 4\nSevere"]
    for i, (orig, over, cam, lbl, fname) in enumerate(rows):
        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f"Original — {class_names[lbl]}", fontsize=9)
        axes[i, 0].axis("off")
        axes[i, 1].imshow(over)
        axes[i, 1].set_title(f"Grad-CAM overlay", fontsize=9)
        axes[i, 1].axis("off")
        im = axes[i, 2].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[i, 2].set_title(f"Heatmap", fontsize=9)
        axes[i, 2].axis("off")
        axes[i, 3].axis("off")
        axes[i, 3].text(0.1, 0.5, f"Image: {fname}\nTrue: {lbl} ({class_names[lbl]})", fontsize=8, va="center")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"gradcam_{version_name}.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_gradcam_comparison(
    models: dict[str, torch.nn.Module],
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str,
    num_per_class: int = 2,
    seed: int = 42,
):
    """Compare Grad-CAM across multiple models for the same test images."""
    os.makedirs(output_dir, exist_ok=True)

    fix, seen = torch.Generator().manual_seed(seed), {}
    all_images = list(test_loader.dataset.image_paths)
    all_labels = list(test_loader.dataset.labels)
    indices = torch.randperm(len(all_images), generator=fix).tolist()

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    selected = []
    for idx in indices:
        lbl = all_labels[idx]
        if lbl in seen and seen[lbl] >= num_per_class:
            continue
        seen.setdefault(lbl, 0)
        seen[lbl] += 1
        selected.append((all_images[idx], lbl))

    n_models = len(models)
    version_names = list(models.keys())
    class_names = ["Gr 0", "Gr 1", "Gr 2", "Gr 3", "Gr 4"]

    fig, axes = plt.subplots(len(selected), n_models + 1, figsize=(4 * (n_models + 1), 3 * len(selected)))
    if len(selected) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, (path, lbl) in enumerate(selected):
        pil_img = Image.open(path).convert("RGB")
        orig = pil_img.resize((224, 224))
        axes[row_idx, 0].imshow(orig)
        axes[row_idx, 0].set_title(f"Original\n{class_names[lbl]}", fontsize=9)
        axes[row_idx, 0].axis("off")

        for col_idx, (vname, model) in enumerate(models.items()):
            layer = model.layer4
            extractor = GradCAM(model, layer)
            input_tensor = val_transform(pil_img).unsqueeze(0).to(device)
            with torch.set_grad_enabled(True):
                cam = extractor.generate(input_tensor, class_idx=lbl)
            extractor.cleanup()
            over = overlay_heatmap(orig, cam, alpha=0.5)
            axes[row_idx, col_idx + 1].imshow(over)
            axes[row_idx, col_idx + 1].set_title(vname, fontsize=8)
            axes[row_idx, col_idx + 1].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "gradcam_comparison.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path
