"""
Generate Grad-CAM and t-SNE visualizations for paper.
Compares v1 (baseline), v2 (standard mixup), v7 (adjacent + BalSoft).

Usage:
    cd src
    python run_visualization.py
"""

import os, sys, importlib
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from data.dataset import KneeDataset

from models.resnet import get_resnet50_model
from visualization.gradcam import run_gradcam_comparison
from visualization.tsne import run_tsne_comparison

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "report_assets", "vis")


def load_model(version_name: str, device: torch.device) -> torch.nn.Module:
    """Load a trained model from its best_model.pth."""
    model = get_resnet50_model(num_classes=5, pretrained=False)
    ckpt_path = os.path.join(RESULTS_DIR, version_name, "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def get_test_loader(batch_size: int = 64) -> DataLoader:
    """Build a test loader with no augmentation."""
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_ds = KneeDataset(DATA_DIR, split="test", transform=val_transform)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    versions = {
        "v1 Baseline (CE)": "v1_baseline",
        "v2 Standard Mixup": "v2_mixup",
        "v7 Adjacent + BalSoft": "v7_adjacent_balanced",
        "v11 OWMix CE τ=1.0": "v11_owmixup_ce",
        "v12 OWMix BalSoft τ=2.0": "v12_owmixup_balanced_t20",
    }

    models = {}
    for display_name, vname in versions.items():
        print(f"Loading {vname}...")
        models[display_name] = load_model(vname, device)

    test_loader = get_test_loader()

    print("\n--- Grad-CAM ---")
    gradcam_path = run_gradcam_comparison(
        models, test_loader, device, OUTPUT_DIR, num_per_class=2
    )
    print(f"Saved: {gradcam_path}")

    print("\n--- t-SNE ---")
    tsne_path = run_tsne_comparison(models, test_loader, device, OUTPUT_DIR)
    print(f"Saved: {tsne_path}")

    print(f"\nDone! Assets saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
