"""
Generate Grad-CAM and t-SNE visualizations for paper.
Uses the 6 selected versions from the progressive narrative.

Usage:
    cd src
    python run_visualization.py                 # KOA (default)
    python run_visualization.py --dataset eyepacs
    python run_visualization.py --save_dir ../paper
"""

import argparse
import os, sys, importlib
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from models.resnet import get_resnet50_model
from visualization.gradcam import run_gradcam_comparison
from visualization.tsne import run_tsne_comparison

SELECTED_VERSIONS = [
    ("v1_baseline",              "Baseline (CE)"),
    ("v3_balanced_softmax",      "Balanced Softmax"),
    ("v5_focal_loss",            "Focal Loss"),
    ("v7_adjacent_balanced",     "Adjacent + BalSoft"),
    ("v11_owmixup_ce_t20",       "OWMixup (CE) τ=2.0"),
    ("v12_owmixup_balanced_t05", "OWMixup + BalSoft τ=0.5"),
]

PIPELINE_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(PIPELINE_ROOT, "results")
DATA_DIR = os.path.join(PIPELINE_ROOT, "data")
DEFAULT_OUTPUT_DIR = os.path.join(PIPELINE_ROOT, "report_assets", "vis")


def load_model(version_name: str, results_subdir: str,
               device: torch.device) -> torch.nn.Module:
    model = get_resnet50_model(num_classes=5, pretrained=False)
    ckpt_path = os.path.join(RESULTS_DIR, results_subdir, version_name, "best_model.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(RESULTS_DIR, results_subdir, version_name,
                                 "seed_42", "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def get_test_loader(dataset: str, batch_size: int = 64) -> DataLoader:
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if dataset == "KOA":
        from data.dataset import KneeDataset
        data_dir = os.path.join(PIPELINE_ROOT, "data")
        ds = KneeDataset(data_dir, split="test", transform=val_transform)
    elif dataset == "eyepacs":
        from data.dataset import EyePACSDataset
        data_dir = os.path.join(PIPELINE_ROOT, "data_dr")
        ds = EyePACSDataset(data_dir, split="test", transform=val_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="KOA", choices=["KOA", "eyepacs"])
    parser.add_argument("--save_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}  | Dataset: {args.dataset}")
    os.makedirs(args.save_dir, exist_ok=True)

    models = {}
    for vname, vdisp in SELECTED_VERSIONS:
        print(f"  Loading {vdisp} ({vname})...")
        try:
            models[vdisp] = load_model(vname, args.dataset, device)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")

    if not models:
        print("No models loaded. Exiting.")
        return

    test_loader = get_test_loader(args.dataset)
    suffix = f"_{args.dataset}"

    print("\n--- Grad-CAM ---")
    gradcam_path = run_gradcam_comparison(models, test_loader, device,
                                          args.save_dir, num_per_class=2,
                                          suffix=suffix)
    print(f"  Saved: {gradcam_path}")

    print("\n--- t-SNE ---")
    tsne_path = run_tsne_comparison(models, test_loader, device,
                                    args.save_dir, suffix=suffix)
    print(f"  Saved: {tsne_path}")

    print(f"\nDone! Assets saved to {args.save_dir}")


if __name__ == "__main__":
    main()
