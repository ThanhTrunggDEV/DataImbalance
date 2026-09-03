r"""
Generate the paper's qualitative figures (Grad-CAM + t-SNE) for the OWMM study.

Loads the six configurations reported in the manuscript and renders side-by-side
comparison panels. The OWMM column uses the validation-selected tempered prior per
dataset (KOA: gamma=0.5, EyePACS: gamma=0.25). Output filenames match the
\includegraphics paths in paper/manuscript.tex (figs/gradcam_<DS>.png, figs/tsne_<DS>.png).

Usage (from knee_osteoarthritis_pipeline/src):
    python run_visualization.py                    # KOA -> paper/figs/*_KOA.png
    python run_visualization.py --dataset eyepacs  # -> paper/figs/*_eyepacs.png
    python run_visualization.py --seed 42          # load a different seed subdir
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from models.resnet import get_resnet50_model
from visualization.gradcam import run_gradcam_comparison
from visualization.tsne import run_tsne_comparison

PIPELINE_ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(PIPELINE_ROOT, "results")
DEFAULT_OUTPUT_DIR = os.path.join(PIPELINE_ROOT, "..", "paper", "figs")

# Per-dataset routing: results subdir, data subdir, and the OWMM winner (validation-
# selected tempered prior). See select_gamma.py for the gamma-selection rationale.
DATASETS = {
    "KOA": {"results": "koa", "data": "data", "owmm": "v19_owmm_tempered_g05"},
    "eyepacs": {"results": "eyepacs", "data": "data_dr", "owmm": "v19_owmm_tempered_g025"},
}


def paper_versions(dataset: str):
    """The six configurations reported in the manuscript, in table order.
    The last (OWMM) column is dataset-dependent via the validation-selected gamma."""
    return [
        ("v1_baseline",          "Baseline (CE)"),
        ("v3_balanced_softmax",  "Balanced Softmax"),
        ("v5_focal_loss",        "Focal Loss"),
        ("v2_mixup",             "Mixup"),
        ("v7_adjacent_balanced", "Adjacent Mixup"),
        (DATASETS[dataset]["owmm"], "OWMM (ours)"),
    ]


def load_model(version_name: str, results_subdir: str,
               device: torch.device, seed: int) -> torch.nn.Module:
    """Load a trained ResNet-50 checkpoint (seed-nested, with legacy flat fallback).

    All configurations — OWMM included — export a standard ResNet-50 state dict, so
    Grad-CAM (layer4) and t-SNE (avgpool) read them uniformly; the OWMM manifold mixing
    only affected training, not the backbone architecture."""
    model = get_resnet50_model(num_classes=5, pretrained=False)
    candidates = [
        os.path.join(RESULTS_DIR, results_subdir, version_name, f"seed_{seed}", "best_model.pth"),
        os.path.join(RESULTS_DIR, results_subdir, version_name, "best_model.pth"),
    ]
    ckpt_path = next((p for p in candidates if os.path.exists(p)), None)
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint for {version_name} (seed {seed}) under {results_subdir}")
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
    from data.dataset import KneeDataset
    data_dir = os.path.join(PIPELINE_ROOT, DATASETS[dataset]["data"])
    ds = KneeDataset(data_dir, split="test", transform=val_transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def run_dataset(dataset: str, save_dir: str, device: torch.device, seed: int,
                do_gradcam: bool, do_tsne: bool):
    """Generate the (separate) Grad-CAM and t-SNE figures for one dataset."""
    results_subdir = DATASETS[dataset]["results"]
    print(f"\n{'='*60}\n[Dataset] {dataset}  | seed: {seed}  | results/{results_subdir}\n{'='*60}")

    models = {}
    for vname, vdisp in paper_versions(dataset):
        try:
            models[vdisp] = load_model(vname, results_subdir, device, seed)
            print(f"  loaded {vdisp:<18} <- {vname}")
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")

    if not models:
        print(f"  No models loaded for {dataset}; skipping.")
        return

    test_loader = get_test_loader(dataset)
    suffix = f"_{dataset}"  # -> _KOA / _eyepacs, matching manuscript figs

    if do_gradcam:
        print("  --- Grad-CAM ---")
        src = run_gradcam_comparison(models, test_loader, device, save_dir,
                                     num_per_class=1, suffix=suffix)
        dst = os.path.join(save_dir, f"gradcam{suffix}.png")
        if os.path.abspath(src) != os.path.abspath(dst):
            os.replace(src, dst)
        print(f"    Saved: {dst}")

    if do_tsne:
        print("  --- t-SNE ---")
        src = run_tsne_comparison(models, test_loader, device, save_dir, suffix=suffix)
        dst = os.path.join(save_dir, f"tsne{suffix}.png")
        if os.path.abspath(src) != os.path.abspath(dst):
            os.replace(src, dst)
        print(f"    Saved: {dst}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="both", choices=list(DATASETS) + ["both"],
                        help="KOA, eyepacs, or both (default: both, one run -> 4 separate PNGs).")
    parser.add_argument("--save_dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output dir (default: paper/figs, matching the manuscript).")
    parser.add_argument("--seed", type=int, default=123,
                        help="Seed subdir to load models from (default: 123, the paper seed).")
    parser.add_argument("--skip_gradcam", action="store_true")
    parser.add_argument("--skip_tsne", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.abspath(args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    datasets = list(DATASETS) if args.dataset == "both" else [args.dataset]
    print(f"[Device] {device}  | Datasets: {', '.join(datasets)}\n[Output] {save_dir}")

    for ds in datasets:
        run_dataset(ds, save_dir, device, args.seed,
                    do_gradcam=not args.skip_gradcam, do_tsne=not args.skip_tsne)

    print("\nDone. Figures are separate per dataset: "
          "figs/gradcam_KOA.png, figs/tsne_KOA.png, figs/gradcam_eyepacs.png, figs/tsne_eyepacs.png")


if __name__ == "__main__":
    main()
