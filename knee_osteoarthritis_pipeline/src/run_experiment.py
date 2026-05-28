"""
run_experiment.py — Run a single experiment version end-to-end.

Usage (standalone):
    cd knee_osteoarthritis_pipeline/src
    python run_experiment.py --version v1_baseline --epochs 5

Called internally by run_all.py.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import (
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, MIXUP_ALPHA, FOCAL_GAMMA,
    NUM_CLASSES, IMG_SIZE, DATA_DIR, VERSIONS,
    get_device,
)
from data.dataset import get_dataloaders
from models.resnet import get_resnet50_model
from training.loss import build_loss
from training.trainer import Trainer
from evaluation.evaluator import evaluate_on_test


SEED = 42


def set_seed(seed: int = SEED):
    """Fix all random sources for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops (slight performance cost — disable if speed matters more)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_version(version_cfg: dict, args) -> dict:
    """
    Execute one experiment version (train + evaluate) and return best metrics.

    Args:
        version_cfg: one entry from VERSIONS list in config.py
        args:        parsed CLI arguments (may override config defaults)

    Returns:
        best_metrics dict from Trainer.fit()
    """
    name        = version_cfg["name"]
    use_mixup   = version_cfg["use_mixup"]
    loss_type   = version_cfg["loss_type"]
    use_sampler = version_cfg["use_sampler"]

    epochs      = args.epochs      if args.epochs      is not None else EPOCHS
    batch_size  = args.batch_size  if args.batch_size  is not None else BATCH_SIZE
    data_dir    = args.data_dir    if args.data_dir    is not None else DATA_DIR
    results_dir = args.results_dir if args.results_dir is not None else "../results"
    save_dir    = os.path.join(results_dir, name)

    # Same seed for every version → fair comparison
    set_seed(SEED)

    device = get_device()
    print(f"\n{'='*70}")
    print(f"  VERSION : {name}")
    print(f"  Loss    : {loss_type}   |  Mixup: {use_mixup}  |  Sampler: {use_sampler}")
    print(f"  Device  : {device}      |  Epochs: {epochs}   |  Seed: {SEED}")
    print(f"{'='*70}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, class_weights, class_counts = get_dataloaders(
        data_dir    = data_dir,
        batch_size  = batch_size,
        img_size    = IMG_SIZE,
        use_sampler = use_sampler,
    )

    if len(train_loader.dataset) == 0:
        print(f"  [ERROR] No training data found in {data_dir}. Skipping {name}.")
        return {}

    print(f"  Train: {len(train_loader.dataset)}  |  Val: {len(val_loader.dataset)}  "
          f"|  Test: {len(test_loader.dataset)}")
    print(f"  Class counts (train): {class_counts.astype(int)}")

    # Try to get class names from dataset
    try:
        class_names = train_loader.dataset.classes
    except AttributeError:
        class_names = [f"Grade {i}" for i in range(NUM_CLASSES)]

    # ── Model ─────────────────────────────────────────────────────────────────
    model = get_resnet50_model(num_classes=NUM_CLASSES, pretrained=True).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = build_loss(
        loss_type     = loss_type,
        class_weights = class_weights,
        class_counts  = class_counts,
        device        = device,
        focal_gamma   = FOCAL_GAMMA,
    )

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    # ── Training ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model                   = model,
        train_loader            = train_loader,
        val_loader              = val_loader,
        criterion               = criterion,
        optimizer               = optimizer,
        device                  = device,
        version_name            = name,
        lr_scheduler            = scheduler,
        early_stopping_patience = EARLY_STOPPING_PATIENCE,
        use_mixup               = use_mixup,
        mixup_alpha             = MIXUP_ALPHA,
    )

    _, best_metrics = trainer.fit(epochs=epochs, save_dir=save_dir)

    # ── Test-set evaluation (load best weights before evaluating) ─────────────
    best_model_path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"\n  Loading best model weights for test evaluation…")
        state = torch.load(best_model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        evaluate_on_test(
            model        = model,
            test_loader  = test_loader,
            criterion    = criterion,
            device       = device,
            save_dir     = save_dir,
            version_name = name,
            class_names  = class_names,
        )
    else:
        print(f"  [WARN] No saved model found at {best_model_path}, skipping test eval.")

    return best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point (for single-version runs)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run a single experiment version")
    parser.add_argument("--version",     type=str, default=None,
                        help="Version name (e.g. v1_baseline). Runs all if omitted.")
    parser.add_argument("--epochs",      type=int, default=None)
    parser.add_argument("--batch_size",  type=int, default=None)
    parser.add_argument("--data_dir",    type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.version:
        cfg = next((v for v in VERSIONS if v["name"] == args.version), None)
        if cfg is None:
            print(f"[ERROR] Unknown version '{args.version}'. "
                  f"Available: {[v['name'] for v in VERSIONS]}")
            sys.exit(1)
        run_version(cfg, args)
    else:
        print("No --version specified. Use run_all.py to run all versions.")
