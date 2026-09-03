"""
run_experiment.py — Run a single experiment version end-to-end.

Usage (standalone):
    cd knee_osteoarthritis_pipeline/src
    python run_experiment.py --version v1_baseline --epochs 5
    python run_experiment.py --version v10_supcon --supcon_epochs 50 --probe_epochs 10

Called internally by run_all.py.
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.optim as optim

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import configs.config as cfg_mod
from configs.config import (
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE, MIXUP_ALPHA, MIXUP_ADJACENT_GAP, MIXUP_RULE_LAM_MAX,
    MIXUP_TEMPERATURE,
    FOCAL_GAMMA, SUPCON_TEMPERATURE, SUPCON_PROJECT_DIM,
    SUPCON_EPOCHS, PROBE_EPOCHS, FINETUNE_EPOCHS,
    NUM_CLASSES, IMG_SIZE, DATA_DIR, VERSIONS,
    get_device,
)
from data.dataset import get_dataloaders, get_supcon_train_loader
from models.resnet import (
    get_resnet50_model, get_resnet50_supcon_model,
    freeze_backbone, unfreeze_all,
)
from training.loss import build_loss, SupConLoss, BalancedSoftmaxLoss
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


def run_version(version_cfg: dict, args, seed: int = 42) -> dict:
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
    mixup_mode  = version_cfg.get("mixup_mode", "standard")
    loss_type   = version_cfg["loss_type"]
    use_sampler = version_cfg["use_sampler"]

    epochs      = args.epochs            if args.epochs      is not None else EPOCHS
    batch_size  = args.batch_size        if args.batch_size  is not None else BATCH_SIZE
    data_dir    = args.data_dir          if args.data_dir    is not None else os.path.join(_SCRIPT_DIR, DATA_DIR)
    results_dir = args.results_dir       if args.results_dir is not None else os.path.join(_SCRIPT_DIR, "../results")
    dataset     = getattr(args, 'dataset', None) or ''
    if dataset:
        results_dir = os.path.join(results_dir, dataset)

    version_dir = name
    mixup_temp = version_cfg.get("mixup_temperature")
    if mixup_temp is None:
        mixup_temp = getattr(args, 'mixup_temperature', None)
    if mixup_temp is not None:
        cfg_mod.MIXUP_TEMPERATURE = mixup_temp
        # Only append the tau suffix if the version name doesn't already encode
        # it. Configs like "v11_owmixup_ce_t20" bake the temperature into the
        # name *and* set mixup_temperature; appending again produced duplicate
        # "..._t20_t20" dirs that split seeds of the same experiment apart.
        suffix = f"_t{str(mixup_temp).replace('.', '')}"
        if not version_dir.endswith(suffix):
            version_dir += suffix

    # prior_gamma override: replace directory name so results go to v19_g{val}
    cli_gamma = getattr(args, "prior_gamma", None)
    cfg_gamma = version_cfg.get("prior_gamma", getattr(cfg_mod, "OWMM_PRIOR_GAMMA", 1.0))
    if cli_gamma is not None and abs(cli_gamma - cfg_gamma) > 1e-9:
        version_dir = f"v19_g{str(cli_gamma).replace('.', '')}"
    save_dir = os.path.join(results_dir, version_dir, f"seed_{seed}")

    set_seed(seed)

    cuda_id = getattr(args, 'cuda', None)
    if cuda_id is not None:
        cfg_mod.CUDA_DEVICE_ID = cuda_id
    device = get_device()
    print(f"\n{'='*70}")
    print(f"  VERSION : {name}")
    if dataset:
        print(f"  Dataset : {dataset}")
    print(f"  Loss    : {loss_type}   |  Mixup: {use_mixup} ({mixup_mode})  |  Sampler: {use_sampler}")
    print(f"  Device  : {device}      |  Epochs: {epochs}   |  Seed: {seed}")
    print(f"{'='*70}")

    # ── Data ─────────────────────────────────────────────────────────────────
    num_workers = getattr(args, 'num_workers', 4)
    train_loader, val_loader, test_loader, class_weights, class_counts = get_dataloaders(
        data_dir    = data_dir,
        batch_size  = batch_size,
        img_size    = IMG_SIZE,
        use_sampler = use_sampler,
        num_workers = num_workers,
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

    # ── Model / Loss / Optimizer ──────────────────────────────────────────────

    # ── SUPERPATH: SupCon (v10) ─────────────────────────────────────────────
    if loss_type == "supcon":
        return _run_supcon(args, device, name, save_dir, data_dir, batch_size)

    # ── All other versions (standard supervised learning) ─────────────────────
    model = get_resnet50_model(num_classes=NUM_CLASSES, pretrained=True).to(device)

    # v19: tempered balanced-softmax. Priority: CLI --prior_gamma > per-version
    # config "prior_gamma" > global default 1.0 (= original full Balanced Softmax).
    prior_gamma = getattr(args, "prior_gamma", None)
    if prior_gamma is None:
        prior_gamma = version_cfg.get("prior_gamma", getattr(cfg_mod, "OWMM_PRIOR_GAMMA", 1.0))

    criterion = build_loss(
        loss_type           = loss_type,
        class_weights       = class_weights,
        class_counts        = class_counts,
        device              = device,
        focal_gamma         = FOCAL_GAMMA,
        supcon_temperature  = SUPCON_TEMPERATURE,
        prior_gamma         = prior_gamma,
    )
    if loss_type == "balanced_softmax" and abs(prior_gamma - 1.0) > 1e-9:
        print(f"  Prior   : tempered balanced_softmax  |  gamma = {prior_gamma}")

    lr           = getattr(args, "lr", None) or LR
    patience     = getattr(args, "patience", None) or EARLY_STOPPING_PATIENCE
    mixup_alpha  = getattr(args, "mixup_alpha", None) or MIXUP_ALPHA
    sched_kind   = getattr(args, "scheduler", "plateau") or "plateau"
    warmup_ep    = getattr(args, "warmup_epochs", 3)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    if sched_kind == "cosine":
        # Linear warmup -> cosine anneal over the remaining epochs. Gives mixup
        # (a regularizer that needs longer, smoother training) room to pay off.
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=max(1, warmup_ep)
        )
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_ep)
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[max(1, warmup_ep)]
        )
    else:
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
        early_stopping_patience = patience,
        use_mixup               = use_mixup,
        mixup_mode              = mixup_mode,
        mixup_alpha             = mixup_alpha,
        mixup_adjacent_gap      = MIXUP_ADJACENT_GAP,
        mixup_rule_lam_max      = MIXUP_RULE_LAM_MAX,
        mixup_temperature       = cfg_mod.MIXUP_TEMPERATURE,
        mixup_queue_size        = getattr(cfg_mod, "MIXUP_QUEUE_SIZE", 64),
        mixup_queue_max_reuse   = getattr(cfg_mod, "MIXUP_QUEUE_MAX_REUSE", 3),
        dataset_class_counts    = class_counts,
        owmm_effnum_beta        = getattr(cfg_mod, "OWMM_EFFNUM_BETA", 0.9999),
        owmm_ord_sigma          = getattr(cfg_mod, "OWMM_ORD_SIGMA", 0.5),
        owmm_mix_scale_min      = getattr(cfg_mod, "OWMM_MIX_SCALE_MIN", 0.3),
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
# SupCon — 3-phase training
# ─────────────────────────────────────────────────────────────────────────────

def _run_supcon(args, device, name, save_dir, data_dir, batch_size):
    """
    Phase 1: SupCon pretrain (backbone + projection head, SupConLoss)
    Phase 2: Linear probe (freeze backbone, train classifier head)
    Phase 3: Full finetune (unfreeze, train all)
    """
    supcon_epochs   = args.supcon_epochs   or SUPCON_EPOCHS
    probe_epochs    = args.probe_epochs    or PROBE_EPOCHS
    finetune_epochs = args.finetune_epochs or FINETUNE_EPOCHS

    print(f"\n{'='*70}")
    print(f"  VERSION : {name}")
    print(f"  Phase 1  SupCon pretrain : {supcon_epochs} epochs")
    print(f"  Phase 2  Linear probe    : {probe_epochs} epochs")
    print(f"  Phase 3  Finetune        : {finetune_epochs} epochs")
    print(f"  Device   : {device}")
    print(f"{'='*70}")

    # ── Phase 1: SupCon pretraining ───────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  [Phase 1] SupCon Pretraining")
    print(f"{'─'*60}")

    supcon_model = get_resnet50_supcon_model(
        num_classes=NUM_CLASSES,
        project_dim=SUPCON_PROJECT_DIM,
        pretrained=True,
    ).to(device)

    supcon_criterion = SupConLoss(temperature=SUPCON_TEMPERATURE)

    supcon_optimizer = optim.AdamW(
        supcon_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )

    supcon_loader = get_supcon_train_loader(
        data_dir   = data_dir,
        batch_size = batch_size,
    )

    supcon_trainer = Trainer(
        model        = supcon_model,
        train_loader = None,  # not used in fit_supcon
        val_loader   = None,
        criterion    = supcon_criterion,
        optimizer    = supcon_optimizer,
        device       = device,
        version_name = f"{name}_stage1",
    )

    supcon_trainer.fit_supcon(
        epochs        = supcon_epochs,
        supcon_loader = supcon_loader,
        save_dir      = save_dir,
    )

    # ── Phase 2: Linear probe ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  [Phase 2] Linear Probe (frozen backbone)")
    print(f"{'─'*60}")

    probe_model = get_resnet50_model(num_classes=NUM_CLASSES, pretrained=False).to(device)

    # Load backbone weights from Phase 1 (skip projector, skip classifier)
    backbone_path = os.path.join(save_dir, "backbone_supcon.pth")
    if os.path.exists(backbone_path):
        state = torch.load(backbone_path, map_location=device, weights_only=True)
        probe_model.load_state_dict(state, strict=False)
        print(f"  Loaded backbone weights from {backbone_path}")

    freeze_backbone(probe_model)

    # Re-create loaders for standard training (needed before building loss)
    train_loader, val_loader, test_loader, class_weights, class_counts = get_dataloaders(
        data_dir    = data_dir,
        batch_size  = batch_size,
        img_size    = IMG_SIZE,
        use_sampler = False,
    )

    try:
        class_names = train_loader.dataset.classes
    except AttributeError:
        class_names = [f"Grade {i}" for i in range(NUM_CLASSES)]

    probe_criterion = BalancedSoftmaxLoss(class_counts=class_counts).to(device)

    # Only train the classifier head
    probe_optimizer = optim.AdamW(
        probe_model.fc.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    probe_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        probe_optimizer, mode="min", patience=3, factor=0.5
    )

    probe_trainer = Trainer(
        model                   = probe_model,
        train_loader            = train_loader,
        val_loader              = val_loader,
        criterion               = probe_criterion,
        optimizer               = probe_optimizer,
        device                  = device,
        version_name            = f"{name}_stage2",
        lr_scheduler            = probe_scheduler,
        early_stopping_patience = EARLY_STOPPING_PATIENCE,
        use_mixup               = False,
    )

    _, best_metrics = probe_trainer.fit(epochs=probe_epochs, save_dir=save_dir)

    # ── Phase 3: Full finetune ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  [Phase 3] Full Finetune")
    print(f"{'─'*60}")

    unfreeze_all(probe_model)

    finetune_criterion = BalancedSoftmaxLoss(class_counts=class_counts).to(device)

    finetune_optimizer = optim.AdamW(
        probe_model.parameters(), lr=LR * 0.5, weight_decay=WEIGHT_DECAY
    )
    finetune_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        finetune_optimizer, mode="min", patience=3, factor=0.5
    )

    finetune_trainer = Trainer(
        model                   = probe_model,
        train_loader            = train_loader,
        val_loader              = val_loader,
        criterion               = finetune_criterion,
        optimizer               = finetune_optimizer,
        device                  = device,
        version_name            = name,
        lr_scheduler            = finetune_scheduler,
        early_stopping_patience = EARLY_STOPPING_PATIENCE,
        use_mixup               = False,
    )

    _, best_metrics = finetune_trainer.fit(epochs=finetune_epochs, save_dir=save_dir)

    # ── Test-set evaluation ───────────────────────────────────────────────────
    best_model_path = os.path.join(save_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"\n  Loading best model weights for test evaluation…")
        state = torch.load(best_model_path, map_location=device, weights_only=True)
        probe_model.load_state_dict(state)
        evaluate_on_test(
            model        = probe_model,
            test_loader  = test_loader,
            criterion    = probe_criterion,
            device       = device,
            save_dir     = save_dir,
            version_name = name,
            class_names  = class_names,
        )

    return best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point (for single-version runs)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Run a single experiment version")
    parser.add_argument("--version",      type=str, default=None,
                        help="Version name (e.g. v1_baseline). Runs all if omitted.")
    parser.add_argument("--epochs",       type=int, default=None)
    parser.add_argument("--batch_size",   type=int, default=None)
    parser.add_argument("--data_dir",     type=str, default=None)
    parser.add_argument("--results_dir",  type=str, default=None)
    parser.add_argument("--num_workers",  type=int, default=4)
    parser.add_argument("--cuda",         type=int, default=None,
                        help="CUDA device ID (overrides config.CUDA_DEVICE_ID)")
    parser.add_argument("--mixup_temperature", type=float, default=None,
                        help="OWMix temperature tau (overrides config.MIXUP_TEMPERATURE)")
    parser.add_argument("--prior_gamma", type=float, default=None,
                        help="v19: tempered balanced-softmax factor gamma in [0,1] "
                             "(1=full BalSoft, 0=plain CE; overrides per-version config)")
    # Retuned-regime overrides (Plan B: give mixup room to pay off on large data)
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides config.LR)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early-stopping patience (overrides config.EARLY_STOPPING_PATIENCE)")
    parser.add_argument("--mixup_alpha", type=float, default=None,
                        help="Mixup Beta alpha (overrides config.MIXUP_ALPHA)")
    parser.add_argument("--scheduler", type=str, default="plateau",
                        choices=["plateau", "cosine"],
                        help="LR schedule: 'plateau' (ReduceLROnPlateau, default) "
                             "or 'cosine' (linear warmup + cosine anneal)")
    parser.add_argument("--warmup_epochs", type=int, default=3,
                        help="Warmup epochs for --scheduler cosine (default: 3)")
    # SupCon-specific
    parser.add_argument("--supcon_epochs",   type=int, default=None,
                        help="SupCon pretraining epochs (v10)")
    parser.add_argument("--probe_epochs",    type=int, default=None,
                        help="Linear probe epochs (v10 phase 2)")
    parser.add_argument("--finetune_epochs", type=int, default=None,
                        help="Full finetune epochs (v10 phase 3)")
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset subdirectory under results_dir (e.g. koa, eyepacs)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.version:
        cfg = next((v for v in VERSIONS if v["name"] == args.version), None)
        if cfg is None:
            print(f"[ERROR] Unknown version '{args.version}'. "
                  f"Available: {[v['name'] for v in VERSIONS]}")
            sys.exit(1)
        run_version(cfg, args, seed=args.seed)
    else:
        print("No --version specified. Use run_all.py to run all versions.")
