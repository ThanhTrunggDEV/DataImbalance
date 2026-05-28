"""
Shared configuration for all experiment versions.
Edit this file to change hyperparameters globally.
"""

import torch

# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────
DATA_DIR = "../data"          # relative to src/
NUM_CLASSES = 5               # Knee OA severity: 0–4
IMG_SIZE = 224                # ResNet50 standard input

# ─────────────────────────────────────────
# Training
# ─────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 7

# ─────────────────────────────────────────
# Mixup
# ─────────────────────────────────────────
MIXUP_ALPHA = 0.4             # Beta distribution parameter

# ─────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────
FOCAL_GAMMA = 2.0

# ─────────────────────────────────────────
# Device
# ─────────────────────────────────────────
def get_device():
    if torch.cuda.device_count() > 1:
        device = torch.device("cuda:1")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"[Device] Using: {device}  "
          f"(CUDA available: {torch.cuda.is_available()}, "
          f"GPU count: {torch.cuda.device_count()})")
    return device

# ─────────────────────────────────────────
# Experiment versions definition
# Each dict key maps to a run_experiment flag
# ─────────────────────────────────────────
VERSIONS = [
    {
        "name": "v1_baseline",
        "display": "Baseline (CE)",
        "use_mixup": False,
        "loss_type": "cross_entropy",      # vanilla CE, no weighting
        "use_sampler": False,              # true baseline: no resampling
    },
    {
        "name": "v2_mixup",
        "display": "Mixup",
        "use_mixup": True,
        "loss_type": "cross_entropy",
        "use_sampler": True,
    },
    {
        "name": "v3_balanced_softmax",
        "display": "Balanced Softmax",
        "use_mixup": False,
        "loss_type": "balanced_softmax",
        "use_sampler": False,  # BalancedSoftmaxLoss handles imbalance via log-prior;
                               # combining with WeightedRandomSampler double-corrects
    },
    {
        "name": "v4_mixup_balanced_softmax",
        "display": "Mixup + Balanced Softmax",
        "use_mixup": True,
        "loss_type": "balanced_softmax",
        "use_sampler": False,  # same reason as v3
    },
    {
        "name": "v5_focal_loss",
        "display": "Focal Loss",
        "use_mixup": False,
        "loss_type": "focal",
        "use_sampler": True,
    },
]
