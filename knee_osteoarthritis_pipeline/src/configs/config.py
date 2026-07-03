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
SUPCON_IMG_SIZE = 256         # Larger crop size for SupCon augmentations

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
MIXUP_ADJACENT_GAP = 1        # Max class gap for adjacent mixup
MIXUP_RULE_LAM_MAX = 0.49     # Max mixing coeff for Grade 0 in rule-based mixup
MIXUP_TEMPERATURE = 1.0       # OWMix temperature (Gaussian kernel width)
MIXUP_QUEUE_SIZE = 64          # OWMix+Queue (v13): per-class CPU memory queue capacity

# ─────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────
FOCAL_GAMMA = 2.0

# ─────────────────────────────────────────
# SupCon (Supervised Contrastive Learning)
# ─────────────────────────────────────────
SUPCON_TEMPERATURE = 0.1      # Temperature scaling in SupCon loss
SUPCON_PROJECT_DIM = 128      # Projection head output dimension
SUPCON_EPOCHS = 50            # Stage 1: SupCon pretraining epochs
PROBE_EPOCHS = 10             # Stage 2: Linear probe epochs
FINETUNE_EPOCHS = 20          # Stage 3: Full finetune epochs

# ─────────────────────────────────────────
# Device
# ─────────────────────────────────────────
CUDA_DEVICE_ID = 0       # GPU index, override via --cuda CLI flag

def get_device():
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{CUDA_DEVICE_ID}")
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
    # ── Adjacent Mixup variants ────────────────────────────────────────────────
    {
        "name": "v6_adjacent_ce",
        "display": "Adjacent Mixup (CE)",
        "use_mixup": True,
        "mixup_mode": "adjacent",
        "loss_type": "cross_entropy",
        "use_sampler": True,
    },
    {
        "name": "v7_adjacent_balanced",
        "display": "Adjacent Mixup + Balanced Softmax",
        "use_mixup": True,
        "mixup_mode": "adjacent",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
    },
    # ── Rule-based Mixup variants ──────────────────────────────────────────────
    {
        "name": "v8_rule_ce",
        "display": "Rule Mixup (CE)",
        "use_mixup": True,
        "mixup_mode": "rule",
        "loss_type": "cross_entropy",
        "use_sampler": True,
    },
    {
        "name": "v9_rule_balanced",
        "display": "Rule Mixup + Balanced Softmax",
        "use_mixup": True,
        "mixup_mode": "rule",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
    },
    # ── Representation Learning ────────────────────────────────────────────────
    {
        "name": "v10_supcon",
        "display": "SupCon (Sup. Contrastive)",
        "use_mixup": False,
        "mixup_mode": None,
        "loss_type": "supcon",
        "use_sampler": False,
    },
    # ── OWMix (Ordinal-Weighted Mixup) ───────────────────────────────────────────
    {   # τ = 1.0 (default)
        "name": "v11_owmixup_ce",
        "display": "OWMixup (CE) τ=1.0",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted",
        "loss_type": "cross_entropy",
        "use_sampler": True,
        "mixup_temperature": 1.0,
    },
    {
        "name": "v11_owmixup_ce_t05",
        "display": "OWMixup (CE) τ=0.5",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted",
        "loss_type": "cross_entropy",
        "use_sampler": True,
        "mixup_temperature": 0.5,
    },
    {
        "name": "v11_owmixup_ce_t15",
        "display": "OWMixup (CE) τ=1.5",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted",
        "loss_type": "cross_entropy",
        "use_sampler": True,
        "mixup_temperature": 1.5,
    },
    {
        "name": "v11_owmixup_ce_t20",
        "display": "OWMixup (CE) τ=2.0",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted",
        "loss_type": "cross_entropy",
        "use_sampler": True,
        "mixup_temperature": 2.0,
    },
    {
        "name": "v12_owmixup_balanced",
        "display": "OWMixup + BalSoft τ=1.0",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
        "mixup_temperature": 1.0,
    },
    {
        "name": "v12_owmixup_balanced_t05",
        "display": "OWMixup + BalSoft τ=0.5",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
        "mixup_temperature": 0.5,
    },
    {
        "name": "v12_owmixup_balanced_t15",
        "display": "OWMixup + BalSoft τ=1.5",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
        "mixup_temperature": 1.5,
    },
    {
        "name": "v12_owmixup_balanced_t20",
        "display": "OWMixup + BalSoft τ=2.0",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
        "mixup_temperature": 2.0,
    },
    # ── OWMix + Memory Queue (v13) ───────────────────────────────────────────
    {   # τ = 1.0 (default)
        "name": "v13_owmixup_queue_ce",
        "display": "OWMixup+Queue (CE) τ=1.0",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted_queue",
        "loss_type": "cross_entropy",
        "use_sampler": True,
        "mixup_temperature": 1.0,
    },
    {
        "name": "v13_owmixup_queue_ce_t05",
        "display": "OWMixup+Queue (CE) τ=0.5",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted_queue",
        "loss_type": "cross_entropy",
        "use_sampler": True,
        "mixup_temperature": 0.5,
    },
    {
        "name": "v13_owmixup_queue_ce_t15",
        "display": "OWMixup+Queue (CE) τ=1.5",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted_queue",
        "loss_type": "cross_entropy",
        "use_sampler": True,
        "mixup_temperature": 1.5,
    },
    {
        "name": "v13_owmixup_queue_ce_t20",
        "display": "OWMixup+Queue (CE) τ=2.0",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted_queue",
        "loss_type": "cross_entropy",
        "use_sampler": True,
        "mixup_temperature": 2.0,
    },
    {
        "name": "v13_owmixup_queue_balanced",
        "display": "OWMixup+Queue + BalSoft τ=1.0",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted_queue",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
        "mixup_temperature": 1.0,
    },
    {
        "name": "v13_owmixup_queue_balanced_t05",
        "display": "OWMixup+Queue + BalSoft τ=0.5",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted_queue",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
        "mixup_temperature": 0.5,
    },
    {
        "name": "v13_owmixup_queue_balanced_t15",
        "display": "OWMixup+Queue + BalSoft τ=1.5",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted_queue",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
        "mixup_temperature": 1.5,
    },
    {
        "name": "v13_owmixup_queue_balanced_t20",
        "display": "OWMixup+Queue + BalSoft τ=2.0",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted_queue",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
        "mixup_temperature": 2.0,
    },
]
