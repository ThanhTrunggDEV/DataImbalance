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
MIXUP_QUEUE_MAX_REUSE = 3      # OWMix+Queue v2 (v14): max times a cached sample may be drawn as a partner before eviction
OWMM_EFFNUM_BETA = 0.9999     # OWMM (v17): effective-number reweight beta (Cui et al. 2019) for partner selection
OWMM_ORD_SIGMA = 0.5          # OWMM (v17): Gaussian width of soft ordinal target smoothing (0 = plain two-hot mixup)
OWMM_MIX_SCALE_MIN = 0.3      # OWMM-Adaptive (v18): min per-sample mixing scale for the majority class (rare classes -> 1.0)
OWMM_PRIOR_GAMMA = 1.0        # v19 default: tempering factor on the BalancedSoftmax log-prior (1=full BalSoft, 0=plain CE); overridden per-version / via --prior_gamma

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
    # ── OWMix + Bounded/Re-augmented/Rescue-only Queue (v14) ─────────────────
    # Fixes v13's regression (worse than v1_baseline on both datasets): the
    # v13 queue caches fully-augmented tensors forever and reuses them
    # unboundedly as mixup partners, which lets the model latch onto a
    # handful of frozen minority-class residuals as a shortcut instead of
    # learning real features. v14 bounds reuse per cached sample,
    # re-augments on every draw, and only pulls from the queue for anchors
    # genuinely under-represented in the current batch. See
    # training/trainer.py::mixup_data_ordinal_weighted_queue_v2.
    {   # τ = 1.0 (default)
        "name": "v14_owmixup_queue_v2_ce",
        "display": "OWMixup+QueueV2 (CE) τ=1.0",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted_queue_v2",
        "loss_type": "cross_entropy",
        "use_sampler": True,
        "mixup_temperature": 1.0,
    },
    {
        "name": "v14_owmixup_queue_v2_ce_t20",
        "display": "OWMixup+QueueV2 (CE) τ=2.0",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted_queue_v2",
        "loss_type": "cross_entropy",
        "use_sampler": True,
        "mixup_temperature": 2.0,
    },
    {
        "name": "v14_owmixup_queue_v2_balanced",
        "display": "OWMixup+QueueV2 + BalSoft τ=1.0",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted_queue_v2",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
        "mixup_temperature": 1.0,
    },
    {
        "name": "v14_owmixup_queue_v2_balanced_t20",
        "display": "OWMixup+QueueV2 + BalSoft τ=2.0",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted_queue_v2",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
        "mixup_temperature": 2.0,
    },

    # v15: same bounded-reuse / re-augment-on-draw queue as v14, but the
    # rescue-only gate now compares the batch's per-class count against that
    # class's TRUE expected count under the dataset's actual training
    # proportions (recovered from inv_freq), instead of an assumed-uniform
    # 1/num_classes split. v14 beat baseline on KOA (mild imbalance, the two
    # thresholds are close) but underperformed it on EyePACS (~35x imbalance
    # ratio), where the uniform bar made rescue fire on almost every batch and
    # eroded majority-class recall. See
    # training/trainer.py::mixup_data_ordinal_weighted_queue_v3.
    {   # τ = 1.0 (default)
        "name": "v15_owmixup_queue_v3_ce",
        "display": "OWMixup+QueueV3 (CE) τ=1.0",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted_queue_v3",
        "loss_type": "cross_entropy",
        "use_sampler": True,
        "mixup_temperature": 1.0,
    },
    {
        "name": "v15_owmixup_queue_v3_ce_t20",
        "display": "OWMixup+QueueV3 (CE) τ=2.0",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted_queue_v3",
        "loss_type": "cross_entropy",
        "use_sampler": True,
        "mixup_temperature": 2.0,
    },
    {
        "name": "v15_owmixup_queue_v3_balanced",
        "display": "OWMixup+QueueV3 + BalSoft τ=1.0",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted_queue_v3",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
        "mixup_temperature": 1.0,
    },
    {
        "name": "v15_owmixup_queue_v3_balanced_t20",
        "display": "OWMixup+QueueV3 + BalSoft τ=2.0",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted_queue_v3",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
        "mixup_temperature": 2.0,
    },

    # v16: OWMix queue with FREQUENCY-MODULATED MIXING. v13-v15 all lost to the
    # CE baseline on EyePACS because they mixed majority-class (DR0) anchors at
    # full strength, dropping DR0 F1 0.900 -> ~0.84 and cancelling every
    # minority gain in the equally-weighted macro F1. v16 modulates each
    # anchor's mixing INTENSITY continuously by its class rarity (s =
    # inv_freq[y]/max(inv_freq), lam_eff = 1 - s*(1-lam)) -- no hard threshold,
    # so it stays faithful to OWMixup's soft, principled design. Majority
    # anchors mix weakly (clean signal preserved), minority mix fully.
    # Self-calibrates from inv_freq. See
    # training/trainer.py::mixup_data_ordinal_weighted_queue_v4.
    {   # τ = 1.0 (default)
        "name": "v16_owmixup_queue_v4_ce",
        "display": "OWMixup+QueueV4 (CE) τ=1.0",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted_queue_v4",
        "loss_type": "cross_entropy",
        "use_sampler": True,
        "mixup_temperature": 1.0,
    },
    {
        "name": "v16_owmixup_queue_v4_ce_t20",
        "display": "OWMixup+QueueV4 (CE) τ=2.0",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted_queue_v4",
        "loss_type": "cross_entropy",
        "use_sampler": True,
        "mixup_temperature": 2.0,
    },
    {
        "name": "v16_owmixup_queue_v4_balanced",
        "display": "OWMixup+QueueV4 + BalSoft τ=1.0",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted_queue_v4",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
        "mixup_temperature": 1.0,
    },
    {
        "name": "v16_owmixup_queue_v4_balanced_t20",
        "display": "OWMixup+QueueV4 + BalSoft τ=2.0",
        "use_mixup": True,
        "mixup_mode": "ordinal_weighted_queue_v4",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
        "mixup_temperature": 2.0,
    },

    # ── OWMM: Ordinal-Weighted Manifold Mixup (v17) ─────────────────────────
    # v11-v16 mix in PIXEL space (alpha-blend of two fundus images is
    # anatomically meaningless), so the only real effect is a zero-sum-with-the
    # -sampler resampling and every variant merely shifted F1 between classes.
    # OWMM keeps OWMix's ordinal-distance x tail-frequency partner selection but
    # (1) interpolates on the layer3 feature manifold, (2) uses effective-number
    # tail reweighting (strong enough for EyePACS's ~35x imbalance), and
    # (3) supervises with a soft target smoothed along the ordinal grade axis.
    # Sampler is OFF for both variants so partner reweighting is the sole
    # balancer (avoids the double-correction that sank majority F1). See
    # training/trainer.py::owmm_* functions.
    {
        "name": "v17_owmm_balanced",
        "display": "OWMM + BalSoft",
        "use_mixup": True,
        "mixup_mode": "ordinal_manifold",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
        "mixup_temperature": 1.0,
    },
    {
        "name": "v17_owmm_ce",
        "display": "OWMM (CE)",
        "use_mixup": True,
        "mixup_mode": "ordinal_manifold",
        "loss_type": "cross_entropy",
        "use_sampler": False,
        "mixup_temperature": 1.0,
    },
    # ── OWMM-Adaptive: scarcity-modulated mixing strength (v18) ──────────────
    # Same manifold + ordinal-partner design as v17, but the per-sample mixing
    # strength is scaled by the anchor class's scarcity (effective-number based):
    # the majority class is mixed only lightly (scale -> OWMM_MIX_SCALE_MIN) to
    # protect its accuracy on heavily imbalanced data (EyePACS ~36x), while rare
    # classes keep full mixing. The scale spread widens with imbalance, so on the
    # milder KOA (~13x) it degrades gracefully toward plain OWMM. Goal: a single
    # variant that beats baseline on BOTH the small/mild and large/severe sets.
    {
        "name": "v18_owmm_adaptive_balanced",
        "display": "OWMM-Adaptive + BalSoft",
        "use_mixup": True,
        "mixup_mode": "ordinal_manifold_adaptive",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
    },
    {
        "name": "v18_owmm_adaptive_ce",
        "display": "OWMM-Adaptive (CE)",
        "use_mixup": True,
        "mixup_mode": "ordinal_manifold_adaptive",
        "loss_type": "cross_entropy",
        "use_sampler": False,
    },
    # ── OWMM-Adaptive + tempered prior (v19) ────────────────────────────────
    # v18 showed adaptive mixing works, but the loss correction was binary:
    # CE (gamma=0) wins EyePACS but drops KOA's grade-1; full BalSoft (gamma=1)
    # wins KOA but catastrophically over-corrects on EyePACS's 36x imbalance
    # (majority abandoned, Acc collapses). v19 makes the correction continuous
    # via a tempered log-prior gamma, sweeping the middle to find a single
    # setting that clears baseline on BOTH datasets. Same adaptive manifold
    # mixing and sampler-off invariant as v18.
    {
        "name": "v19_owmm_tempered_g025",
        "display": "OWMM-Adaptive + Tempered γ=0.25",
        "use_mixup": True,
        "mixup_mode": "ordinal_manifold_adaptive",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
        "prior_gamma": 0.25,
    },
    {
        "name": "v19_owmm_tempered_g05",
        "display": "OWMM-Adaptive + Tempered γ=0.5",
        "use_mixup": True,
        "mixup_mode": "ordinal_manifold_adaptive",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
        "prior_gamma": 0.5,
    },
    {
        "name": "v19_owmm_tempered_g075",
        "display": "OWMM-Adaptive + Tempered γ=0.75",
        "use_mixup": True,
        "mixup_mode": "ordinal_manifold_adaptive",
        "loss_type": "balanced_softmax",
        "use_sampler": False,
        "prior_gamma": 0.75,
    },
]

# Curated subset reported in the paper: the classic imbalance baselines
# (Baseline, Balanced Softmax, Focal, Adjacent+BalSoft) plus the OWMM-Adaptive
# method with its tempered-prior gamma sweep — the two v18 endpoints (gamma=0
# ~= CE, gamma=1 ~= full Balanced Softmax) and the v19 middle (gamma in
# {0.25, 0.5, 0.75}). The gamma is selected per dataset on validation QWK (see
# select_gamma.py). Use via `run_all.py --paper` to avoid running every version.
PAPER_VERSION_NAMES = [
    "v1_baseline",
    "v3_balanced_softmax",
    "v5_focal_loss",
    "v7_adjacent_balanced",
    "v18_owmm_adaptive_ce",        # gamma = 0 endpoint
    "v18_owmm_adaptive_balanced",  # gamma = 1 endpoint
] + [v["name"] for v in VERSIONS if v["name"].startswith("v19_owmm_tempered")]
