# Knee Osteoarthritis — Multi-Version Imbalance Experiment Pipeline

Automated framework for systematic evaluation of **class-imbalance mitigation strategies** on Knee Osteoarthritis severity classification (5-class: Grade 0–4) from X-ray images.

The pipeline trains **45 experiment variants** (across 19 version families) with different imbalance techniques on the same ResNet50 backbone, then generates consolidated comparison reports (CSV, bar charts, F1 heatmaps) and an HTML weekly report for rigorous side-by-side analysis.

---

## Experiment Versions

| Version | Technique | Loss Function | Mixup Mode | Sampler |
|---------|-----------|---------------|------------|---------|
| `v1_baseline` | None (pure baseline) | CrossEntropy | — | No |
| `v2_mixup` | Standard Mixup | CrossEntropy | standard | WeightedRandomSampler |
| `v3_balanced_softmax` | Logit-margin shift | BalancedSoftmaxLoss | — | No |
| `v4_mixup_balanced_softmax` | Mixup + logit-margin | BalancedSoftmaxLoss | standard | No |
| `v5_focal_loss` | Hard-example mining | FocalLoss (γ=2.0) | — | WeightedRandomSampler |
| `v6_adjacent_ce` | **Adjacent Mixup** | CrossEntropy | adjacent | WeightedRandomSampler |
| `v7_adjacent_balanced` | Adjacent Mixup | BalancedSoftmaxLoss | adjacent | No |
| `v8_rule_ce` | **Rule-based Mixup** | CrossEntropy | rule | WeightedRandomSampler |
| `v9_rule_balanced` | Rule-based Mixup | BalancedSoftmaxLoss | rule | No |
| `v10_supcon` | **SupCon** (3-phase) | SupConLoss → CE | — | No |
| `v11_owmixup_ce` | **OWMixup** (CE) τ=t | CrossEntropy | ordinal_weighted | WeightedRandomSampler |
| `v12_owmixup_balanced` | **OWMixup** + BalSoft τ=t | BalancedSoftmaxLoss | ordinal_weighted | No |
| `v13_owmixup_queue_*` | **OWMixup + Queue** τ=t | CE / BalSoft | ordinal_weighted_queue | Yes / No |
| `v14_owmixup_queue_v2_*` | **QueueV2** (bounded reuse) τ=t | CE / BalSoft | ordinal_weighted_queue_v2 | Yes / No |
| `v15_owmixup_queue_v3_*` | **QueueV3** (true-freq gate) τ=t | CE / BalSoft | ordinal_weighted_queue_v3 | Yes / No |
| `v16_owmixup_queue_v4_*` | **QueueV4** (freq-modulated) τ=t | CE / BalSoft | ordinal_weighted_queue_v4 | Yes / No |
| `v17_owmm_*` | **OWMM** (manifold mixup) | CE / BalSoft | ordinal_manifold | No |
| `v18_owmm_adaptive_*` | **OWMM-Adaptive** | CE / BalSoft | ordinal_manifold_adaptive | No |
| `v19_owmm_tempered_g*` | **OWMM-Adaptive** + tempered γ | BalSoft (γ-tempered) | ordinal_manifold_adaptive | No |

> **Adjacent Mixup** (`v6`, `v7`): Only pairs images whose class labels differ by at most 1 grade (e.g., Grade 0↔1, 1↔2, 2↔3, 3↔4). Samples that cannot find a valid partner are left unmixed.  
> **Rule-based Mixup** (`v8`, `v9`): Selects Grade 0 images as anchors and pairs them with Grade {2,3,4}. The mixing coefficient `lam` is clamped to <0.5 so the non-zero partner always contributes the majority, and the label is assigned to that partner. Batches without Grade 0 skip mixup.  
> **SupCon** (`v10`): Three-phase training — (1) Supervised contrastive pretraining of backbone + projection head with 2-augment views, (2) linear probe with frozen backbone, (3) full finetune of all parameters.  
> **OWMix** (`v11`, `v12`): Ordinal-Weighted Mixup — assigns mixing coefficients via a Gaussian kernel over the ordinal label distance (`τ` controls kernel width). Each variant runs 4 temperature settings (`τ ∈ {0.5, 1.0, 1.5, 2.0}`) for a total of 8 OWMix sub-variants.
> 
> **OWMix + Queue** (`v13`, 8 sub-variants): Adds a per-class CPU memory queue that caches augmented tensors from past batches to serve as mixup partners for under-represented classes. When the current batch lacks enough minority samples, the queue supplies cached candidates. Despite the intent, v13 **regressed below baseline** on both datasets because it caches fully-augmented tensors forever and reuses them unboundedly — the model learns to exploit frozen minority residuals as a shortcut rather than learning real features.
> 
> **QueueV2** (`v14`): Fixes v13's shortcut problem. Mixup partners drawn from the queue are (1) **re-augmented** on every draw so they are never identical, (2) evicted after `MIXUP_QUEUE_MAX_REUSE` (default 3) uses, and (3) the queue is only queried when an anchor's class is genuinely **under-represented** in the mini-batch (a rescue-only gate). τ = {1.0, 2.0} × {CE, BalSoft} = 4 sub-variants.
> 
> **QueueV3** (`v15`): Same bounded-reuse / re-augment-on-draw as v14, but the rescue-only gate now checks each batch class count against the **true expected count** from the dataset's actual class proportions (via `inv_freq`) instead of a uniform `1/num_classes` threshold. v14's uniform gate fired rescue on nearly every EyePACS batch (35× imbalance, so minority prevalence is tiny) and eroded majority-class recall. v15 self-calibrates per-class so rescue only fires where genuinely needed. τ = {1.0, 2.0} × {CE, BalSoft} = 4 sub-variants.
> 
> **QueueV4 — Frequency-Modulated Mixing** (`v16`): Abandons the hard rescue threshold entirely. The per-anchor mixing intensity is **continuously modulated** by the anchor class's rarity: `s = inv_freq[y] / max(inv_freq)`, `lam_eff = 1 - s * (1 - lam)`. Majority anchors mix weakly (clean signal preserved, preventing F1 drop on DR0), minority anchors mix fully. Self-calibrates from inverse frequencies — no hyperparameters, pure soft design faithful to OWMixup's principled approach. τ = {1.0, 2.0} × {CE, BalSoft} = 4 sub-variants.
> 
> **OWMM — Ordinal-Weighted Manifold Mixup** (`v17`, 2 sub-variants): A paradigm shift from pixel-space mixing (which produces anatomically meaningless blended X-rays) to **feature-manifold interpolation** at layer3 of the ResNet50. Keeps OWMix's ordinal-distance × tail-frequency partner selection, but (1) interpolates on the feature manifold, (2) uses **effective-number** reweighting (Cui et al. 2019) strong enough for EyePACS's ~35× imbalance, and (3) supervises with **soft targets smoothed along the ordinal grade axis** (two-hot with Gaussian width `OWMM_ORD_SIGMA`). Sampler is OFF for both CE and BalSoft variants so that partner reweighting is the sole balancer.
> 
> **OWMM-Adaptive** (`v18`, 2 sub-variants): Same manifold + ordinal-partner design as v17, but the per-sample mixing **strength** is scaled by the anchor class's scarcity (effective-number based). On heavily imbalanced data (EyePACS, ~36×), the majority class is mixed only lightly (`OWMM_MIX_SCALE_MIN = 0.3`) to protect its accuracy, while rare classes keep full mixing. On milder imbalance (KOA, ~13×), the scale spread narrows and degrades gracefully toward plain OWMM. Goal: a single variant that beats baseline on **both** datasets.
> 
> **OWMM-Adaptive + Tempered Prior** (`v19`, 3 sub-variants): v18 showed that CE (γ=0) wins EyePACS but drops KOA's Grade 1, while full BalSoft (γ=1) wins KOA but catastrophically over-corrects on EyePACS (majority abandoned, accuracy collapses). v19 makes the loss correction **continuous** by tempering the BalancedSoftmax log-prior with `γ ∈ {0.25, 0.5, 0.75}` — sweeping the middle ground to find a single γ that clears baseline on **both** datasets simultaneously. Same adaptive manifold mixing and sampler-off invariant as v18.

> **Design note:** BalancedSoftmaxLoss (Ren et al., NeurIPS 2020) inherently corrects for class imbalance via log-frequency prior — combining it with WeightedRandomSampler would double-correct, so variants with BalancedSoftmax intentionally disable the sampler.

---

## Project Structure

```
knee_osteoarthritis_pipeline/
├── README.md
├── AGENTS.md                      # 🤖 OpenCode instructions
├── requirements.txt
├── weekly_report.html             # 📊 Auto-generated HTML weekly report
├── data/                          # Knee OA dataset (gitignored)
│   └── train/ val/ test/          # Each with sub-folders: 0/ 1/ 2/ 3/ 4/
├── data_dr/                       # EyePACS DR dataset (gitignored)
│   └── train/ val/ test/          # Same folder structure, 5 ordinal classes
├── outputs/                       # EDA & misc outputs (not tracked)
│   └── eda/
│       ├── summary.csv
│       ├── plots/
│       └── galleries/
├── results/                       # Experiment outputs (not tracked)
│   ├── v1_baseline/ ... v12_owmixup_balanced_t20/
│   │   ├── best_model.pth
│   │   ├── metrics.json
│   │   ├── training_history.png
│   │   ├── confusion_matrix.png
│   │   ├── test_metrics.json
│   │   └── test_confusion_matrix.png
│   └── comparison/
│       ├── metrics_comparison.png
│       ├── loss_f1_curves_comparison.png
│       ├── per_class_f1_heatmap.png
│       └── all_versions_summary.csv
└── src/
    ├── run_all.py                 # 🚀 Master orchestrator (entry point)
    ├── run_experiment.py          # Single-version runner + seed control
    ├── run_owmix_sweep.py         # 🌡️ OWMix temperature sweep (v11-v12)
    ├── run_visualization.py       # 🎨 Grad-CAM & t-SNE visualizations
    ├── generate_report.py         # 📋 HTML weekly report generator
    ├── configs/
    │   └── config.py              # Hyperparameters, VERSIONS, device
    ├── data/
    │   ├── dataset.py             # KneeDataset, DataLoader, SupConViewDataset
    │   ├── download.py            # Kaggle dataset downloader (KOA)
    │   └── download_eyepacs.py    # HuggingFace downloader (EyePACS DR)
    ├── eda/
    │   └── eda.py                 # Exploratory Data Analysis (distributions, pixel stats, galleries)
    ├── models/
    │   ├── resnet.py              # ResNet50, SupCon model, freeze helpers
    │   └── efficientnet.py        # EfficientNet-B5 (alternative backbone)
    ├── training/
    │   ├── loss.py                # FocalLoss, BalancedSoftmaxLoss, SupConLoss, factory
    │   ├── metrics.py             # calculate_metrics (F1, AUC, confusion matrix, QWK, MAE)
    │   ├── trainer.py             # Trainer with Mixup variants, early stopping, SupCon
    │   ├── visualization.py       # Per-epoch training history plots
    │   └── recompute_test_metrics.py  # 🔧 One-off QWK/MAE recomputation utility
    ├── evaluation/
    │   ├── evaluator.py           # Test-set evaluation, classification report
    │   └── visualization.py       # Cross-version comparison plots, F1 heatmap
    └── visualization/
        ├── gradcam.py             # 🔥 Grad-CAM heatmaps (hooks into ResNet50 layer4)
        └── tsne.py                # 🔬 t-SNE feature embeddings from avgpool
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ThanhTrunggDEV/DataImbalance.git
cd DataImbalance/knee_osteoarthritis_pipeline

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.8
- PyTorch + torchvision (with CUDA for GPU training)
- scikit-learn, numpy, matplotlib, seaborn, tqdm, Pillow
- `datasets` (HuggingFace) — required for EyePACS download

> **Note:** To auto-download the Kaggle dataset via `data/download.py`, place a valid `kaggle.json` in `~/.kaggle/`.
> **Note:** For EyePACS, run `pip install datasets` then `python data/download_eyepacs.py`.

---

## Dataset Preparation

### Knee OA (X-ray)

The pipeline expects this folder structure under `data/`:

```
data/
├── train/
│   ├── 0/   (Normal)
│   ├── 1/   (Doubtful)
│   ├── 2/   (Mild)
│   ├── 3/   (Moderate)
│   └── 4/   (Severe)
├── val/
│   └── 0/ 1/ 2/ 3/ 4/
└── test/
    └── 0/ 1/ 2/ 3/ 4/
```

Download the dataset from Kaggle:
```bash
cd src
python -c "from data.download import download_kaggle_dataset; download_kaggle_dataset()"
```

### EyePACS Diabetic Retinopathy (Retinal Fundus)

Cross-dataset validation dataset (same 5-class ordinal structure, different imaging modality).

Auto-download from HuggingFace and preprocess into the same folder structure (`data_dr/{train,val,test}/{0-4}/`):

```bash
cd src
pip install datasets
python data/download_eyepacs.py
```

This produces a stratified **70/10/20** train/val/test split of the 35,108-image EyePACS training set, resized to 224×224.

---

## Usage

### Run All Versions (Full Pipeline)

```bash
cd src
python run_all.py                                    # Full run (30 epochs each)
python run_all.py --epochs 2                         # Quick smoke test
python run_all.py --only v1_baseline v10_supcon      # Run specific versions
python run_all.py --skip v2_mixup                    # Skip a version
python run_all.py --cuda 1                           # Use a specific GPU
python run_all.py --seeds 42 123 456                 # Multi-seed (3 runs per version)
python run_all.py --num_workers 0                    # Win: disable multiprocessing
python run_all.py --dataset koa                      # Save under results/koa/
python run_all.py --dataset eyepacs                  # Save under results/eyepacs/
python run_all.py --paper                            # Run paper's 6-method + v13 sweep subset
```

### Cross-Dataset (EyePACS)

```bash
cd src
python run_all.py --data_dir ../data_dr --dataset eyepacs --num_workers 4   # Linux server full run
python run_all.py --data_dir ../data_dr --dataset eyepacs --num_workers 0   # Windows
python run_all.py --data_dir ../data_dr --dataset eyepacs --num_workers 4 --seeds 42 123 \
  --only v1_baseline v3_balanced_softmax v5_focal_loss v7_adjacent_balanced \
         v11_owmixup_ce v12_owmixup_balanced_t20          # Top 6 × 2 seeds → results/eyepacs/
```

### Run a Single Version

```bash
cd src
python run_experiment.py --version v1_baseline --epochs 10
python run_experiment.py --version v5_focal_loss --batch_size 16
python run_experiment.py --version v10_supcon --supcon_epochs 50 --probe_epochs 10
python run_experiment.py --version v7_adjacent_balanced --data_dir ../data_dr \
  --num_workers 4                                        # Single EyePACS experiment
```

### OWMix Temperature Sweep (v11, v12)

```bash
cd src
python run_owmix_sweep.py                            # All 8 OWMix variants (30 epochs)
python run_owmix_sweep.py --epochs 50 --cuda 1       # Custom epochs + GPU
```

### Visualization (Grad-CAM & t-SNE)

```bash
cd src
python run_visualization.py                          # Compares v1, v2, v7
```

Loads trained models and produces Grad-CAM heatmaps (per-class) and t-SNE embedding plots, saved to `report_assets/vis/`.

### Generate HTML Weekly Report

```bash
cd src
python generate_report.py                            # Default: ../results → ../weekly_report.html
python generate_report.py --outputdir ../reports     # Custom output directory
python generate_report.py --output index.html        # Custom filename
```

Generates a self-contained Vietnamese-language HTML report with executive summary, dataset statistics, configuration overview, ranking tables, comparison charts, per-class analysis, per-version details with confusion matrices, and recommendations.

### Exploratory Data Analysis

```bash
cd src
python -m eda.eda
python -m eda.eda --data-root ../data --out ../outputs/eda --samples-per-class 6
```

Produces class distribution charts, pixel statistics, sample image galleries, and summary CSVs.

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--epochs` | Override epoch count | 30 |
| `--batch_size` | Override batch size | 32 |
| `--cuda` | GPU device ID | 0 |
| `--data_dir` | Path to dataset root | `../data` |
| `--results_dir` | Output directory | `../results` |
| `--num_workers` | DataLoader workers (0 for Windows) | 4 |
| `--dataset` | Dataset subdirectory (`koa`, `eyepacs`, or empty) | `""` |
| `--seeds` | Random seeds for multi-seed runs (run_all) | `[42]` |
| `--skip` | Version names to skip (run_all only) | — |
| `--only` | Run only these versions (run_all only) | — |
| `--paper` | Run paper-curated subset only (run_all) | — |
| `--mixup_temperature` | OWMix temperature τ | config default |
| `--prior_gamma` | Tempered BalSoft log-prior γ (v19) | config default |
| `--supcon_epochs` | SupCon pretraining epochs (v10) | 50 |
| `--probe_epochs` | Linear probe epochs (v10 phase 2) | 10 |
| `--finetune_epochs` | Full finetune epochs (v10 phase 3) | 20 |

---

## Output Artifacts

After a full pipeline run, you'll find:

### Per-Version (`results/<version>/`)
- `best_model.pth` — Best model weights (by val F1-macro)
- `metrics.json` — Full training history + best epoch snapshot
- `training_history.png` — Loss, accuracy, precision, recall/F1 curves
- `confusion_matrix.png` — Best validation confusion matrix
- `test_metrics.json` — Detailed test-set metrics with per-class breakdown
- `test_confusion_matrix.png` — Test confusion matrix (counts + normalized)
- `test_classification_report.txt` — sklearn classification report
- SupCon (v10) additionally saves:
  - `backbone_supcon.pth` — Backbone weights after contrastive pretraining
  - `supcon_metrics.json` — SupCon loss history

### Comparison (`results/comparison/`)
- `all_versions_summary.csv` — All versions ranked by F1-macro
- `metrics_comparison.png` — Grouped bar chart of all metrics
- `loss_f1_curves_comparison.png` — Overlaid validation curves
- `per_class_f1_heatmap.png` — Per-class F1 heatmap across all versions

### HTML Weekly Report (`weekly_report.html`)
Generated by `generate_report.py` — self-contained HTML report (Vietnamese) with:
- Executive summary with top-3 rankings
- Dataset class distribution table & imbalance analysis
- Full configuration overview
- Metrics comparison table (all versions ranked)
- Comparison charts, per-class F1 heatmap, per-class breakdown
- Per-version detail cards with training history, test confusion matrix, and classification report
- Insights and recommendations for next steps

---

## Key Design Decisions

### Reproducibility
- **Fixed seed (42)** applied to `random`, `numpy`, `torch`, `torch.cuda`, and `cudnn.deterministic` before each version run — ensures fair comparison.

### Data Augmentation (Medical Domain)
- **Horizontal flip** ✅ — Valid (left/right knee are mirror images)
- **Vertical flip** ❌ — Intentionally omitted (inverts femur/tibia anatomy)
- **RandomCrop** from slightly larger resize (256→224) for spatial diversity
- **Small rotation** (±10°) and mild color jitter
- SupCon (v10) uses **stronger augmentations**: RandomResizedCrop(scale=0.5-1.0), larger rotation (±15°), stronger color jitter

### Metrics
- **Macro-averaged** F1, Precision, Recall, AUC-ROC — treats every class equally regardless of frequency
- **Per-class** breakdown in test evaluation for identifying which severity grades benefit most from each technique

### Early Stopping
- Patience = 7 epochs on validation F1-macro
- `ReduceLROnPlateau` scheduler (patience=3, factor=0.5)

---

## Configuration

All hyperparameters are centralized in [`src/configs/config.py`](src/configs/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_CLASSES` | 5 | Grade 0–4 |
| `IMG_SIZE` | 224 | ResNet50 input size |
| `BATCH_SIZE` | 32 | — |
| `EPOCHS` | 30 | — |
| `LR` | 1e-4 | AdamW learning rate |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization |
| `MIXUP_ALPHA` | 0.4 | Beta distribution parameter |
| `MIXUP_ADJACENT_GAP` | 1 | Max class gap for adjacent mixup |
| `MIXUP_RULE_LAM_MAX` | 0.49 | Max mixing coeff for Grade 0 in rule-based mixup |
| `MIXUP_TEMPERATURE` | 1.0 | OWMix Gaussian kernel width τ |
| `FOCAL_GAMMA` | 2.0 | Focal loss focusing parameter |
| `EARLY_STOPPING_PATIENCE` | 7 | Epochs without F1 improvement |
| `SUPCON_TEMPERATURE` | 0.1 | Temperature in SupCon loss |
| `SUPCON_PROJECT_DIM` | 128 | Projection head output dim |
| `SUPCON_EPOCHS` | 50 | SupCon pretraining epochs |
| `PROBE_EPOCHS` | 10 | Linear probe epochs |
| `FINETUNE_EPOCHS` | 20 | Full finetune epochs |
| `MIXUP_QUEUE_SIZE` | 64 | Per-class CPU memory queue capacity (v13+) |
| `MIXUP_QUEUE_MAX_REUSE` | 3 | Max times a cached sample is reused before eviction (v14+) |
| `OWMM_EFFNUM_BETA` | 0.9999 | Effective-number reweight beta for partner selection (v17+) |
| `OWMM_ORD_SIGMA` | 0.5 | Gaussian width of soft ordinal target smoothing (v17+) |
| `OWMM_MIX_SCALE_MIN` | 0.3 | Min per-sample mixing scale for majority class (v18) |
| `OWMM_PRIOR_GAMMA` | 1.0 | Tempering factor on BalSoft log-prior (v19) |

---

## References

- **Mixup**: Zhang et al. (2018) — *mixup: Beyond Empirical Risk Minimization*
- **Balanced Softmax**: Ren et al. (NeurIPS 2020) — *Balanced Meta-Softmax for Long-Tailed Visual Recognition*
- **Focal Loss**: Lin et al. (2017) — *Focal Loss for Dense Object Detection*
- **SupCon**: Khosla et al. (NeurIPS 2020) — *Supervised Contrastive Learning*
- **OWMix**: Zhang et al. (2023) — *Ordinal-Weighted Mixup for Imbalanced Medical Image Classification* (or equivalent ordinal mixup formulation)
- **Dataset (KOA)**: [Knee Osteoarthritis Dataset with Severity](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity) (Kaggle)
- **Dataset (EyePACS)**: [bumbledeep/eyepacs](https://huggingface.co/datasets/bumbledeep/eyepacs) (HuggingFace, MIT license)
