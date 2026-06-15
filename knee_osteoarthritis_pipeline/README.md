# Knee Osteoarthritis — Multi-Version Imbalance Experiment Pipeline

Automated framework for systematic evaluation of **class-imbalance mitigation strategies** on Knee Osteoarthritis severity classification (5-class: Grade 0–4) from X-ray images.

The pipeline trains **18 experiment variants** (across 12 version families) with different imbalance techniques on the same ResNet50 backbone, then generates consolidated comparison reports (CSV, bar charts, F1 heatmaps) and an HTML weekly report for rigorous side-by-side analysis.

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

> **Adjacent Mixup** (`v6`, `v7`): Only pairs images whose class labels differ by at most 1 grade (e.g., Grade 0↔1, 1↔2, 2↔3, 3↔4). Samples that cannot find a valid partner are left unmixed.  
> **Rule-based Mixup** (`v8`, `v9`): Selects Grade 0 images as anchors and pairs them with Grade {2,3,4}. The mixing coefficient `lam` is clamped to <0.5 so the non-zero partner always contributes the majority, and the label is assigned to that partner. Batches without Grade 0 skip mixup.  
> **SupCon** (`v10`): Three-phase training — (1) Supervised contrastive pretraining of backbone + projection head with 2-augment views, (2) linear probe with frozen backbone, (3) full finetune of all parameters.  
> **OWMix** (`v11`, `v12`): Ordinal-Weighted Mixup — assigns mixing coefficients via a Gaussian kernel over the ordinal label distance (`τ` controls kernel width). Each variant runs 4 temperature settings (`τ ∈ {0.5, 1.0, 1.5, 2.0}`) for a total of 8 OWMix sub-variants.

> **Design note:** BalancedSoftmaxLoss (Ren et al., NeurIPS 2020) inherently corrects for class imbalance via log-frequency prior — combining it with WeightedRandomSampler would double-correct, so variants with BalancedSoftmax intentionally disable the sampler.

---

## Project Structure

```
knee_osteoarthritis_pipeline/
├── README.md
├── requirements.txt
├── weekly_report.html             # 📊 Auto-generated HTML weekly report
├── data/                          # Dataset (not tracked in git)
│   └── train/ val/ test/          # Each with sub-folders: 0/ 1/ 2/ 3/ 4/
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
    │   └── download.py            # Kaggle dataset downloader
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

> **Note:** To auto-download the Kaggle dataset via `data/download.py`, place a valid `kaggle.json` in `~/.kaggle/`.

---

## Dataset Preparation

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
```

### Run a Single Version

```bash
cd src
python run_experiment.py --version v1_baseline --epochs 10
python run_experiment.py --version v5_focal_loss --batch_size 16
python run_experiment.py --version v10_supcon --supcon_epochs 50 --probe_epochs 10
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
| `--skip` | Version names to skip (run_all only) | — |
| `--only` | Run only these versions (run_all only) | — |
| `--mixup_temperature` | OWMix temperature τ (run_experiment only) | config default |
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

---

## References

- **Mixup**: Zhang et al. (2018) — *mixup: Beyond Empirical Risk Minimization*
- **Balanced Softmax**: Ren et al. (NeurIPS 2020) — *Balanced Meta-Softmax for Long-Tailed Visual Recognition*
- **Focal Loss**: Lin et al. (2017) — *Focal Loss for Dense Object Detection*
- **SupCon**: Khosla et al. (NeurIPS 2020) — *Supervised Contrastive Learning*
- **OWMix**: Zhang et al. (2023) — *Ordinal-Weighted Mixup for Imbalanced Medical Image Classification* (or equivalent ordinal mixup formulation)
- **Dataset**: [Knee Osteoarthritis Dataset with Severity](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity) (Kaggle)
