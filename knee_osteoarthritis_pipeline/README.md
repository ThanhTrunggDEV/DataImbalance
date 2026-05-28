# Knee Osteoarthritis — Multi-Version Imbalance Experiment Pipeline

Automated framework for systematic evaluation of **class-imbalance mitigation strategies** on Knee Osteoarthritis severity classification (5-class: Grade 0–4) from X-ray images.

The pipeline trains **5 experiment versions** with different imbalance techniques on the same ResNet50 backbone, then generates consolidated comparison reports (CSV, bar charts, F1 heatmaps) for rigorous side-by-side analysis.

---

## Experiment Versions

| Version | Technique | Loss Function | Sampler |
|---------|-----------|---------------|---------|
| `v1_baseline` | None (pure baseline) | CrossEntropy | No |
| `v2_mixup` | Mixup augmentation | CrossEntropy | WeightedRandomSampler |
| `v3_balanced_softmax` | Logit-margin shift | BalancedSoftmaxLoss | No (handled by loss) |
| `v4_mixup_balanced_softmax` | Mixup + logit-margin | BalancedSoftmaxLoss | No (handled by loss) |
| `v5_focal_loss` | Hard-example mining | FocalLoss (γ=2.0) | WeightedRandomSampler |

> **Design note:** BalancedSoftmaxLoss (Ren et al., NeurIPS 2020) inherently corrects for class imbalance via log-frequency prior — combining it with WeightedRandomSampler would double-correct, so v3/v4 intentionally disable the sampler.

---

## Project Structure

```
knee_osteoarthritis_pipeline/
├── README.md
├── requirements.txt
├── data/                          # Dataset (not tracked in git)
│   └── train/ val/ test/          # Each with sub-folders: 0/ 1/ 2/ 3/ 4/
├── results/                       # Auto-generated outputs (not tracked)
│   ├── v1_baseline/
│   │   ├── best_model.pth
│   │   ├── metrics.json
│   │   ├── training_history.png
│   │   ├── confusion_matrix.png
│   │   ├── test_metrics.json
│   │   └── test_confusion_matrix.png
│   ├── v2_mixup/ ...
│   ├── v3_balanced_softmax/ ...
│   ├── v4_mixup_balanced_softmax/ ...
│   ├── v5_focal_loss/ ...
│   └── comparison/
│       ├── metrics_comparison.png
│       ├── loss_f1_curves_comparison.png
│       ├── per_class_f1_heatmap.png
│       └── all_versions_summary.csv
└── src/
    ├── run_all.py                 # 🚀 Master orchestrator (entry point)
    ├── run_experiment.py          # Single-version runner + seed control
    ├── configs/
    │   └── config.py              # Hyperparameters, VERSIONS, device
    ├── data/
    │   ├── dataset.py             # KneeDataset, DataLoader factory, augmentation
    │   └── download.py            # Kaggle dataset downloader
    ├── models/
    │   ├── resnet.py              # ResNet50 (primary backbone)
    │   └── efficientnet.py        # EfficientNet-B5 (alternative, not used)
    ├── training/
    │   ├── loss.py                # FocalLoss, BalancedSoftmaxLoss, build_loss factory
    │   ├── metrics.py             # calculate_metrics (F1, AUC, confusion matrix)
    │   ├── trainer.py             # Trainer class with Mixup, early stopping, checkpoints
    │   └── visualization.py       # Per-epoch training history plots
    └── evaluation/
        ├── evaluator.py           # Test-set evaluation, classification report
        └── visualization.py       # Cross-version comparison plots, F1 heatmap
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

### Run All 5 Versions (Full Pipeline)

```bash
cd src
python run_all.py                        # Full run (30 epochs each)
python run_all.py --epochs 2             # Quick smoke test
python run_all.py --only v1_baseline v5_focal_loss   # Run specific versions
python run_all.py --skip v2_mixup        # Skip a version
```

### Run a Single Version

```bash
cd src
python run_experiment.py --version v1_baseline --epochs 10
python run_experiment.py --version v5_focal_loss --batch_size 16
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--epochs` | Override epoch count | 30 |
| `--batch_size` | Override batch size | 32 |
| `--data_dir` | Path to dataset root | `../data` |
| `--results_dir` | Output directory | `../results` |
| `--skip` | Version names to skip (run_all only) | — |
| `--only` | Run only these versions (run_all only) | — |

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

### Comparison (`results/comparison/`)
- `all_versions_summary.csv` — All versions ranked by F1-macro
- `metrics_comparison.png` — Grouped bar chart of all metrics
- `loss_f1_curves_comparison.png` — Overlaid validation curves
- `per_class_f1_heatmap.png` — Per-class F1 heatmap across all versions

---

## Key Design Decisions

### Reproducibility
- **Fixed seed (42)** applied to `random`, `numpy`, `torch`, `torch.cuda`, and `cudnn.deterministic` before each version run — ensures fair comparison.

### Data Augmentation (Medical Domain)
- **Horizontal flip** ✅ — Valid (left/right knee are mirror images)
- **Vertical flip** ❌ — Intentionally omitted (inverts femur/tibia anatomy)
- **RandomCrop** from slightly larger resize (256→224) for spatial diversity
- **Small rotation** (±10°) and mild color jitter

### Metrics
- **Macro-averaged** F1, Precision, Recall, AUC-ROC — treats every class equally regardless of frequency
- **Per-class** breakdown in test evaluation for identifying which severity grades benefit most from each technique

### Early Stopping
- Patience = 7 epochs on validation F1-macro
- `ReduceLROnPlateau` scheduler (patience=3, factor=0.5)

---

## Configuration

All hyperparameters are centralized in [`src/configs/config.py`](src/configs/config.py):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_CLASSES` | 5 | Grade 0–4 |
| `IMG_SIZE` | 224 | ResNet50 input size |
| `BATCH_SIZE` | 32 | — |
| `EPOCHS` | 30 | — |
| `LR` | 1e-4 | AdamW learning rate |
| `WEIGHT_DECAY` | 1e-4 | L2 regularization |
| `MIXUP_ALPHA` | 0.4 | Beta distribution parameter |
| `FOCAL_GAMMA` | 2.0 | Focal loss focusing parameter |
| `EARLY_STOPPING_PATIENCE` | 7 | Epochs without F1 improvement |

---

## References

- **Mixup**: Zhang et al. (2018) — *mixup: Beyond Empirical Risk Minimization*
- **Balanced Softmax**: Ren et al. (NeurIPS 2020) — *Balanced Meta-Softmax for Long-Tailed Visual Recognition*
- **Focal Loss**: Lin et al. (2017) — *Focal Loss for Dense Object Detection*
- **Dataset**: [Knee Osteoarthritis Dataset with Severity](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity) (Kaggle)