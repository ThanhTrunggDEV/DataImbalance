# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A research project (with an accompanying LaTeX paper in `paper/`) that systematically
compares **class-imbalance mitigation strategies** on 5-class *ordinal* severity
classification. All code lives in `knee_osteoarthritis_pipeline/`; two datasets are used:
Knee Osteoarthritis X-rays (`data/`) and EyePACS Diabetic Retinopathy fundus images
(`data_dr/`), both structured as `{train,val,test}/{0,1,2,3,4}/`.

The novel contributions are the ordinal-aware Mixup variants (Adjacent, Rule-based, and
OWMix / Ordinal-Weighted Mixup), benchmarked against baselines (CE, Balanced Softmax,
Focal Loss, SupCon).

`README.md` (in the pipeline dir) and `AGENTS.md` (repo root) are authoritative for the
full version table, CLI flags, and dataset prep — read them before deep work rather than
re-deriving. This file captures the cross-file architecture and the gotchas that bite.

## Commands

**All scripts must be run from `knee_osteoarthritis_pipeline/src/`** — imports assume that
CWD, and paths like `../data`, `../results` are relative to it.

```bash
cd knee_osteoarthritis_pipeline/src
python run_all.py --dataset koa                          # full pipeline, all versions → results/koa/
python run_all.py --epochs 2                             # smoke test (2 epochs)
python run_experiment.py --version v1_baseline --epochs 10   # one version
python run_all.py --only v1_baseline v7_adjacent_balanced --seeds 42 123   # subset × multi-seed
python generate_report.py --dataset koa                  # build weekly_report.html from results/
python run_visualization.py                              # Grad-CAM + t-SNE (loads trained models)
python -m eda.eda                                        # exploratory data analysis
```

EyePACS runs point at the other dataset and its own results subdir:
```bash
python run_all.py --data_dir ../data_dr --dataset eyepacs --num_workers 4
```

There is no test suite, linter, or build step — this is a research pipeline. "Testing" a
change means running a version with `--epochs 2` and confirming it trains and writes
`results/.../metrics.json` without error.

## Critical gotchas

- **`cd src` first.** Nothing runs correctly from the repo root or the pipeline dir.
- **`--num_workers 0` on Windows.** With large datasets (EyePACS ~24K train images),
  `num_workers>0` causes shared-memory crashes on Windows. `num_workers=4` is fine on Linux.
- **Never combine Balanced Softmax with the sampler.** Variants with
  `loss_type="balanced_softmax"` (v3, v4, v7, v9, v12) set `use_sampler=False` deliberately —
  Balanced Softmax already corrects imbalance via a log-frequency prior, so adding
  `WeightedRandomSampler` double-corrects. This invariant is encoded in `VERSIONS` in
  `configs/config.py`; preserve it when editing.
- **Fixed seed.** `set_seed()` (default 42) fixes `random`/`numpy`/`torch`/cudnn before every
  run for fair comparison. Don't remove it; use `--seeds` to run multiple.
- **`--dataset` routes outputs.** Passing `--dataset koa|eyepacs` nests results under
  `results/{dataset}/`. Pass the *same* flag to `generate_report.py` or it reads the wrong dir.

## Architecture

**`configs/config.py` is the single source of truth for experiments.** The `VERSIONS` list
defines every experiment as a dict (`name`, `loss_type`, `use_mixup`, `mixup_mode`,
`use_sampler`, optional `mixup_temperature`). Adding an experiment = adding a dict here; the
runners iterate this list. All hyperparameters are module-level constants in the same file.

**Execution flow:**
`run_all.py` (orchestrator, loops versions × seeds, aggregates mean±std, builds comparison
plots + summary CSV) → `run_experiment.py::run_version` (builds model/loss/optimizer for one
version+seed, dispatches SupCon to its own 3-phase path) → `training/trainer.py::Trainer`
(shared training loop) → `evaluation/evaluator.py` (test-set eval on the reloaded best
checkpoint).

**Results layout is seed-nested:** `results/{dataset}/{version_dir}/seed_{seed}/`, where
`version_dir` appends a `_t{temp}` suffix for OWMix variants. `run_all.py::_resolve_path`
falls back to the legacy flat `results/{dataset}/{version}/` layout when the seed subdir is
absent — keep both paths working if you touch result I/O.

**The Mixup variants are the core novelty**, all in `training/trainer.py` as standalone
functions dispatched by `mixup_mode` inside `Trainer.train_epoch`:
- `mixup_data` (standard), `mixup_data_adjacent` (only pairs `|y_a−y_b| ≤ gap`),
  `mixup_data_rule_based` (mixes Grade-0 anchors with Grade {2,3,4}, `lam` clamped `<0.5`),
  `mixup_data_ordinal_weighted` (OWMix: partner sampled from a Gaussian kernel over ordinal
  label distance × inverse-sqrt class frequency).
- All four return `(mixed_x, y_a, y_b, lam)` and share `mixup_criterion`, which computes
  `lam·L(y_a) + (1−lam)·L(y_b)`. Unmixed samples fall out naturally as `y_a==y_b` — that's why
  the functions return full batches rather than subsets.

**SupCon (v10) is the exception to the shared path** — `run_experiment.py::_run_supcon` runs
three phases (contrastive pretrain of backbone+projection head via `fit_supcon` → linear probe
with frozen backbone → full finetune), passing the backbone through `backbone_supcon.pth`
between phases.

**Loss factory:** `training/loss.py::build_loss` maps `loss_type` strings to loss objects
(`FocalLoss`, `BalancedSoftmaxLoss`, `SupConLoss`, plain CE). `metrics.py::calculate_metrics`
computes the ordinal-aware metrics that matter here — macro F1/P/R/AUC plus **QWK** (quadratic
weighted kappa) and **MAE**, which reward "close" ordinal predictions.

**Backbone:** ResNet50 (`models/resnet.py`) is the default across all versions;
`efficientnet.py` is an alternative. Grad-CAM hooks `layer4`; t-SNE reads `avgpool` features.

## The paper

`paper/manuscript.tex` (Springer `svproc` class) is the write-up; figures in `paper/` and
`report_assets/` are generated from pipeline outputs. The active branch is `paper`. When
editing the manuscript, keep terminology consistent with the code (e.g. "Adjacent Mixup",
"OWMix") and the design rationale in `README.md`.
