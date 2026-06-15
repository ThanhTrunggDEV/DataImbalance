# OpenCode Instructions for Knee Osteoarthritis Pipeline

## Core Context
- **Project**: Systematic comparison of imbalance mitigation strategies (18 experiment versions) on 5-class ordinal classification.
- **Datasets**: Knee OA (Chen 2018, `data/`) + EyePACS Diabetic Retinopathy (`data_dr/`).
- **Entry Points**: All scripts must be executed from the `src/` directory.
    - Full Pipeline: `python run_all.py`
    - Single Experiment: `python run_experiment.py --version <name>`
- **Configuration**: Hyperparameters, model settings, and experiment versions are centralized in `src/configs/config.py`.

## Critical Operational Gotchas
- **Execution Path**: Always `cd src/` before running experiments.
- **Dataset**: Both `data/` and `data_dr/` are gitignored. Run `python data/download_eyepacs.py` for EyePACS (requires `pip install datasets`).
- **Reproducibility**: Experiments use fixed seed 42; do not modify seeds in `src/run_experiment.py`.
- **Architecture Constraint**: Variants using `BalancedSoftmaxLoss` (`v3, v4, v7, v9`) must NOT be run with `WeightedRandomSampler` (they inherently correct imbalance; combining them causes double-correction).
- **SupCon (v10)**: Resource-intensive, requiring 3-phase training (pretrain, probe, finetune).
- **Windows num_workers**: Large datasets (EyePACS: 24K train) cause shared memory errors with `num_workers=4`. Use `--num_workers 0` on Windows. On Linux, `num_workers=4` is safe.
- **--dataset flag**: Always use `--dataset koa` or `--dataset eyepacs` to separate results into `results/{dataset}/` subdirectories. Use `--dataset` with `generate_report.py` too.

## Common Commands
- **Full Run (KOA)**: `cd src && python run_all.py --dataset koa`
- **Full Run (EyePACS)**: `cd src && python run_all.py --data_dir ../data_dr --dataset eyepacs --num_workers 4`
- **Smoke Test**: `cd src && python run_all.py --epochs 2`
- **Single Experiment**: `cd src && python run_experiment.py --version v1_baseline --epochs 10`
- **Top 6 × 2 seeds (KOA, Linux server)**: `cd src && python run_all.py --dataset koa --num_workers 4 --only v1_baseline v3_balanced_softmax v5_focal_loss v7_adjacent_balanced v11_owmixup_ce v12_owmixup_balanced_t20 --seeds 42 123`
- **Top 6 × 2 seeds (EyePACS, Linux server)**: `cd src && python run_all.py --data_dir ../data_dr --dataset eyepacs --num_workers 4 --only v1_baseline v3_balanced_softmax v5_focal_loss v7_adjacent_balanced v11_owmixup_ce v12_owmixup_balanced_t20 --seeds 42 123`
- **Generate Report**: `cd src && python generate_report.py --dataset koa`
- **EyePACS Download**: `cd src && pip install datasets && python data/download_eyepacs.py`
