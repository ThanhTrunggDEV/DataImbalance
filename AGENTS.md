# OpenCode Instructions for Knee Osteoarthritis Pipeline

## Core Context
- **Project**: Systematic comparison of imbalance mitigation strategies (10 experiment versions) on 5-class Osteoarthritis severity classification.
- **Entry Points**: All scripts must be executed from the `src/` directory.
    - Full Pipeline: `python run_all.py`
    - Single Experiment: `python run_experiment.py --version <name>`
- **Configuration**: Hyperparameters, model settings, and experiment versions are centralized in `src/configs/config.py`.

## Critical Operational Gotchas
- **Execution Path**: Always `cd src/` before running experiments.
- **Dataset**: The `data/` directory is untracked. Ensure it follows the required `data/<split>/<0-4>/` structure.
- **Reproducibility**: Experiments use fixed seed 42; do not modify seeds in `src/run_experiment.py`.
- **Architecture Constraint**: Variants using `BalancedSoftmaxLoss` (`v3, v4, v7, v9`) must NOT be run with `WeightedRandomSampler` (they inherently correct imbalance; combining them causes double-correction).
- **SupCon (v10)**: This version is resource-intensive, requiring 3-phase training (pretrain, probe, finetune).

## Common Commands
- **Full Run**: `cd src && python run_all.py`
- **Smoke Test**: `cd src && python run_all.py --epochs 2`
- **Single Experiment**: `cd src && python run_experiment.py --version v1_baseline --epochs 10`
