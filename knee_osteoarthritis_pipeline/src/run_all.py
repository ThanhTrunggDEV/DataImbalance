"""
run_all.py — Master orchestrator for all 5 experiment versions.

Usage:
    cd knee_osteoarthritis_pipeline/src
    python run_all.py                                       # full run (30 epochs each)
    python run_all.py --epochs 2                            # quick smoke test
    python run_all.py --skip v1_baseline                    # skip a version
    python run_all.py --only v3_balanced_softmax v5_focal_loss

After all versions finish, produces:
  results/comparison/metrics_comparison.png
  results/comparison/loss_f1_curves_comparison.png
  results/comparison/per_class_f1_heatmap.png
  results/comparison/all_versions_summary.csv
"""

import argparse
import csv
import json
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

import numpy as np

from configs.config import VERSIONS, DATA_DIR, PAPER_VERSION_NAMES
from run_experiment import run_version, set_seed
from evaluation.visualization import plot_all_versions_comparison, plot_per_class_f1_heatmap


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_path(results_dir, version_name, filename, seed=42):
    """Try seed subdirectory first, then root (legacy)."""
    path = os.path.join(results_dir, version_name, f"seed_{seed}", filename)
    if os.path.exists(path):
        return path
    path = os.path.join(results_dir, version_name, filename)
    return path if os.path.exists(path) else None


def _discover_seed_metrics(results_dir, version_name, filename):
    """Load `filename` from EVERY seed_* subdir found on disk for a version
    (not just seed_42), falling back to the legacy flat layout if no seed
    subdirs exist. Returns a list of parsed JSON dicts, one per seed found."""
    version_dir = os.path.join(results_dir, version_name)
    results = []
    if os.path.isdir(version_dir):
        seed_dirs = sorted(
            d for d in os.listdir(version_dir)
            if d.startswith("seed_") and os.path.isdir(os.path.join(version_dir, d))
        )
        for d in seed_dirs:
            path = os.path.join(version_dir, d, filename)
            if os.path.exists(path):
                with open(path) as f:
                    results.append(json.load(f))
    if not results:
        legacy_path = _resolve_path(results_dir, version_name, filename)
        if legacy_path:
            with open(legacy_path) as f:
                results.append(json.load(f))
    return results


def save_summary_csv(all_results: list, results_dir: str) -> str:
    """Write a CSV table of best val metrics for all versions."""
    comp_dir = os.path.join(results_dir, "comparison")
    os.makedirs(comp_dir, exist_ok=True)
    csv_path = os.path.join(comp_dir, "all_versions_summary.csv")

    fields = [
        "version", "display", "epoch",
        "val_f1_macro", "val_f1_macro_std",
        "val_accuracy", "val_accuracy_std",
        "val_precision_macro", "val_recall_macro",
        "val_auc_macro", "val_auc_macro_std",
        "val_loss", "elapsed_min",
        "n_seeds",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"\n  Summary CSV → {csv_path}")
    return csv_path


def print_summary_table(all_results: list):
    """Pretty-print a comparison table to stdout."""
    if not all_results:
        return

    header = (f"{'Version':<38} {'F1':>7} {'Acc':>7} "
              f"{'Prec':>7} {'Rec':>7} {'AUC':>7} {'Epoch':>6} {'Min':>6}")
    sep = "─" * len(header)
    print(f"\n{'='*len(header)}")
    print("  FINAL RESULTS SUMMARY")
    print(f"{'='*len(header)}")
    print(f"  {header}")
    print(f"  {sep}")

    for r in sorted(all_results, key=lambda x: x.get("val_f1_macro", 0), reverse=True):
        f1   = r.get("val_f1_macro",        0) or 0
        acc  = r.get("val_accuracy",         0) or 0
        prec = r.get("val_precision_macro",  0) or 0
        rec  = r.get("val_recall_macro",     0) or 0
        auc  = r.get("val_auc_macro",        0) or 0
        ep   = r.get("epoch",               "—")
        mins = r.get("elapsed_min",          0)
        disp = r.get("display", r.get("version", "?"))
        print(f"  {disp:<38} {f1:>7.4f} {acc:>7.4f} "
              f"{prec:>7.4f} {rec:>7.4f} {auc:>7.4f} {str(ep):>6} {mins:>6.1f}")

    print(f"{'='*len(header)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ALL experiment versions and compare results"
    )
    parser.add_argument("--epochs",      type=int,  default=None,
                        help="Override EPOCHS from config (e.g. 2 for smoke test)")
    parser.add_argument("--batch_size",  type=int,  default=None)
    parser.add_argument("--data_dir",    type=str,  default=None,
                        help=f"Path to data root (default: {DATA_DIR})")
    parser.add_argument("--results_dir", type=str, default=os.path.join(_SCRIPT_DIR, "../results"),
                        help="Where to store all outputs")
    parser.add_argument("--num_workers",  type=int, default=4,
                        help="DataLoader workers (0 to disable multiprocessing)")
    parser.add_argument("--skip",  nargs="*", default=[],
                        help="Version names to skip")
    parser.add_argument("--only",  nargs="*", default=[],
                        help="Run only these version names (overrides --skip)")
    parser.add_argument("--paper", action="store_true",
                        help="Run only PAPER_VERSION_NAMES from config.py "
                             "(the versions reported in manuscript.tex + v13). "
                             "Overridden by --only if both are given.")
    parser.add_argument("--supcon_epochs",   type=int, default=None,
                        help="SupCon pretrain epochs (default from config)")
    parser.add_argument("--probe_epochs",    type=int, default=None,
                        help="Linear probe epochs (default from config)")
    parser.add_argument("--finetune_epochs", type=int, default=None,
                        help="Full finetune epochs (default from config)")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42],
                        help="Random seeds for multi-seed runs (default: [42])")
    parser.add_argument("--cuda",  type=int, default=None,
                        help="CUDA device ID (overrides config.CUDA_DEVICE_ID)")
    parser.add_argument("--mixup_temperature", type=float, default=None,
                        help="OWMix temperature tau (overrides config.MIXUP_TEMPERATURE)")
    parser.add_argument("--prior_gamma", type=float, default=None,
                        help="v19: tempered balanced-softmax factor gamma in [0,1] "
                             "(overrides per-version config for ALL selected versions; "
                             "leave unset to use each version's baked gamma)")
    # Retuned-regime overrides (Plan B), forwarded to run_version via args
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides config.LR)")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early-stopping patience (overrides config.EARLY_STOPPING_PATIENCE)")
    parser.add_argument("--mixup_alpha", type=float, default=None,
                        help="Mixup Beta alpha (overrides config.MIXUP_ALPHA)")
    parser.add_argument("--scheduler", type=str, default="plateau",
                        choices=["plateau", "cosine"],
                        help="LR schedule: 'plateau' (default) or 'cosine' (warmup+cosine)")
    parser.add_argument("--warmup_epochs", type=int, default=3,
                        help="Warmup epochs for --scheduler cosine (default: 3)")
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset subdirectory under results_dir (e.g. koa, eyepacs)")
    return parser.parse_args()


def main():
    args = parse_args()

    # run_version() applies the dataset subdir itself; pre-appending here too
    # would nest results twice (results/koa/koa/...). Keep args.results_dir as
    # the base and compute the nested path locally for post-processing only.
    results_dir = os.path.join(args.results_dir, args.dataset) if args.dataset else args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    # Determine which versions to run
    to_run = VERSIONS
    if args.only:
        to_run = [v for v in VERSIONS if v["name"] in args.only]
    elif args.paper:
        to_run = [v for v in VERSIONS if v["name"] in PAPER_VERSION_NAMES]
    elif args.skip:
        to_run = [v for v in VERSIONS if v["name"] not in args.skip]

    print(f"\n{'#'*70}")
    print(f"  MULTI-VERSION IMBALANCE PIPELINE")
    if args.dataset:
        print(f"  Dataset         : {args.dataset}")
    print(f"  Versions to run : {[v['name'] for v in to_run]}")
    print(f"  Results dir     : {os.path.abspath(results_dir)}")
    print(f"{'#'*70}")

    seeds = args.seeds
    all_results  = []
    total_start  = time.time()

    for vcfg in to_run:
        print(f"\n{'#'*70}")
        print(f"  RUNNING: {vcfg['name']} over seeds {seeds}")
        print(f"{'#'*70}")

        seed_results = []
        for seed_idx, seed in enumerate(seeds):
            print(f"\n{'─'*60}")
            print(f"  Seed {seed} ({seed_idx+1}/{len(seeds)})")
            print(f"{'─'*60}")
            t0 = time.time()
            try:
                best = run_version(vcfg, args, seed=seed)
                elapsed = time.time() - t0
                seed_results.append({
                    "seed": seed,
                    "elapsed_min": round(elapsed / 60, 2),
                    **best,
                })
                print(f"\n  ✓ {vcfg['name']} (seed={seed}) done in {elapsed/60:.1f} min")
            except Exception as exc:
                print(f"\n  ✗ {vcfg['name']} (seed={seed}) FAILED: {exc}")
                import traceback; traceback.print_exc()

        if seed_results:
            row = {"version": vcfg["name"], "display": vcfg["display"]}
            metrics_keys = [
                "val_loss", "val_accuracy", "val_precision_macro",
                "val_recall_macro", "val_f1_macro", "val_auc_macro",
            ]
            for k in metrics_keys:
                vals = [r.get(k, 0) or 0 for r in seed_results]
                row[k] = round(float(np.mean(vals)), 6)
                row[f"{k}_std"] = round(float(np.std(vals)), 6)
            row["epoch"] = seed_results[0].get("epoch", "")
            row["elapsed_min"] = round(sum(r["elapsed_min"] for r in seed_results), 2)
            row["_per_seed"] = seed_results
            all_results.append(row)

    # ── Post-processing ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  All versions done. Generating comparison artefacts…")
    print(f"{'='*70}\n")

    # results_dir (nested with dataset) was computed at the top of main()

    # Validation metric comparison (bar chart + curves) — reads all versions from disk
    plot_all_versions_comparison(results_dir=results_dir, version_configs=VERSIONS)

    # Per-class F1 heatmap (test set) — reads all versions from disk
    plot_per_class_f1_heatmap(results_dir=results_dir, version_configs=VERSIONS)

    # Collect results from ALL versions on disk (old + new), averaged across
    # every seed_* subdir found for that version (not just seed_42).
    all_version_results = []
    for vcfg in VERSIONS:
        seed_metrics = _discover_seed_metrics(results_dir, vcfg["name"], "metrics.json")
        if not seed_metrics:
            continue
        bests = [data.get("best", {}) for data in seed_metrics]
        row = {
            "version": vcfg["name"],
            "display": vcfg["display"],
            "epoch":   bests[0].get("epoch", ""),
            "elapsed_min": 0,
            "n_seeds": len(bests),
        }
        for k in ("val_f1_macro", "val_accuracy", "val_precision_macro",
                  "val_recall_macro", "val_auc_macro", "val_loss"):
            vals = [b.get(k, 0) or 0 for b in bests]
            row[k] = round(float(np.mean(vals)), 6)
            row[f"{k}_std"] = round(float(np.std(vals)), 6)
        all_version_results.append(row)

    # Merge current-run elapsed times into the full results
    cur_results = {r["version"]: r for r in all_results}
    for row in all_version_results:
        if row["version"] in cur_results:
            row["elapsed_min"] = cur_results[row["version"]]["elapsed_min"]

    # Summary CSV + console table
    if all_version_results:
        save_summary_csv(all_version_results, results_dir)
        print_summary_table(all_version_results)

    total_elapsed = (time.time() - total_start) / 60
    print(f"\n  Total wall-clock time : {total_elapsed:.1f} min")
    print(f"  All outputs in        : {os.path.abspath(results_dir)}\n")


if __name__ == "__main__":
    main()
