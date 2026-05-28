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
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import VERSIONS, DATA_DIR
from run_experiment import run_version
from evaluation.visualization import plot_all_versions_comparison, plot_per_class_f1_heatmap


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_summary_csv(all_results: list, results_dir: str) -> str:
    """Write a CSV table of best val metrics for all versions."""
    comp_dir = os.path.join(results_dir, "comparison")
    os.makedirs(comp_dir, exist_ok=True)
    csv_path = os.path.join(comp_dir, "all_versions_summary.csv")

    fields = [
        "version", "display", "epoch",
        "val_f1_macro", "val_accuracy",
        "val_precision_macro", "val_recall_macro", "val_auc_macro",
        "val_loss", "elapsed_min",
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
    parser.add_argument("--results_dir", type=str,  default="../results",
                        help="Where to store all outputs")
    parser.add_argument("--skip",  nargs="*", default=[],
                        help="Version names to skip")
    parser.add_argument("--only",  nargs="*", default=[],
                        help="Run only these version names (overrides --skip)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine which versions to run
    to_run = VERSIONS
    if args.only:
        to_run = [v for v in VERSIONS if v["name"] in args.only]
    elif args.skip:
        to_run = [v for v in VERSIONS if v["name"] not in args.skip]

    print(f"\n{'#'*70}")
    print(f"  KNEE OA — MULTI-VERSION IMBALANCE PIPELINE")
    print(f"  Versions to run : {[v['name'] for v in to_run]}")
    print(f"  Results dir     : {os.path.abspath(args.results_dir)}")
    print(f"{'#'*70}")

    all_results  = []
    total_start  = time.time()

    for vcfg in to_run:
        t0 = time.time()
        try:
            best = run_version(vcfg, args)
            elapsed = time.time() - t0
            row = {
                "version":     vcfg["name"],
                "display":     vcfg["display"],
                **best,
                "elapsed_min": round(elapsed / 60, 2),
            }
            all_results.append(row)
            print(f"\n  ✓ {vcfg['name']} completed in {elapsed/60:.1f} min")
        except Exception as exc:
            print(f"\n  ✗ {vcfg['name']} FAILED: {exc}")
            import traceback; traceback.print_exc()

    # ── Post-processing ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  All versions done. Generating comparison artefacts…")
    print(f"{'='*70}\n")

    results_dir = args.results_dir

    # Validation metric comparison (bar chart + curves)
    plot_all_versions_comparison(results_dir=results_dir, version_configs=to_run)

    # Per-class F1 heatmap (test set)
    plot_per_class_f1_heatmap(results_dir=results_dir, version_configs=to_run)

    # Summary CSV + console table
    if all_results:
        save_summary_csv(all_results, results_dir)
        print_summary_table(all_results)

    total_elapsed = (time.time() - total_start) / 60
    print(f"\n  Total wall-clock time : {total_elapsed:.1f} min")
    print(f"  All outputs in        : {os.path.abspath(results_dir)}\n")


if __name__ == "__main__":
    main()
