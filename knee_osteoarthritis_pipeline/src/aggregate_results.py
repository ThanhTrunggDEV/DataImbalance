#!/usr/bin/env python3
"""
Aggregate experiment results from all version directories into the summary CSV.
Scans results/{dataset}/<version>/seed_*/ for metrics.json and test_metrics.json.
"""
import argparse
import csv
import json
import os
import sys
import math
from pathlib import Path
from collections import defaultdict

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from configs.config import VERSIONS

METRICS_KEYS = [
    "val_f1_macro", "val_accuracy", "val_precision_macro",
    "val_recall_macro", "val_auc_macro", "val_loss",
]
SUMMARY_KEYS = [
    "version", "display", "epoch", "val_f1_macro", "val_f1_macro_std",
    "val_accuracy", "val_accuracy_std", "val_precision_macro",
    "val_recall_macro", "val_auc_macro", "val_auc_macro_std",
    "val_loss", "elapsed_min", "n_seeds",
]


def find_seed_dirs(version_dir):
    """Find all seed_* subdirectories, or return [None] if results are in root."""
    if not os.path.isdir(version_dir):
        return []
    seeds = sorted([d for d in os.listdir(version_dir) if d.startswith("seed_")])
    if not seeds:
        return [None]
    return seeds


def load_metrics(version_dir, seed):
    """Load metrics.json from seed subdir or version root."""
    if seed:
        path = os.path.join(version_dir, seed, "metrics.json")
    else:
        path = os.path.join(version_dir, "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def load_test_metrics(version_dir, seed):
    """Load test_metrics.json from seed subdir or version root."""
    if seed:
        path = os.path.join(version_dir, seed, "test_metrics.json")
    else:
        path = os.path.join(version_dir, "test_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def extract_best(metrics_data):
    """Extract best epoch metrics."""
    if not metrics_data:
        return {}
    best = metrics_data.get("best", {})
    return {
        "epoch": best.get("epoch", ""),
        "val_f1_macro": best.get("val_f1_macro"),
        "val_accuracy": best.get("val_accuracy"),
        "val_precision_macro": best.get("val_precision_macro"),
        "val_recall_macro": best.get("val_recall_macro"),
        "val_auc_macro": best.get("val_auc_macro"),
        "val_loss": best.get("val_loss"),
    }


def build_display_map():
    """Build version -> display name from VERSIONS config."""
    dmap = {}
    for v in VERSIONS:
        dmap[v["name"]] = v["display"]
    return dmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--dataset", default="koa")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.results_dir:
        results_root = Path(args.results_dir)
    else:
        results_root = script_dir.parent / "results"

    results_dir = results_root / args.dataset
    if not results_dir.is_dir():
        print(f"[ERROR] Results dir not found: {results_dir}")
        sys.exit(1)

    display_map = build_display_map()

    # Collect all version directories (exclude "comparison")
    version_dirs = sorted([
        d for d in os.listdir(results_dir)
        if os.path.isdir(results_dir / d) and d != "comparison"
    ])

    rows = []
    for vn in version_dirs:
        vdir = results_dir / vn
        seeds = find_seed_dirs(vdir)

        seed_bests = []
        display = display_map.get(vn, vn)

        for seed in seeds:
            m = load_metrics(vdir, seed)
            t = load_test_metrics(vdir, seed)
            if m and t:
                best = extract_best(m)
                seed_bests.append({
                    **best,
                    "test_f1": t.get("test_f1_macro"),
                })

        if not seed_bests:
            continue

        n = len(seed_bests)
        row = {"version": vn, "display": display, "n_seeds": str(n)}

        # Mean & std across seeds for val metrics
        for key in ["val_f1_macro", "val_accuracy", "val_precision_macro",
                     "val_recall_macro", "val_auc_macro", "val_loss", "epoch"]:
            vals = [s.get(key) for s in seed_bests if s.get(key) is not None]
            if vals:
                mean_val = sum(vals) / len(vals)
                row[key] = f"{mean_val:.6f}"
            else:
                row[key] = ""

            # std for selected metrics
            std_key = f"{key}_std" if key != "epoch" else None
            if std_key and key in ["val_f1_macro", "val_accuracy", "val_auc_macro"]:
                if len(vals) > 1:
                    var = sum((v - sum(vals)/len(vals))**2 for v in vals) / len(vals)
                    std_val = math.sqrt(var)
                    row[std_key] = f"{std_val:.6f}"
                else:
                    row[std_key] = ""

        # Determine best epoch (use mode-like approach: pick epoch with best avg val_f1)
        f1_vals = [(s.get("val_f1_macro", 0), s.get("epoch", 0)) for s in seed_bests]
        if f1_vals:
            best_epoch = max(f1_vals, key=lambda x: x[0])[1]
            row["epoch"] = str(int(best_epoch))

        row["elapsed_min"] = ""
        rows.append(row)

        print(f"  {vn}: {n} seed(s), val_f1={seed_bests[0].get('val_f1_macro', 'N/A')}")

    # Sort by val_f1_macro descending
    rows.sort(key=lambda r: float(r.get("val_f1_macro", 0) or 0), reverse=True)

    # Write CSV
    comparison_dir = results_dir / "comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    csv_path = comparison_dir / "all_versions_summary.csv"

    # Deduplicate: keep only last entry per version
    seen = set()
    deduped = []
    for r in rows:
        if r["version"] not in seen:
            seen.add(r["version"])
            deduped.append(r)
        else:
            print(f"  [WARN] Duplicate version entry: {r['version']}")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_KEYS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(deduped)

    print(f"\nSummary CSV written: {csv_path}")
    print(f"Total versions: {len(deduped)}")

    # Also print a comparison table to stdout
    print("\n===== SO SANH CAC PHIEN BAN =====")
    print(f"{'#':<3} {'Version':<35} {'F1 Macro':<10} {'Acc':<10} {'Prec':<10} {'Recall':<10} {'AUC':<10} {'#Seeds':<7}")
    print("-" * 95)
    for i, r in enumerate(deduped):
        f1 = r.get("val_f1_macro", "")
        acc = r.get("val_accuracy", "")
        prec = r.get("val_precision_macro", "")
        rec = r.get("val_recall_macro", "")
        auc = r.get("val_auc_macro", "")
        ns = r.get("n_seeds", "")
        print(f"{i+1:<3} {r['version']:<35} {float(f1 or 0):<10.4f} {float(acc or 0):<10.4f} {float(prec or 0):<10.4f} {float(rec or 0):<10.4f} {float(auc or 0):<10.4f} {ns:<7}")


if __name__ == "__main__":
    main()
