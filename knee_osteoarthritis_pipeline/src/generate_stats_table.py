#!/usr/bin/env python3
"""
Generate statistical comparison table (Markdown) of all experiment versions vs baseline.
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from collections import defaultdict

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
from configs.config import VERSIONS, NUM_CLASSES

CLASS_NAMES = [f"Grade {i}" for i in range(NUM_CLASSES)]

# Versions numbered >= this are the "novel" contributions (ordinal-mixup family);
# derived from the vN_ prefix so newly added versions (v16, v17, v18, …) are
# picked up automatically instead of being hardcoded to v13/v14/v15.
NEW_VERSION_MIN = 13


def version_number(vn):
    """Integer N from a 'vN_...' directory name, or -1 if it doesn't match."""
    m = re.match(r"v(\d+)_", vn)
    return int(m.group(1)) if m else -1


def is_new_version(vn):
    return version_number(vn) >= NEW_VERSION_MIN


def find_file(version_dir, filename):
    """Find file in any seed_* subdir (lowest seed first) or the version root."""
    seed_dirs = sorted(glob.glob(os.path.join(version_dir, "seed_*")))
    for sd in seed_dirs:
        path = os.path.join(sd, filename)
        if os.path.exists(path):
            return path
    path = os.path.join(version_dir, filename)
    return path if os.path.exists(path) else None


def load_results(results_dir):
    """Load test metrics for all available versions."""
    results = {}
    for v in VERSIONS:
        vn = v["name"]
        vdir = os.path.join(results_dir, vn)
        if not os.path.isdir(vdir):
            continue
        test_path = find_file(vdir, "test_metrics.json")
        metrics_path = find_file(vdir, "metrics.json")
        if test_path:
            with open(test_path) as f:
                results[vn] = json.load(f)
            if metrics_path:
                with open(metrics_path) as f:
                    m = json.load(f)
                results[vn]["val_f1"] = m.get("best", {}).get("val_f1_macro", None)
        else:
            # Try the exact directory name as version name
            # (for v14/v15 with _t10 and _t20_t20 suffixes)
            pass
    return results


def main():
    # The report body is Vietnamese; force UTF-8 stdout so it doesn't crash on
    # Windows' default cp1252 console (UnicodeEncodeError on 'ả', 'ế', …).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--dataset", default="koa", help="Dataset subdirectory")
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

    # Build display name map
    display_map = {}
    for v in VERSIONS:
        display_map[v["name"]] = v["display"]

    # Scan all version directories (including non-standard names)
    all_dirs = sorted([
        d for d in os.listdir(results_dir)
        if os.path.isdir(results_dir / d) and d != "comparison"
    ])

    results = {}
    for vn in all_dirs:
        vdir = results_dir / vn
        test_path = find_file(vdir, "test_metrics.json")
        metrics_path = find_file(vdir, "metrics.json")
        if test_path:
            with open(test_path) as f:
                data = json.load(f)
            data["display"] = display_map.get(vn, vn)
            data["version"] = vn
            if metrics_path:
                with open(metrics_path) as f:
                    m = json.load(f)
                data["val_f1"] = m.get("best", {}).get("val_f1_macro", None)
                data["best_epoch"] = m.get("best", {}).get("epoch", None)
            results[vn] = data

    if not results:
        print("No results found.")
        return

    baseline = results.get("v1_baseline", None)
    baseline_f1 = baseline.get("test_f1_macro", 0) if baseline else 0

    # Sort by test F1 macro descending
    sorted_vns = sorted(results.keys(), key=lambda vn: results[vn].get("test_f1_macro", 0), reverse=True)

    # ===== TABLE 1: Overall test metrics =====
    print("# Bảng thống kê kết quả thực nghiệm (Test Set)\n")
    print(f"**Dataset:** {args.dataset.upper()}  ")
    print(f"**Baseline F1 Macro:** {baseline_f1:.4f}  ")
    print()

    header = ["#", "Phiên bản", "Test F1 Macro", "Δ vs Baseline", "Accuracy", "Precision", "Recall", "AUC", "Val F1"]
    sep = ["---"] * len(header)
    rows = []
    for rank, vn in enumerate(sorted_vns, 1):
        r = results[vn]
        tf1 = r.get("test_f1_macro", 0) or 0
        acc = r.get("test_accuracy", 0) or 0
        prec = r.get("test_precision_macro", 0) or 0
        rec = r.get("test_recall_macro", 0) or 0
        auc = r.get("test_auc_macro", 0) or 0
        vf1 = r.get("val_f1", 0) or 0
        delta = tf1 - baseline_f1
        delta_str = f"{delta:+.4f}"
        if vn == "v1_baseline":
            delta_str = "—"
        display = r.get("display", vn)
        rows.append([str(rank), display, f"{tf1:.4f}", delta_str, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{auc:.4f}", f"{vf1:.4f}"])

    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join(sep) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")

    # ===== TABLE 2: Per-class F1 for all versions =====
    print("\n## Per-class F1 trên Test Set\n")
    print("| Phiên bản | " + " | ".join(CLASS_NAMES) + " | F1 Macro |")
    print("|" + "---|" * (NUM_CLASSES + 2) + "|")
    for vn in sorted_vns:
        r = results[vn]
        pc = r.get("per_class", {})
        f1s = []
        for c in range(NUM_CLASSES):
            f1 = pc.get(str(c), {}).get("f1", 0)
            f1s.append(f"{f1:.4f}")
        macro = f"{r.get('test_f1_macro', 0):.4f}"
        display = r.get("display", vn)
        print(f"| {display} | " + " | ".join(f1s) + f" | {macro} |")

    # ===== TABLE 3: New versions (v13+) vs Baseline analysis =====
    print("\n## So sánh các phiên bản mới (v13+) với Baseline\n")
    print("| Phiên bản | Strategy | Test F1 | Δ F1 | Acc | AUC | Grade0 F1 | Grade1 F1 | Grade2 F1 | Grade3 F1 | Grade4 F1 |")
    print("|---|" + "---|" * 10 + "|")

    # Only show new versions (v13+), sorted by F1
    new_vns = [vn for vn in sorted_vns if is_new_version(vn)]
    if baseline:
        bf1 = baseline.get("test_f1_macro", 0)
        bpc = baseline.get("per_class", {})
        bf1s = [f"{bpc.get(str(c), {}).get('f1', 0):.4f}" for c in range(NUM_CLASSES)]
        bacc = baseline.get("test_accuracy", 0)
        bauc = baseline.get("test_auc_macro", 0)
        print(f"| **v1_baseline** | Cross-Entropy (baseline) | {bf1:.4f} | — | {bacc:.4f} | {bauc:.4f} | " + " | ".join(bf1s) + " |")

    for vn in new_vns:
        r = results[vn]
        tf1 = r.get("test_f1_macro", 0) or 0
        acc = r.get("test_accuracy", 0) or 0
        auc = r.get("test_auc_macro", 0) or 0
        pc = r.get("per_class", {})
        f1s = [f"{pc.get(str(c), {}).get('f1', 0):.4f}" for c in range(NUM_CLASSES)]
        delta = tf1 - baseline_f1
        display = r.get("display", vn)
        print(f"| {vn} | {display} | {tf1:.4f} | {delta:+.4f} | {acc:.4f} | {auc:.4f} | " + " | ".join(f1s) + " |")

    # ===== TABLE 4: Summary statistics =====
    print("\n## Thống kê tổng hợp\n")
    print(f"- **Tổng số phiên bản:** {len(results)}")
    print(f"- **Số phiên bản mới (v13+):** {len([v for v in all_dirs if is_new_version(v)])}")
    print(f"- **Baseline (v1)** test F1: {baseline_f1:.4f}")

    # Best overall
    best_vn = sorted_vns[0]
    best = results[best_vn]
    print(f"- **Best overall:** {best.get('display', best_vn)} (F1={best.get('test_f1_macro',0):.4f}, Δ={best.get('test_f1_macro',0)-baseline_f1:+.4f})")

    # Best among new versions (v13+)
    new_filtered = [vn for vn in sorted_vns if is_new_version(vn)]
    if new_filtered:
        best_new_vn = new_filtered[0]
        best_new = results[best_new_vn]
        print(f"- **Best v13+:** {best_new.get('display', best_new_vn)} (F1={best_new.get('test_f1_macro',0):.4f}, Δ={best_new.get('test_f1_macro',0)-baseline_f1:+.4f})")

    # Number of v13+ versions that beat baseline
    beat_baseline = sum(1 for vn in new_vns if (results[vn].get("test_f1_macro", 0) or 0) > baseline_f1)
    total_new = len(new_vns)
    print(f"- **v13+ beat baseline:** {beat_baseline}/{total_new} ({beat_baseline/total_new*100:.0f}%)")

    # Worst class analysis
    class_f1s = {c: [] for c in range(NUM_CLASSES)}
    for vn, r in results.items():
        pc = r.get("per_class", {})
        for c in range(NUM_CLASSES):
            f1 = pc.get(str(c), {}).get("f1", 0)
            if f1 > 0:
                class_f1s[c].append(f1)
    print(f"\n**Phân tích lớp (average across all versions):**")
    for c in range(NUM_CLASSES):
        vals = class_f1s[c]
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  - {CLASS_NAMES[c]}: F1={avg:.4f} (n={len(vals)})")


if __name__ == "__main__":
    main()
