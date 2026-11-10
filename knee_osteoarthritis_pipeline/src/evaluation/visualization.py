"""
evaluation/visualization.py

All visualizations related to EVALUATION and CROSS-VERSION COMPARISON.

Functions:
  - plot_confusion_matrix          : shared by evaluator + trainer
  - plot_all_versions_comparison   : grouped bars + loss/F1 overlay curves
  - plot_per_class_comparison      : per-class F1 heatmap across all versions
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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


# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Confusion Matrix  (used by both training + evaluator)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names=None,
    save_path: str = "confusion_matrix.png",
    title_suffix: str = "",
):
    """Side-by-side raw counts + row-normalised confusion matrix heatmap."""
    if class_names is None:
        class_names = [f"Grade {i}" for i in range(cm.shape[0])]

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="g", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Counts", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1],
                vmin=0, vmax=1)
    axes[1].set_title("Row-normalised", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    suptitle = f"Confusion Matrix — {title_suffix}" if title_suffix else "Confusion Matrix"
    plt.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. All-versions comparison  (reads metrics.json from each sub-folder)
# ─────────────────────────────────────────────────────────────────────────────

def plot_all_versions_comparison(results_dir: str, version_configs: list):
    """
    Produce comparison figures across all experiment versions.

    Reads:  results_dir/<version_name>/metrics.json   (val metrics)
    Writes: results_dir/comparison/metrics_comparison.png
            results_dir/comparison/loss_f1_curves_comparison.png
    """
    comp_dir = os.path.join(results_dir, "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    # "best" is averaged across every seed_* run found on disk for the bar
    # chart; "history" (per-epoch curves) uses the first seed found only,
    # since epoch counts differ across seeds (early stopping) and can't be
    # meaningfully averaged elementwise.
    version_data = []
    for vcfg in version_configs:
        seed_metrics = _discover_seed_metrics(results_dir, vcfg["name"], "metrics.json")
        if not seed_metrics:
            print(f"  [WARN] Missing metrics.json for {vcfg['name']}, skipping.")
            continue
        bests = [data["best"] for data in seed_metrics]
        avg_best = {
            k: float(np.mean([b.get(k, 0) or 0 for b in bests]))
            for k in bests[0].keys()
        }
        version_data.append({
            "name":    vcfg["name"],
            "display": vcfg["display"],
            "best":    avg_best,
            "history": seed_metrics[0]["history"],
        })

    if not version_data:
        print("[WARN] No version data found — skipping comparison plots.")
        return

    # ── 2a. Grouped bar chart ─────────────────────────────────────────────────
    metric_keys = [
        ("val_f1_macro",         "Macro F1"),
        ("val_accuracy",         "Accuracy"),
        ("val_precision_macro",  "Macro Precision"),
        ("val_recall_macro",     "Macro Recall"),
        ("val_auc_macro",        "Macro AUC-ROC"),
    ]

    n_groups = len(metric_keys)
    n_bars   = len(version_data)
    x        = np.arange(n_groups)
    width    = 0.15

    fig, ax = plt.subplots(figsize=(16, 7))
    for i, vd in enumerate(version_data):
        values = [float(vd["best"].get(mk, 0) or 0) for mk, _ in metric_keys]
        bars = ax.bar(
            x + i * width, values, width=width,
            label=vd["display"], color=PALETTE[i % len(PALETTE)],
            edgecolor="white", linewidth=0.6,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.006,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5,
            )

    ax.set_xticks(x + width * (n_bars - 1) / 2)
    ax.set_xticklabels([label for _, label in metric_keys], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Comparison of All Versions — Best Validation Metrics",
        fontsize=14, fontweight="bold",
    )
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=10)
    plt.tight_layout()

    bar_path = os.path.join(comp_dir, "metrics_comparison.png")
    plt.savefig(bar_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {bar_path}")

    # ── 2b. Val Loss + Val F1 overlay curves ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, vd in enumerate(version_data):
        hist  = vd["history"]
        ep    = range(1, len(hist["val_loss"]) + 1)
        color = PALETTE[i % len(PALETTE)]
        axes[0].plot(ep, hist["val_loss"], "-o", markersize=3,
                     label=vd["display"], color=color)
        axes[1].plot(ep, hist["val_f1"],   "-o", markersize=3,
                     label=vd["display"], color=color)

    for ax, (ylabel, title) in zip(axes, [
        ("Loss",      "Validation Loss Curves"),
        ("Macro F1",  "Validation F1-Macro Curves"),
    ]):
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel,  fontsize=11)
        ax.set_title(title,    fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)

    plt.suptitle("All Versions — Validation Curves", fontsize=15, fontweight="bold")
    plt.tight_layout()

    curves_path = os.path.join(comp_dir, "loss_f1_curves_comparison.png")
    plt.savefig(curves_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {curves_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-class F1 heatmap across versions  (reads test_metrics.json)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_f1_heatmap(results_dir: str, version_configs: list):
    """
    Heatmap: rows = versions, columns = classes, cell = test F1 per class.
    Reads results_dir/<version>/test_metrics.json.
    """
    comp_dir = os.path.join(results_dir, "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    matrix_rows, row_labels = [], []
    class_names = None

    for vcfg in version_configs:
        seed_metrics = _discover_seed_metrics(results_dir, vcfg["name"], "test_metrics.json")
        if not seed_metrics:
            continue

        # Normalize keys ("Grade 0" → "0") and average per-class F1 across
        # every seed_* run found on disk.
        per_class_by_seed = []
        for data in seed_metrics:
            per_class_raw = data.get("per_class", {})
            per_class_by_seed.append({
                k.replace("Grade ", ""): v for k, v in per_class_raw.items()
            })

        if class_names is None:
            class_names = sorted(per_class_by_seed[0].keys(), key=int)

        row = [
            float(np.mean([
                pc.get(cn, {}).get("f1", 0) for pc in per_class_by_seed
            ]))
            for cn in class_names
        ]
        matrix_rows.append(row)
        row_labels.append(vcfg["display"])

    if not matrix_rows:
        print("  [WARN] No test_metrics.json found — skipping per-class heatmap.")
        return

    matrix = np.array(matrix_rows)

    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 1.5), len(row_labels) * 1.2 + 1.5))
    sns.heatmap(
        matrix, annot=True, fmt=".3f", cmap="RdYlGn",
        xticklabels=class_names, yticklabels=row_labels,
        vmin=0, vmax=1, linewidths=0.5, ax=ax,
    )
    ax.set_title("Per-Class F1 Score — Test Set (all versions)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Version", fontsize=11)
    plt.tight_layout()

    save_path = os.path.join(comp_dir, "per_class_f1_heatmap.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
