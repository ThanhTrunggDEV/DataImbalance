"""
paper_figures.py — Generate publication-quality figures for 6 selected versions.

Usage:
    cd knee_osteoarthritis_pipeline
    python paper_figures.py
"""

import json, os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

RESULTS_DIR = "results"
PAPER_DIR = os.path.join(os.path.dirname(__file__), "..", "paper")
os.makedirs(PAPER_DIR, exist_ok=True)

SELECTED_VERSIONS = [
    ("v1_baseline",              "Baseline (CE)"),
    ("v3_balanced_softmax",      "Balanced Softmax"),
    ("v5_focal_loss",            "Focal Loss"),
    ("v7_adjacent_balanced",     "Adjacent Mixup\n+ BalSoft"),
    ("v11_owmixup_ce_t20",       "OWMixup (CE)\nτ=2.0"),
    ("v12_owmixup_balanced_t05", "OWMixup + BalSoft\nτ=0.5"),
]

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
CLASS_NAMES = ["0", "1", "2", "3", "4"]
CLASS_LABELS = {"KOA": ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"],
                "eyepacs": ["DR 0", "DR 1", "DR 2", "DR 3", "DR 4"]}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

def load_metrics(dataset, vname):
    base = os.path.join(RESULTS_DIR, dataset, vname)
    test_path = os.path.join(base, "test_metrics.json")
    metrics_path = os.path.join(base, "metrics.json")
    data = {"test": None, "val": None, "seed_42": False}
    if not os.path.exists(test_path):
        test_path = os.path.join(base, "seed_42", "test_metrics.json")
        metrics_path = os.path.join(base, "seed_42", "metrics.json")
        if os.path.exists(test_path):
            data["seed_42"] = True
    if os.path.exists(test_path):
        with open(test_path) as f:
            data["test"] = json.load(f)
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            data["val"] = json.load(f)
    return data

def fig1_heatmap(dataset, dataset_label):
    matrix, row_labels = [], []
    for vname, vdisp in SELECTED_VERSIONS:
        d = load_metrics(dataset, vname)
        if d["test"] is None:
            continue
        pc = d["test"].get("per_class", {})
        row = []
        for cn in CLASS_NAMES:
            for k, v in pc.items():
                knorm = k.replace("Grade ", "")
                if knorm == cn:
                    row.append(v.get("f1", 0))
                    break
            else:
                row.append(0)
        matrix.append(row)
        row_labels.append(vdisp)
    if not matrix:
        return
    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="RdYlGn",
                xticklabels=CLASS_LABELS[dataset],
                yticklabels=row_labels,
                vmin=0.1, vmax=0.9, linewidths=0.8, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title(f"Per-Class F1 — {dataset_label}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Class", fontsize=10)
    ax.set_ylabel("Method", fontsize=10)
    plt.tight_layout()
    path = os.path.join(PAPER_DIR, f"fig1_heatmap_{dataset}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def fig2_barchart(dataset, dataset_label):
    metric_keys = [
        ("val_f1_macro", "Macro F1"),
        ("val_accuracy", "Accuracy"),
        ("val_auc_macro", "AUC-ROC"),
    ]
    n_metrics = len(metric_keys)
    n_vers = len(SELECTED_VERSIONS)
    x = np.arange(n_metrics)
    w = 0.13
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (vname, vdisp) in enumerate(SELECTED_VERSIONS):
        d = load_metrics(dataset, vname)
        if d["val"] is None:
            continue
        best = d["val"].get("best", {})
        vals = [float(best.get(mk, 0) or 0) for mk, _ in metric_keys]
        bars = ax.bar(x + i * w, vals, w, label=vdisp.replace("\n", " "),
                      color=COLORS[i], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x + w * (n_vers - 1) / 2)
    ax.set_xticklabels([lbl for _, lbl in metric_keys], fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(f"Validation Metrics — {dataset_label}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    path = os.path.join(PAPER_DIR, f"fig2_barchart_{dataset}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def fig3_curves(dataset, dataset_label):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for i, (vname, vdisp) in enumerate(SELECTED_VERSIONS):
        d = load_metrics(dataset, vname)
        if d["val"] is None:
            continue
        hist = d["val"].get("history", {})
        if not hist.get("val_loss"):
            continue
        ep = range(1, len(hist["val_loss"]) + 1)
        axes[0].plot(ep, hist["val_loss"], "-o", markersize=2.5,
                     label=vdisp.replace("\n", " "), color=COLORS[i], linewidth=1)
        if hist.get("val_f1"):
            axes[1].plot(ep, hist["val_f1"], "-o", markersize=2.5,
                         label=vdisp.replace("\n", " "), color=COLORS[i], linewidth=1)
    for ax, (ylbl, title) in zip(axes, [
        ("Loss", "Validation Loss"),
        ("Macro F1", "Validation F1"),
    ]):
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(ylbl, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=6.5, loc="best")
    plt.suptitle(f"Training Curves — {dataset_label}", fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(PAPER_DIR, f"fig3_curves_{dataset}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def main():
    print("Generating paper figures (6 selected versions)...")
    for dataset, label in [("KOA", "Knee OA"), ("eyepacs", "EyePACS DR")]:
        print(f"\n--- {label} ---")
        fig1_heatmap(dataset, label)
        fig2_barchart(dataset, label)
        fig3_curves(dataset, label)
    print(f"\nAll figures saved to {PAPER_DIR}")

if __name__ == "__main__":
    main()
