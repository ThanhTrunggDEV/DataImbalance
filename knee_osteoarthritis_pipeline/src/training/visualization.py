"""
training/visualization.py

Per-epoch training plots (used inside Trainer.fit).
Heavy evaluation/comparison plots live in evaluation/visualization.py.
"""

import os
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# NOTE: plot_confusion_matrix now lives in evaluation/visualization.py
# Trainer imports it directly from there to avoid circular import risk.


def plot_training_history(history: dict, version_name: str = "", save_path: str = "training_history.png"):
    """
    4-panel training history:
      [Loss]  [Accuracy]  [Precision]  [Recall + F1 overlay]
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training History — {version_name}", fontsize=15, fontweight="bold")

    panels = [
        ("Loss",             "train_loss",      "val_loss"),
        ("Accuracy",         "train_accuracy",  "val_accuracy"),
        ("Macro Precision",  "train_precision", "val_precision"),
        ("Macro Recall",     "train_recall",    "val_recall"),
    ]

    for ax, (title, train_key, val_key) in zip(axes.flat, panels):
        ax.plot(epochs, history[train_key], "b-o", markersize=4, label="Train")
        ax.plot(epochs, history[val_key],   "r-o", markersize=4, label="Val")

        if "recall" in train_key:
            ax.plot(epochs, history["train_f1"], "b--s", markersize=4, alpha=0.6, label="Train F1")
            ax.plot(epochs, history["val_f1"],   "r--s", markersize=4, alpha=0.6, label="Val F1")
            ax.set_title("Macro Recall & F1", fontsize=12)
        else:
            ax.set_title(title, fontsize=12)

        ax.set_xlabel("Epoch")
        ax.legend(fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")