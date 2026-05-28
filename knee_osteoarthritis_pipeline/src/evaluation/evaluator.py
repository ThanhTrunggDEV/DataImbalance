"""
evaluation/evaluator.py

Post-training evaluation on the held-out TEST set.
Produces:
  - Per-class classification report (text + JSON)
  - Per-class precision / recall / F1 / support table
  - Confusion matrix saved to results/<version>/test_confusion_matrix.png
  - test_metrics.json  (mirrors format of training metrics.json)
"""

import json
import os
from typing import List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from .visualization import plot_confusion_matrix
from configs.config import NUM_CLASSES


# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_test(
    model,
    test_loader,
    criterion,
    device,
    save_dir: str,
    version_name: str = "experiment",
    class_names: Optional[List[str]] = None,
) -> dict:
    """
    Run the model on the test set and persist all evaluation artefacts.

    Args:
        model       : trained nn.Module (already loaded with best weights)
        test_loader : DataLoader for the test split
        criterion   : loss function (same as training)
        device      : torch.device
        save_dir    : directory where artefacts are written
        version_name: used in print headers
        class_names : list of human-readable class labels

    Returns:
        dict with scalar test metrics
    """
    os.makedirs(save_dir, exist_ok=True)

    if class_names is None:
        try:
            num_cls = len(test_loader.dataset.classes)
        except AttributeError:
            num_cls = NUM_CLASSES
        class_names = [f"Grade {i}" for i in range(num_cls)]

    # ── Inference loop ────────────────────────────────────────────────────────
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        loop = tqdm(test_loader, desc=f"[{version_name}] Test Eval", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    # ── Scalar metrics ────────────────────────────────────────────────────────
    test_loss = total_loss / len(test_loader)
    accuracy  = accuracy_score(all_labels, all_preds)

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0,
        labels=list(range(NUM_CLASSES)),
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0,
        labels=list(range(NUM_CLASSES)),
    )

    try:
        auc_macro = roc_auc_score(
            all_labels, all_probs,
            multi_class="ovr", average="macro",
            labels=list(range(NUM_CLASSES)),
        )
    except (ValueError, IndexError):
        auc_macro = float("nan")

    # Fixed-shape 5×5 confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))

    # ── Per-class table (console + text file) ─────────────────────────────────
    report_str = classification_report(
        all_labels, all_preds,
        labels=list(range(NUM_CLASSES)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    auc_str = f"{auc_macro:.4f}" if not np.isnan(auc_macro) else "N/A"
    print(f"\n  {'─'*58}")
    print(f"  TEST SET RESULTS  —  {version_name}")
    print(f"  {'─'*58}")
    print(f"  Loss: {test_loss:.4f}  |  Accuracy: {accuracy:.4f}  |  "
          f"Macro F1: {f1_macro:.4f}  |  Macro AUC: {auc_str}")
    print(f"\n{report_str}")

    report_path = os.path.join(save_dir, "test_classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Version: {version_name}\n\n")
        f.write(report_str)
    print(f"  Report saved → {report_path}")

    # ── Per-class metrics as structured dict ──────────────────────────────────
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(float(precision[i]), 6),
            "recall":    round(float(recall[i]),    6),
            "f1":        round(float(f1[i]),        6),
            "support":   int(support[i]),
        }

    test_metrics = {
        "version":             version_name,
        "test_loss":           round(test_loss,  6),
        "test_accuracy":       round(accuracy,   6),
        "test_f1_macro":       round(f1_macro,   6),
        "test_precision_macro":round(p_macro,    6),
        "test_recall_macro":   round(r_macro,    6),
        "test_auc_macro":      round(auc_macro,  6) if not np.isnan(auc_macro) else None,
        "per_class":           per_class,
    }

    metrics_path = os.path.join(save_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")

    # ── Test confusion matrix ─────────────────────────────────────────────────
    plot_confusion_matrix(
        cm,
        class_names=class_names,
        save_path=os.path.join(save_dir, "test_confusion_matrix.png"),
        title_suffix=f"Test Set — {version_name}",
    )

    return test_metrics
