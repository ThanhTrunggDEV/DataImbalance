import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

from configs.config import NUM_CLASSES


def calculate_metrics(y_true, y_pred, y_probs=None):
    """
    Calculate detailed metrics suitable for highly imbalanced medical datasets.
    Macro average treats every class with equal weight — critical for minority classes.

    Note: confusion_matrix always returns a (NUM_CLASSES x NUM_CLASSES) matrix
    via the `labels` parameter, so downstream plots are always consistent even
    when some classes don't appear in a particular split.
    """
    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    metrics = {
        'accuracy':         acc,
        'precision_macro':  precision,
        'recall_macro':     recall,
        'f1_macro':         f1,
    }

    # AUC-ROC (requires probability scores; only available at validation)
    if y_probs is not None:
        try:
            auc = roc_auc_score(
                y_true, y_probs,
                multi_class='ovr', average='macro',
                labels=list(range(NUM_CLASSES)),
            )
            metrics['auc_macro'] = auc
        except (ValueError, IndexError):
            # Edge case: some classes missing from a very small split
            metrics['auc_macro'] = float('nan')

    # Fixed-shape confusion matrix — rows=true, cols=predicted
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    metrics['confusion_matrix'] = cm

    return metrics