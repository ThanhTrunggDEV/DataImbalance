import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix

def calculate_metrics(y_true, y_pred, y_probs=None):
    """
    Calculate detailed metrics suitable for highly imbalanced medical datasets.
    Using Macro average is critical: it treats minority classes with equal importance.
    """
    acc = accuracy_score(y_true, y_pred)
    
    # Macro avg precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    
    metrics = {
        'accuracy': acc,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1
    }
    
    # AUC macro
    if y_probs is not None:
        try:
            auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
            metrics['auc_macro'] = auc
        except ValueError:
            # Handles edge case if some classes are missing in the batch
            metrics['auc_macro'] = float('nan')
            
    # Confusion Matrix to observe misclassification between severities
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    return metrics