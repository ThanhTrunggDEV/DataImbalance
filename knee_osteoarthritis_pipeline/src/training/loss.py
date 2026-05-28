"""
Loss functions for class-imbalance experiments.

Includes:
  - CrossEntropyLoss       (vanilla, for baseline)
  - WeightedCrossEntropy   (class-frequency-weighted CE)
  - FocalLoss              (focal term + optional class weights)
  - BalancedSoftmaxLoss    (logit-margin shift by class frequency)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss — Lin et al. (2017).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    - gamma > 0 reduces relative loss for well-classified examples,
      focusing training on hard negatives / minority classes.
    - alpha (optional) is a per-class weight tensor.
    """

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        # Register alpha as buffer so it auto-moves with .to(device)
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)                          # p_t
        focal_term = (1.0 - pt) ** self.gamma * ce_loss  # (1-p_t)³ * CE

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_term = alpha_t * focal_term

        if self.reduction == 'mean':
            return focal_term.mean()
        elif self.reduction == 'sum':
            return focal_term.sum()
        return focal_term


# ─────────────────────────────────────────────────────────────────────────────
# 2. Balanced Softmax Loss
# ─────────────────────────────────────────────────────────────────────────────

class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax Loss — Ren et al. (NeurIPS 2020).
    "Balanced Meta-Softmax for Long-Tailed Visual Recognition"

    Modifies the softmax by adding a log-frequency prior to each class logit:
        adjusted_logit_j = logit_j + log(n_j)

    where n_j is the number of training samples for class j.
    This shifts the decision boundary toward minority classes at inference time.

    Args:
        class_counts: 1-D array/tensor of per-class sample counts (train set).
    """

    def __init__(self, class_counts):
        super().__init__()
        # Register as buffer so it moves with .to(device) automatically
        counts = torch.tensor(class_counts, dtype=torch.float32)
        self.register_buffer("log_prior", torch.log(counts + 1e-8))

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Shift logits by log class frequency (log_prior is a registered buffer,
        # so it auto-moves with .to(device) — no explicit .to() needed)
        adjusted = inputs + self.log_prior
        return F.cross_entropy(adjusted, targets)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Weighted Cross-Entropy (convenience wrapper)
# ─────────────────────────────────────────────────────────────────────────────

class WeightedCrossEntropyLoss(nn.Module):
    """
    Standard Cross-Entropy with per-class weights.
    Weight for class j = (1 / n_j), normalised so they sum to num_classes.
    """

    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", class_weights)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(inputs, targets, weight=self.weight)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_loss(
    loss_type: str,
    class_weights: np.ndarray,
    class_counts: np.ndarray,
    device: torch.device,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """
    Factory function — build the correct loss based on version config.

    Args:
        loss_type:      'cross_entropy' | 'focal' | 'balanced_softmax'
        class_weights:  Inverse-freq weights per class (from dataset).
        class_counts:   Raw sample counts per class.
        device:         Target device.
        focal_gamma:    Gamma for FocalLoss.
    """
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # Normalise so mean weight == 1  (numerically stable)
    w_mean = weights_tensor.mean()
    if w_mean > 0:
        weights_tensor = weights_tensor / w_mean

    if loss_type == "cross_entropy":
        # True baseline: no weighting
        return nn.CrossEntropyLoss()

    elif loss_type == "focal":
        return FocalLoss(alpha=weights_tensor, gamma=focal_gamma)

    elif loss_type == "balanced_softmax":
        return BalancedSoftmaxLoss(class_counts=class_counts).to(device)

    else:
        raise ValueError(f"Unknown loss_type: '{loss_type}'. "
                         f"Choose from: cross_entropy, focal, balanced_softmax")