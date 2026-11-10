"""
Enhanced Trainer with:
  - Standard Mixup augmentation
  - Adjacent Mixup (only mix nearby classes)
  - Rule-based Mixup (Grade 0 + partner, lam < 0.5)
  - SupCon pretraining (supervised contrastive learning)
  - Full metrics history (loss, acc, precision, recall, f1 per epoch)
  - Saves metrics.json at end of training
  - Named checkpoints per experiment version
"""

import os
import json
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from tqdm import tqdm

from .loss import BalancedSoftmaxLoss
from .metrics import calculate_metrics
from .visualization import plot_training_history
from evaluation.visualization import plot_confusion_matrix


# -----------------------------------------------------------------------------
# Mixup helpers
# -----------------------------------------------------------------------------

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """Standard Mixup -- random permutation."""
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_data_adjacent(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4, max_gap: int = 1):
    """Adjacent Mixup -- only pairs with |y_a - y_b| <= max_gap.
    Falls back to unmixed (lam=1) for samples that cannot find a valid partner."""
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    # Resample indices that violate the gap constraint (max 20 attempts)
    for _ in range(20):
        gaps = (y - y[index]).abs()
        if gaps.max() <= max_gap:
            break
        mask = gaps > max_gap
        index[mask] = torch.randint(0, batch_size, (mask.sum().item(),), device=x.device)

    # For any remaining violations, set lam=1 (no mixing)
    gaps = (y - y[index]).abs()
    valid = gaps <= max_gap
    mixed_x = x.clone()
    if valid.any():
        mixed_x[valid] = lam * x[valid] + (1.0 - lam) * x[index[valid]]
    # samples with invalid partner stay as original x

    return mixed_x, y, y[index], lam


def mixup_data_rule_based(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4, lam_max: float = 0.49):
    """
    Rule-based Mixup: modify Grade 0 images by mixing with non-zero partners.
    - Mix ~70% of Grade 0 with non-zero partners (lam < 0.5, label = partner)
    - Keep ~30% of Grade 0 clean (label = 0)
    - ALL non-zero samples stay unchanged
    Returns full batch (not subset) with per-sample modifications.
    """
    batch_size = x.size(0)
    zero_mask = (y == 0)

    lam = float(np.random.beta(alpha, alpha))
    lam = min(lam, lam_max)

    # Start with original batch: y_a = original labels, y_b = original labels
    # For clean samples: y_a == y_b -> mixup_criterion reduces to CE(original)
    # For mixed samples: y_a != y_b -> weighted CE
    mixed_x = x.clone()
    y_b = y.clone()

    if zero_mask.any():
        partner_mask = (y >= 2)  # only mix Grade 0 with {2,3,4}, exclude Grade 1
        partner_pool = torch.where(partner_mask)[0]

        if len(partner_pool) > 0:
            zero_idx = torch.where(zero_mask)[0]
            # Mix ~70% of Grade 0 samples
            n_zero = len(zero_idx)
            n_mix = max(1, int(n_zero * 0.7))
            perm = torch.randperm(n_zero, device=x.device)
            mix_idx = zero_idx[perm[:n_mix]]

            partner_indices = partner_pool[torch.randint(0, len(partner_pool), (n_mix,), device=x.device)]
            mixed_x[mix_idx] = lam * x[mix_idx] + (1.0 - lam) * x[partner_indices]
            y_b[mix_idx] = y[partner_indices]  # label = partner (majority)

    # y_a = original labels (y)
    # y_b = original for clean, partner label for mixed Grade 0
    # lam is applied as: lam * L(y_a) + (1-lam) * L(y_b)
    # For clean (y_a==y_b): lam*L + (1-lam)*L = L(original)
    # For mixed (y_a!=y_b): lam*L(0) + (1-lam)*L(partner) with lam < 0.5
    return mixed_x, y, y_b, lam


def mixup_data_ordinal_weighted(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4, temperature: float = 1.0):
    """
    Ordinal-Weighted Mixup (OWMix):
    - Pair probability proportional to Gaussian kernel over ordinal distance
    - Weighted by inverse sqrt class frequency (focus more on tail classes)
    - No self-mix (diagonal = 0)
    """
    batch_size = x.size(0)
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    # Ordinal distance kernel: P(i,j) = exp(-|yi - yj|^2 / tau^2)
    y_i = y.float().unsqueeze(0)        # [1, B]
    y_j = y.float().unsqueeze(1)        # [B, 1]
    dist = (y_i - y_j).abs()             # [B, B]
    w = torch.exp(-dist ** 2 / temperature ** 2)

    # Inverse sqrt class frequency weighting
    counts = torch.bincount(y, minlength=5).float()
    inv_freq = 1.0 / (counts.sqrt() + 1e-8)
    w_freq = inv_freq[y].unsqueeze(0) * inv_freq[y].unsqueeze(1)

    w = w * w_freq
    w.fill_diagonal_(0)  # no self-mix
    probs = w / w.sum(dim=1, keepdim=True)

    # Sample partner from categorical distribution
    index = torch.multinomial(probs, num_samples=1).squeeze(1)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_data_ordinal_weighted_queue(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    temperature: float,
    class_queue: "defaultdict[int, deque]",
    inv_freq: torch.Tensor,
):
    """
    OWMix + cross-batch memory queue:
    - Partner pool = current batch UNION a per-class CPU queue of recently seen
      samples, so a batch with zero (or one) minority-class members can still
      draw a minority partner from earlier batches.
    - inv_freq is a precomputed DATASET-level inverse-sqrt class frequency
      (not recomputed per-batch), so weighting is stable regardless of batch
      composition.
    - lam is per-sample (not a single scalar for the whole batch).
    """
    batch_size = x.size(0)
    device = x.device

    lam = torch.from_numpy(np.random.beta(alpha, alpha, size=batch_size)).float().to(device)

    # Build partner pool: current batch + queued samples (queue lives on CPU).
    pool_x_list = [x]
    pool_y_list = [y]
    for cls_queue in class_queue.values():
        for qx, qy in cls_queue:
            pool_x_list.append(qx.unsqueeze(0).to(device))
            pool_y_list.append(qy.unsqueeze(0).to(device))
    pool_x = torch.cat(pool_x_list, dim=0)
    pool_y = torch.cat(pool_y_list, dim=0)
    pool_size = pool_x.size(0)

    # Gaussian ordinal-distance kernel x inverse-sqrt frequency weighting.
    y_anchor = y.float().unsqueeze(1)          # [B, 1]
    y_pool = pool_y.float().unsqueeze(0)       # [1, P]
    dist = (y_anchor - y_pool).abs()
    w = torch.exp(-dist ** 2 / temperature ** 2)
    w = w * inv_freq[y].unsqueeze(1) * inv_freq[pool_y].unsqueeze(0)

    # No self-mix: zero out the diagonal within the batch region of the pool.
    self_idx = torch.arange(batch_size, device=device)
    w[self_idx, self_idx] = 0.0

    probs = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    index = torch.multinomial(probs, num_samples=1).squeeze(1)

    lam_view = lam.view(-1, 1, 1, 1)
    mixed_x = lam_view * x + (1.0 - lam_view) * pool_x[index]
    y_b = pool_y[index]

    # Enqueue current batch (CPU, detached) for future batches to draw on.
    x_cpu = x.detach().to("cpu")
    y_cpu = y.detach().to("cpu")
    for i in range(batch_size):
        cls = int(y_cpu[i].item())
        class_queue[cls].append((x_cpu[i], y_cpu[i]))

    return mixed_x, y, y_b, lam


def _reaugment_cached_tensor(t: torch.Tensor) -> torch.Tensor:
    """
    Cheap tensor-level re-randomization applied every time a cached queue
    sample is drawn as a mixup partner (v14). Operates directly on the
    already-augmented/normalized tensor (no PIL image needed), so it works
    identically regardless of image domain (grayscale X-ray, color fundus,
    ...) and needs no dataset-specific tuning. This ensures a single cached
    sample never contributes identical pixels on every reuse, which is the
    root cause of v13's cross-batch queue underperforming even the no-mixup
    baseline (see mixup_data_ordinal_weighted_queue docstring).
    """
    if torch.rand(1).item() < 0.5:
        t = torch.flip(t, dims=[-1])
    angle = float(torch.empty(1).uniform_(-10, 10))
    t = TF.rotate(t, angle)
    brightness_scale = float(torch.empty(1).uniform_(0.9, 1.1))
    return t * brightness_scale


def mixup_data_ordinal_weighted_queue_v2(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    temperature: float,
    class_queue: "defaultdict[int, deque]",
    inv_freq: torch.Tensor,
    max_reuse: int,
):
    """
    OWMix + bounded, re-augmented, rescue-only memory queue (v14).

    Fixes the three failure modes identified in v13's plain memory queue
    (mixup_data_ordinal_weighted_queue), all via dataset-derived quantities so
    the same version runs unchanged across datasets (KOA, EyePACS, ...):
      1. Bounded reuse: each queue slot carries a `uses_remaining` counter
         (from `max_reuse`) and is evicted once exhausted, so no single cached
         image can imprint on the model an unbounded number of times.
      2. Re-augmented on draw: `_reaugment_cached_tensor` is applied to every
         queued sample each time the pool is built, so repeated draws of the
         same slot never contribute identical pixels.
      3. Rescue-only gating: the queue only supplements anchors whose class is
         under-represented in the *current* batch (fewer than a fair share of
         `batch_size / num_classes`); well-represented anchors mix purely
         in-batch, exactly like v11/v12 (which already outperform baseline).
    """
    batch_size = x.size(0)
    device = x.device
    num_classes = inv_freq.numel()

    lam = torch.from_numpy(np.random.beta(alpha, alpha, size=batch_size)).float().to(device)

    # Build this batch's partner pool: current batch + a re-augmented snapshot
    # of every queued sample. `queue_origin` records where each queue-region
    # pool slot came from so we can bound its reuse count after selection.
    pool_x_list = [x]
    pool_y_list = [y]
    queue_origin = []
    for cls, cls_queue in class_queue.items():
        for i, (qx, qy, _uses) in enumerate(cls_queue):
            pool_x_list.append(_reaugment_cached_tensor(qx).unsqueeze(0).to(device))
            pool_y_list.append(qy.unsqueeze(0).to(device))
            queue_origin.append((cls, i))
    pool_x = torch.cat(pool_x_list, dim=0)
    pool_y = torch.cat(pool_y_list, dim=0)

    # Gaussian ordinal-distance kernel x inverse-sqrt frequency weighting.
    y_anchor = y.float().unsqueeze(1)          # [B, 1]
    y_pool = pool_y.float().unsqueeze(0)       # [1, P]
    dist = (y_anchor - y_pool).abs()
    w = torch.exp(-dist ** 2 / temperature ** 2)
    w = w * inv_freq[y].unsqueeze(1) * inv_freq[pool_y].unsqueeze(0)

    self_idx = torch.arange(batch_size, device=device)
    w[self_idx, self_idx] = 0.0

    # Rescue-only gating: anchors with at least a fair share of same-class
    # partners already in the batch may not draw from the queue region.
    fair_share = -(-batch_size // num_classes)  # ceil division, no extra import
    counts_in_batch = torch.bincount(y, minlength=num_classes).to(device)
    rescue_needed = counts_in_batch[y] < fair_share  # [B] bool
    if pool_x.size(0) > batch_size:
        pool_positions = torch.arange(pool_x.size(0), device=device)
        queue_region = (pool_positions >= batch_size).unsqueeze(0)      # [1, P]
        no_rescue = (~rescue_needed).unsqueeze(1)                       # [B, 1]
        w = w.masked_fill(no_rescue & queue_region, 0.0)

    probs = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    index = torch.multinomial(probs, num_samples=1).squeeze(1)

    lam_view = lam.view(-1, 1, 1, 1)
    mixed_x = lam_view * x + (1.0 - lam_view) * pool_x[index]
    y_b = pool_y[index]

    # Bound reuse: decrement/evict any queue slot that was actually drawn.
    if queue_origin:
        affected_classes = set()
        chosen = (index[index >= batch_size] - batch_size).tolist()
        for pos in chosen:
            cls, deque_idx = queue_origin[pos]
            class_queue[cls][deque_idx][2] -= 1
            affected_classes.add(cls)
        for cls in affected_classes:
            class_queue[cls] = deque(
                (slot for slot in class_queue[cls] if slot[2] > 0),
                maxlen=class_queue[cls].maxlen,
            )

    # Enqueue current batch (fresh reuse budget) for future batches to draw on.
    x_cpu = x.detach().to("cpu")
    y_cpu = y.detach().to("cpu")
    for i in range(batch_size):
        cls = int(y_cpu[i].item())
        class_queue[cls].append([x_cpu[i], y_cpu[i], max_reuse])

    return mixed_x, y, y_b, lam


def mixup_data_ordinal_weighted_queue_v3(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    temperature: float,
    class_queue: "defaultdict[int, deque]",
    inv_freq: torch.Tensor,
    max_reuse: int,
):
    """
    OWMix + bounded/re-augmented queue, with a rescue-gate derived from the
    TRUE per-class training proportions instead of an assumed-uniform 1/K
    split (v15).

    v14 (mixup_data_ordinal_weighted_queue_v2) gated rescue on
    `counts_in_batch[y] < ceil(batch_size / num_classes)`, i.e. it assumed
    every class "should" fill an equal share of each batch. That assumption
    holds when a WeightedRandomSampler is active (CE variants), but is wrong
    by construction for BalancedSoftmax variants, which deliberately run
    WITHOUT a sampler so the natural (skewed) distribution reaches the loss.
    On a mildly-imbalanced dataset (KOA) the two thresholds are close, so v14
    worked. On a heavily-skewed one (EyePACS, ~35x between rarest and most
    common class) the uniform 1/K bar is almost never met by minority
    classes, so rescue fired on nearly every batch — flooding training with
    queue-based mixing and eroding majority-class recall (observed: v14 test
    QWK/MAE clearly worse than baseline on EyePACS, while beating baseline on
    KOA).

    Fix: derive the expected per-class batch count from the dataset's actual
    class proportions (recovered from `inv_freq = 1/sqrt(class_counts)`,
    already computed per-dataset) instead of assuming 1/num_classes. Rescue
    now only fires when a class is under-drawn relative to its OWN natural
    share of a batch this size — a true "bad luck" safety net rather than a
    forced rebalancing tool — so it self-calibrates to any dataset's true
    imbalance ratio without a new constant or a use_sampler branch.
    """
    batch_size = x.size(0)
    device = x.device
    num_classes = inv_freq.numel()

    lam = torch.from_numpy(np.random.beta(alpha, alpha, size=batch_size)).float().to(device)

    pool_x_list = [x]
    pool_y_list = [y]
    queue_origin = []
    for cls, cls_queue in class_queue.items():
        for i, (qx, qy, _uses) in enumerate(cls_queue):
            pool_x_list.append(_reaugment_cached_tensor(qx).unsqueeze(0).to(device))
            pool_y_list.append(qy.unsqueeze(0).to(device))
            queue_origin.append((cls, i))
    pool_x = torch.cat(pool_x_list, dim=0)
    pool_y = torch.cat(pool_y_list, dim=0)

    y_anchor = y.float().unsqueeze(1)
    y_pool = pool_y.float().unsqueeze(0)
    dist = (y_anchor - y_pool).abs()
    w = torch.exp(-dist ** 2 / temperature ** 2)
    w = w * inv_freq[y].unsqueeze(1) * inv_freq[pool_y].unsqueeze(0)

    self_idx = torch.arange(batch_size, device=device)
    w[self_idx, self_idx] = 0.0

    # Rescue-only gating (v15): expected count under the dataset's true
    # per-class proportions, recovered from inv_freq (= 1/sqrt(counts)).
    class_props = 1.0 / inv_freq.clamp_min(1e-8) ** 2
    class_props = class_props / class_props.sum()
    expected_count = class_props * batch_size                      # [num_classes]
    counts_in_batch = torch.bincount(y, minlength=num_classes).to(device)
    rescue_needed = counts_in_batch[y].float() < expected_count[y]  # [B] bool
    if pool_x.size(0) > batch_size:
        pool_positions = torch.arange(pool_x.size(0), device=device)
        queue_region = (pool_positions >= batch_size).unsqueeze(0)
        no_rescue = (~rescue_needed).unsqueeze(1)
        w = w.masked_fill(no_rescue & queue_region, 0.0)

    probs = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    index = torch.multinomial(probs, num_samples=1).squeeze(1)

    lam_view = lam.view(-1, 1, 1, 1)
    mixed_x = lam_view * x + (1.0 - lam_view) * pool_x[index]
    y_b = pool_y[index]

    if queue_origin:
        affected_classes = set()
        chosen = (index[index >= batch_size] - batch_size).tolist()
        for pos in chosen:
            cls, deque_idx = queue_origin[pos]
            class_queue[cls][deque_idx][2] -= 1
            affected_classes.add(cls)
        for cls in affected_classes:
            class_queue[cls] = deque(
                (slot for slot in class_queue[cls] if slot[2] > 0),
                maxlen=class_queue[cls].maxlen,
            )

    x_cpu = x.detach().to("cpu")
    y_cpu = y.detach().to("cpu")
    for i in range(batch_size):
        cls = int(y_cpu[i].item())
        class_queue[cls].append([x_cpu[i], y_cpu[i], max_reuse])

    return mixed_x, y, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixed loss: lam * L(y_a) + (1-lam) * L(y_b)."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def mixup_data_ordinal_weighted_queue_v4(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
    temperature: float,
    class_queue: "defaultdict[int, deque]",
    inv_freq: torch.Tensor,
    max_reuse: int,
):
    """
    OWMix + bounded/re-augmented queue with FREQUENCY-MODULATED MIXING (v16).

    v13-v15 all lost to the CE baseline on EyePACS for one root reason: every
    variant mixed majority-class anchors at full strength too, dragging the
    dominant class (DR0, F1 0.900 -> ~0.84) down far enough to cancel every
    minority-class gain in the equally-weighted macro F1. On a large dataset
    the majority class needs no augmentation and is only harmed by it.

    v16 fixes this WITHOUT a hard threshold (which would reintroduce exactly
    the discontinuous design OWMixup was built to avoid). Instead, each
    anchor's mixing INTENSITY is modulated continuously by its class rarity,
    reusing the same 1/sqrt(class_count) inverse-frequency term that already
    drives OWMixup's partner selection: s_i = inv_freq[y_i] / max(inv_freq) in
    (0, 1], normalized so the rarest class gets s=1 (full mixing) and the most
    common is mixed only weakly (its clean signal is preserved). The effective
    per-sample coefficient becomes lam_eff = 1 - s_i * (1 - lam). This is a
    natural extension of OWMixup: rarity now weights not only WHICH partner is
    drawn but HOW STRONGLY the anchor is mixed, and it self-calibrates to any
    dataset's imbalance ratio with no new constant.
    """
    batch_size = x.size(0)
    device = x.device

    lam = torch.from_numpy(np.random.beta(alpha, alpha, size=batch_size)).float().to(device)

    # Partner pool: current batch + re-augmented queued samples.
    pool_x_list = [x]
    pool_y_list = [y]
    queue_origin = []
    for cls, cls_queue in class_queue.items():
        for i, (qx, qy, _uses) in enumerate(cls_queue):
            pool_x_list.append(_reaugment_cached_tensor(qx).unsqueeze(0).to(device))
            pool_y_list.append(qy.unsqueeze(0).to(device))
            queue_origin.append((cls, i))
    pool_x = torch.cat(pool_x_list, dim=0)
    pool_y = torch.cat(pool_y_list, dim=0)

    y_anchor = y.float().unsqueeze(1)
    y_pool = pool_y.float().unsqueeze(0)
    dist = (y_anchor - y_pool).abs()
    w = torch.exp(-dist ** 2 / temperature ** 2)
    w = w * inv_freq[y].unsqueeze(1) * inv_freq[pool_y].unsqueeze(0)

    self_idx = torch.arange(batch_size, device=device)
    w[self_idx, self_idx] = 0.0

    probs = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    index = torch.multinomial(probs, num_samples=1).squeeze(1)

    # Frequency-modulated mixing intensity (continuous, no hard threshold):
    # a sample participates in mixup in proportion to its class rarity, using
    # the same 1/sqrt(n) inverse-frequency term that drives partner selection.
    # s in (0, 1], normalized so the rarest class gets s=1 (full mixing) and
    # the most common is mixed only weakly (clean signal preserved).
    s = (inv_freq[y] / inv_freq.max()).clamp(0.0, 1.0)     # [B]
    lam = 1.0 - s * (1.0 - lam)                            # per-sample lam_eff

    lam_view = lam.view(-1, 1, 1, 1)
    mixed_x = lam_view * x + (1.0 - lam_view) * pool_x[index]
    y_b = pool_y[index]

    # Bounded reuse: decrement/evict queue slots actually drawn as partners.
    # Majority anchors keep lam_eff ~= 1 so contribute ~0 partner weight, but
    # still technically consume the slot they sampled -- count it uniformly.
    if queue_origin:
        affected_classes = set()
        chosen = (index[index >= batch_size] - batch_size).tolist()
        for pos in chosen:
            cls, deque_idx = queue_origin[pos]
            class_queue[cls][deque_idx][2] -= 1
            affected_classes.add(cls)
        for cls in affected_classes:
            class_queue[cls] = deque(
                (slot for slot in class_queue[cls] if slot[2] > 0),
                maxlen=class_queue[cls].maxlen,
            )

    x_cpu = x.detach().to("cpu")
    y_cpu = y.detach().to("cpu")
    for i in range(batch_size):
        cls = int(y_cpu[i].item())
        class_queue[cls].append([x_cpu[i], y_cpu[i], max_reuse])

    return mixed_x, y, y_b, lam


def mixup_criterion_queue(criterion, pred, y_a, y_b, lam):
    """
    Compute mixed loss for per-sample lam (a tensor, not a scalar).
    Needs per-sample (unreduced) loss, which nn.CrossEntropyLoss() and
    BalancedSoftmaxLoss don't expose by default, so it's recomputed here
    rather than calling `criterion` directly.
    """
    if isinstance(criterion, BalancedSoftmaxLoss):
        adjusted = pred + criterion.log_prior
        loss_a = F.cross_entropy(adjusted, y_a, reduction="none")
        loss_b = F.cross_entropy(adjusted, y_b, reduction="none")
    else:
        loss_a = F.cross_entropy(pred, y_a, reduction="none")
        loss_b = F.cross_entropy(pred, y_b, reduction="none")
    return (lam * loss_a + (1.0 - lam) * loss_b).mean()


# -----------------------------------------------------------------------------
# OWMM — Ordinal-Weighted Manifold Mixup (v17)
# -----------------------------------------------------------------------------
# v11-v16 all interpolate in PIXEL space, then feed the blend through the model.
# On fundus images an alpha-blend of two retinas is anatomically meaningless, so
# the "regularization" signal is largely noise and the only real effect left is
# the (zero-sum-with-the-sampler) implicit resampling from partner selection —
# which is why every variant merely shifted F1 between classes instead of
# raising it. OWMM keeps OWMix's ordinal-distance x tail-frequency partner
# selection (the contribution) but moves the interpolation to the layer3 feature
# manifold, where linear interpolation is semantically valid, and supervises
# with a soft target smoothed along the ordinal grade axis. Three baked-in
# design decisions (not swept):
#   1. Manifold (layer3) mixing instead of pixel mixing.
#   2. Effective-number tail reweight (Cui et al. 2019) for partner selection,
#      strong enough for EyePACS's ~35x imbalance where 1/sqrt(count) is not.
#   3. Sampler OFF (set in the version config) so partner reweighting is the
#      sole balancer — removes the double-correction that sank majority F1.


def owmm_effective_number_weights(class_counts, beta: float, device):
    """Per-class partner weight = (1 - beta) / (1 - beta^n_c), normalized to
    mean 1. Rare classes (small n_c) get a large weight; the majority class
    saturates toward (1 - beta). Recovers uniform weighting as beta -> 0."""
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    eff_num = 1.0 - torch.pow(torch.tensor(beta, dtype=torch.float32), counts)
    w = (1.0 - beta) / eff_num.clamp_min(1e-12)
    w = w / w.mean().clamp_min(1e-12)
    return w.to(device)


def owmm_scarcity_scale(class_counts, beta: float, device, scale_min: float = 0.3):
    """Per-class mixing-strength scale in [scale_min, 1], derived from the same
    effective-number weight used for partner selection (mean-normalized to 1).

    Majority classes (small effective-number weight) clamp toward scale_min so
    they are mixed only lightly — protecting their clean representation and
    accuracy — while rare classes (weight >= 1) clamp to 1 and receive the full
    mixing strength. Because the weight spread widens with imbalance severity,
    the modulation self-adapts: on a mildly imbalanced set (KOA, ~13x) the
    majority scale stays near 1 (behaves like plain OWMM), whereas on a heavily
    imbalanced set (EyePACS, ~36x) the 73%-majority is strongly protected."""
    cw = owmm_effective_number_weights(class_counts, beta, device)
    return cw.clamp(min=scale_min, max=1.0)


def owmm_select_partners(y, alpha, temperature, class_weight, mix_scale=None):
    """OWMix partner selection (in-batch, no queue): partner j for anchor i is
    sampled proportional to exp(-|y_i - y_j|^2 / tau^2) * cw[y_i] * cw[y_j],
    with no self-mix. Returns (index, y_b, per-sample lam).

    When `mix_scale` (a per-class tensor) is given, the per-sample mixing
    strength is scaled by the anchor class's scarcity: lam_i is pushed toward 1
    (less partner content) for majority anchors and left at full strength for
    rare anchors. `mix_scale=None` reproduces the original v17 behavior."""
    batch_size = y.size(0)
    device = y.device
    lam = torch.from_numpy(np.random.beta(alpha, alpha, size=batch_size)).float().to(device)

    if mix_scale is not None:
        # partner fraction (1 - lam) shrinks for majority anchors:
        # lam_i = 1 - (1 - lam_i) * scale[y_i]
        lam = 1.0 - (1.0 - lam) * mix_scale[y]

    y_i = y.float().unsqueeze(0)
    y_j = y.float().unsqueeze(1)
    dist = (y_i - y_j).abs()
    w = torch.exp(-dist ** 2 / temperature ** 2)
    cw = class_weight[y]
    w = w * cw.unsqueeze(0) * cw.unsqueeze(1)
    w.fill_diagonal_(0.0)

    probs = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
    index = torch.multinomial(probs, num_samples=1).squeeze(1)
    return index, y[index], lam


def owmm_manifold_forward(model, x, index, lam):
    """Forward `x` through a torchvision ResNet backbone, mixing the layer3
    feature maps of each sample with its selected partner (per-sample lam)
    before layer4. Assumes the ResNet50 stage layout used by
    models/resnet.py (all experiment versions use it)."""
    lam_view = lam.view(-1, 1, 1, 1)
    h = model.conv1(x)
    h = model.bn1(h)
    h = model.relu(h)
    h = model.maxpool(h)
    h = model.layer1(h)
    h = model.layer2(h)
    h = model.layer3(h)
    h = lam_view * h + (1.0 - lam_view) * h[index]   # manifold mix @ layer3
    h = model.layer4(h)
    h = model.avgpool(h)
    h = torch.flatten(h, 1)
    return model.fc(h)


def owmm_ordinal_criterion(criterion, logits, y_a, y_b, lam, num_classes, ord_sigma):
    """Soft-CE against a mixup target that is additionally smoothed along the
    ordinal grade axis by a Gaussian kernel (width `ord_sigma`). With
    ord_sigma -> 0 the target is the two-hot mixup target and this reduces to
    the standard mixup criterion; ord_sigma > 0 spreads a little mass to
    neighbouring grades, teaching the ordinal structure and softening the noisy
    grade-1 boundary. Adds the BalancedSoftmax log-prior to the logits when
    that loss is in use, mirroring mixup_criterion_queue."""
    device = logits.device
    onehot_a = F.one_hot(y_a, num_classes).float()
    onehot_b = F.one_hot(y_b, num_classes).float()
    lam_c = lam.unsqueeze(1)
    target = lam_c * onehot_a + (1.0 - lam_c) * onehot_b       # [B, C]

    if ord_sigma and ord_sigma > 0:
        cls = torch.arange(num_classes, device=device).float()
        kern = torch.exp(-(cls.unsqueeze(0) - cls.unsqueeze(1)) ** 2 / (2.0 * ord_sigma ** 2))
        kern = kern / kern.sum(dim=1, keepdim=True)            # row-normalized [C, C]
        target = target @ kern                                # rows still sum to 1

    if isinstance(criterion, BalancedSoftmaxLoss):
        logits = logits + criterion.log_prior
    log_probs = F.log_softmax(logits, dim=1)
    return -(target * log_probs).sum(dim=1).mean()


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        version_name: str = "experiment",
        lr_scheduler=None,
        early_stopping_patience: int = 7,
        use_mixup: bool = False,
        mixup_mode: str = "standard",
        mixup_alpha: float = 0.4,
        mixup_adjacent_gap: int = 1,
        mixup_rule_lam_max: float = 0.49,
        mixup_temperature: float = 1.0,
        mixup_queue_size: int = 64,
        mixup_queue_max_reuse: int = 3,
        dataset_class_counts=None,
        owmm_effnum_beta: float = 0.9999,
        owmm_ord_sigma: float = 0.5,
        owmm_mix_scale_min: float = 0.3,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.version_name = version_name
        self.lr_scheduler = lr_scheduler
        self.early_stopping_patience = early_stopping_patience
        self.use_mixup = use_mixup
        self.mixup_mode = mixup_mode
        self.mixup_alpha = mixup_alpha
        self.mixup_adjacent_gap = mixup_adjacent_gap
        self.mixup_rule_lam_max = mixup_rule_lam_max
        self.mixup_temperature = mixup_temperature
        self.mixup_queue_max_reuse = mixup_queue_max_reuse

        # OWMix + memory queue (v13): per-class CPU queue of recent samples,
        # used as extra mixing partners so minority classes remain reachable
        # even when a mini-batch contains none of them.
        self.minority_queue = defaultdict(lambda: deque(maxlen=mixup_queue_size))
        # OWMix + bounded/re-augmented/rescue-only queue (v14) — kept separate
        # from v13's queue since slot format differs ([x, y, uses_remaining]).
        self.minority_queue_v2 = defaultdict(lambda: deque(maxlen=mixup_queue_size))
        # v15: same slot format as v14, but a separate queue instance so the
        # two versions never share state when compared in the same process.
        self.minority_queue_v3 = defaultdict(lambda: deque(maxlen=mixup_queue_size))
        # v16: frequency-modulated mixing queue (separate instance / slot state).
        self.minority_queue_v4 = defaultdict(lambda: deque(maxlen=mixup_queue_size))
        if dataset_class_counts is not None:
            counts = torch.as_tensor(dataset_class_counts, dtype=torch.float32)
            self.inv_freq = (1.0 / (counts.sqrt() + 1e-8)).to(device)
        else:
            self.inv_freq = None

        # OWMM (v17): effective-number partner weight + soft-ordinal target width.
        self.owmm_ord_sigma = owmm_ord_sigma
        if dataset_class_counts is not None:
            self.num_classes = int(torch.as_tensor(dataset_class_counts).numel())
            self.owmm_class_weight = owmm_effective_number_weights(
                dataset_class_counts, owmm_effnum_beta, device
            )
            # v18: per-class mixing-strength scale (scarcity-adaptive intensity).
            self.owmm_mix_scale = owmm_scarcity_scale(
                dataset_class_counts, owmm_effnum_beta, device, owmm_mix_scale_min
            )
        else:
            self.num_classes = None
            self.owmm_class_weight = None
            self.owmm_mix_scale = None

    # -- Single epoch helpers --------------------------------------------------

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        loop = tqdm(self.train_loader, desc=f"[{self.version_name}] Train", leave=False)
        for images, labels in loop:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            batch_labels = labels

            if self.use_mixup and self.mixup_mode == "adjacent":
                mixed_x, y_a, y_b, lam = mixup_data_adjacent(
                    images, labels, self.mixup_alpha, self.mixup_adjacent_gap
                )
                outputs = self.model(mixed_x)
                loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)

            elif self.use_mixup and self.mixup_mode == "rule":
                mixed_x, y_a, y_b, lam = mixup_data_rule_based(
                    images, labels, self.mixup_alpha, self.mixup_rule_lam_max
                )
                outputs = self.model(mixed_x)
                loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)
                batch_labels = y_a

            elif self.use_mixup and self.mixup_mode == "ordinal_weighted":
                mixed_x, y_a, y_b, lam = mixup_data_ordinal_weighted(
                    images, labels, self.mixup_alpha, self.mixup_temperature
                )
                outputs = self.model(mixed_x)
                loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)

            elif self.use_mixup and self.mixup_mode == "ordinal_weighted_queue":
                mixed_x, y_a, y_b, lam = mixup_data_ordinal_weighted_queue(
                    images, labels, self.mixup_alpha, self.mixup_temperature,
                    self.minority_queue, self.inv_freq,
                )
                outputs = self.model(mixed_x)
                loss = mixup_criterion_queue(self.criterion, outputs, y_a, y_b, lam)

            elif self.use_mixup and self.mixup_mode == "ordinal_weighted_queue_v2":
                mixed_x, y_a, y_b, lam = mixup_data_ordinal_weighted_queue_v2(
                    images, labels, self.mixup_alpha, self.mixup_temperature,
                    self.minority_queue_v2, self.inv_freq, self.mixup_queue_max_reuse,
                )
                outputs = self.model(mixed_x)
                loss = mixup_criterion_queue(self.criterion, outputs, y_a, y_b, lam)

            elif self.use_mixup and self.mixup_mode == "ordinal_weighted_queue_v3":
                mixed_x, y_a, y_b, lam = mixup_data_ordinal_weighted_queue_v3(
                    images, labels, self.mixup_alpha, self.mixup_temperature,
                    self.minority_queue_v3, self.inv_freq, self.mixup_queue_max_reuse,
                )
                outputs = self.model(mixed_x)
                loss = mixup_criterion_queue(self.criterion, outputs, y_a, y_b, lam)

            elif self.use_mixup and self.mixup_mode == "ordinal_weighted_queue_v4":
                mixed_x, y_a, y_b, lam = mixup_data_ordinal_weighted_queue_v4(
                    images, labels, self.mixup_alpha, self.mixup_temperature,
                    self.minority_queue_v4, self.inv_freq, self.mixup_queue_max_reuse,
                )
                outputs = self.model(mixed_x)
                loss = mixup_criterion_queue(self.criterion, outputs, y_a, y_b, lam)

            elif self.use_mixup and self.mixup_mode == "ordinal_manifold":
                index, y_b, lam = owmm_select_partners(
                    labels, self.mixup_alpha, self.mixup_temperature, self.owmm_class_weight
                )
                outputs = owmm_manifold_forward(self.model, images, index, lam)
                loss = owmm_ordinal_criterion(
                    self.criterion, outputs, labels, y_b, lam,
                    self.num_classes, self.owmm_ord_sigma,
                )

            elif self.use_mixup and self.mixup_mode == "ordinal_manifold_adaptive":
                # v18: OWMM with scarcity-adaptive per-sample mixing strength —
                # majority anchors are mixed lightly (protect accuracy on heavily
                # imbalanced sets), rare anchors keep full mixing.
                index, y_b, lam = owmm_select_partners(
                    labels, self.mixup_alpha, self.mixup_temperature,
                    self.owmm_class_weight, self.owmm_mix_scale,
                )
                outputs = owmm_manifold_forward(self.model, images, index, lam)
                loss = owmm_ordinal_criterion(
                    self.criterion, outputs, labels, y_b, lam,
                    self.num_classes, self.owmm_ord_sigma,
                )

            elif self.use_mixup:
                # Standard mixup (default)
                mixed_x, y_a, y_b, lam = mixup_data(images, labels, self.mixup_alpha)
                outputs = self.model(mixed_x)
                loss = mixup_criterion(self.criterion, outputs, y_a, y_b, lam)

            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

            loop.set_postfix(loss=f"{loss.item():.4f}")

        metrics = calculate_metrics(all_labels, all_preds)
        return total_loss / len(self.train_loader), metrics

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc=f"[{self.version_name}] Val  ", leave=False)
            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        metrics = calculate_metrics(all_labels, all_preds, np.array(all_probs))
        return total_loss / len(self.val_loader), metrics

    # -- Main training loop ----------------------------------------------------

    def fit(self, epochs: int, save_dir: str = "../results/experiment"):
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, "best_model.pth")

        best_f1 = 0.0
        best_cm = None
        patience_counter = 0

        # Full history for all tracked metrics
        history = {
            "train_loss": [], "val_loss": [],
            "train_accuracy": [], "val_accuracy": [],
            "train_precision": [], "val_precision": [],
            "train_recall": [], "val_recall": [],
            "train_f1": [], "val_f1": [],
            "train_qwk": [], "val_qwk": [],
            "train_mae": [], "val_mae": [],
        }
        best_metrics_snapshot = {}

        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"  [{self.version_name}]  Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")

            train_loss, train_m = self.train_epoch()
            val_loss,   val_m   = self.validate()

            # -- Record history ------------------------------------------------
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_accuracy"].append(train_m["accuracy"])
            history["val_accuracy"].append(val_m["accuracy"])
            history["train_precision"].append(train_m["precision_macro"])
            history["val_precision"].append(val_m["precision_macro"])
            history["train_recall"].append(train_m["recall_macro"])
            history["val_recall"].append(val_m["recall_macro"])
            history["train_f1"].append(train_m["f1_macro"])
            history["val_f1"].append(val_m["f1_macro"])
            history["train_qwk"].append(train_m.get("qwk", 0))
            history["val_qwk"].append(val_m.get("qwk", 0))
            history["train_mae"].append(train_m.get("mae", 0))
            history["val_mae"].append(val_m.get("mae", 0))

            # -- LR scheduler --------------------------------------------------
            if self.lr_scheduler:
                # ReduceLROnPlateau needs the monitored metric; epoch-based
                # schedulers (Cosine, SequentialLR warmup+cosine) step blind.
                if isinstance(self.lr_scheduler,
                              torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            # -- Console summary -----------------------------------------------
            auc_str = f"{val_m['auc_macro']:.4f}" if 'auc_macro' in val_m and not np.isnan(val_m['auc_macro']) else "N/A "
            qwk_str = f"{val_m['qwk']:.4f}" if 'qwk' in val_m else "N/A "
            mae_str = f"{val_m['mae']:.4f}" if 'mae' in val_m else "N/A "
            print(f"  Train  | Loss: {train_loss:.4f} | Acc: {train_m['accuracy']:.4f} | "
                  f"P: {train_m['precision_macro']:.4f} | R: {train_m['recall_macro']:.4f} | "
                  f"F1: {train_m['f1_macro']:.4f} | QWK: {qwk_str}")
            print(f"  Val    | Loss: {val_loss:.4f} | Acc: {val_m['accuracy']:.4f} | "
                  f"P: {val_m['precision_macro']:.4f} | R: {val_m['recall_macro']:.4f} | "
                  f"F1: {val_m['f1_macro']:.4f} | AUC: {auc_str} | QWK: {qwk_str} | MAE: {mae_str}")

            # -- Checkpointing (best val F1) -----------------------------------
            if val_m["f1_macro"] > best_f1:
                best_f1 = val_m["f1_macro"]
                best_cm = val_m["confusion_matrix"]
                torch.save(self.model.state_dict(), model_path)
                print(f"  v Saved best model  ->  Val F1 = {best_f1:.4f}")
                patience_counter = 0
                best_metrics_snapshot = {
                    "epoch": epoch + 1,
                    "val_loss": round(val_loss, 6),
                    "val_accuracy": round(val_m["accuracy"], 6),
                    "val_precision_macro": round(val_m["precision_macro"], 6),
                    "val_recall_macro": round(val_m["recall_macro"], 6),
                    "val_f1_macro": round(val_m["f1_macro"], 6),
                    # None -> JSON null (NaN is invalid JSON)
                    "val_auc_macro": (
                        round(val_m["auc_macro"], 6)
                        if "auc_macro" in val_m and not np.isnan(val_m["auc_macro"])
                        else None
                    ),
                }
            else:
                patience_counter += 1
                print(f"  EarlyStopping: {patience_counter}/{self.early_stopping_patience}")

            if patience_counter >= self.early_stopping_patience:
                print(f"\n  [!] Early stopping triggered at epoch {epoch+1}.")
                break

        # -- Post-training: save artefacts -------------------------------------
        print(f"\n  Saving artefacts to  {save_dir} ...")

        # Metrics JSON
        output = {
            "version": self.version_name,
            "total_epochs_run": len(history["train_loss"]),
            "best": best_metrics_snapshot,
            "history": {k: [round(v, 6) for v in vals] for k, vals in history.items()},
        }
        with open(os.path.join(save_dir, "metrics.json"), "w") as f:
            json.dump(output, f, indent=2)

        # Plots
        plot_training_history(
            history,
            version_name=self.version_name,
            save_path=os.path.join(save_dir, "training_history.png"),
        )
        if best_cm is not None:
            plot_confusion_matrix(
                best_cm,
                save_path=os.path.join(save_dir, "confusion_matrix.png"),
            )

        print(f"  Done! Best Val F1 = {best_f1:.4f}  (epoch {best_metrics_snapshot.get('epoch', '?')})")
        return history, best_metrics_snapshot

    # -- SupCon pretraining (Stage 1) ------------------------------------------

    def train_supcon_epoch(self, train_loader):
        """Single epoch for SupCon pretraining -- handles (x1, x2, labels) triplets."""
        self.model.train()
        total_loss = 0.0

        loop = tqdm(train_loader, desc=f"[{self.version_name}] SupCon Train", leave=False)
        for x1, x2, labels in loop:
            x1, x2, labels = x1.to(self.device), x2.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            # Forward both views through backbone + projection head
            z1 = self.model(x1)  # [B, D]
            z2 = self.model(x2)  # [B, D]
            features = torch.cat([z1, z2], dim=0)  # [2*B, D]

            loss = self.criterion(features, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(train_loader)

    def fit_supcon(self, epochs: int, supcon_loader, save_dir: str = "../results/experiment"):
        """
        SupCon pretraining loop (Stage 1).
        Saves backbone weights to 'backbone_supcon.pth' after training.
        """
        os.makedirs(save_dir, exist_ok=True)

        history = {"supcon_loss": []}

        for epoch in range(epochs):
            print(f"\n{'-'*60}")
            print(f"  [{self.version_name}]  SupCon Epoch {epoch+1}/{epochs}")
            print(f"{'-'*60}")

            train_loss = self.train_supcon_epoch(supcon_loader)

            history["supcon_loss"].append(round(train_loss, 6))

            print(f"  SupCon Loss: {train_loss:.4f}")

        # Save only backbone weights (exclude projector)
        backbone_path = os.path.join(save_dir, "backbone_supcon.pth")
        backbone_state = {
            k: v for k, v in self.model.state_dict().items()
            if not k.startswith("projector.")
        }
        torch.save(backbone_state, backbone_path)
        print(f"  Saved backbone (without projector) -> {backbone_path}")

        # Save history
        output = {
            "version": self.version_name,
            "total_epochs_run": epochs,
            "history": history,
        }
        with open(os.path.join(save_dir, "supcon_metrics.json"), "w") as f:
            json.dump(output, f, indent=2)

        return history
