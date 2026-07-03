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


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixed loss: lam * L(y_a) + (1-lam) * L(y_b)."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


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
        dataset_class_counts=None,
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

        # OWMix + memory queue (v13): per-class CPU queue of recent samples,
        # used as extra mixing partners so minority classes remain reachable
        # even when a mini-batch contains none of them.
        self.minority_queue = defaultdict(lambda: deque(maxlen=mixup_queue_size))
        if dataset_class_counts is not None:
            counts = torch.as_tensor(dataset_class_counts, dtype=torch.float32)
            self.inv_freq = (1.0 / (counts.sqrt() + 1e-8)).to(device)
        else:
            self.inv_freq = None

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
                self.lr_scheduler.step(val_loss)

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
