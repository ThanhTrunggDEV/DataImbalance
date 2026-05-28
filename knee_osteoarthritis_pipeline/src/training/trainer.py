"""
Enhanced Trainer with:
  - Mixup augmentation support
  - Full metrics history (loss, acc, precision, recall, f1 per epoch)
  - Saves metrics.json at end of training
  - Named checkpoints per experiment version
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm

from .metrics import calculate_metrics
from .visualization import plot_training_history
from evaluation.visualization import plot_confusion_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Mixup helper
# ─────────────────────────────────────────────────────────────────────────────

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """
    Apply Mixup augmentation to a batch.

    Returns:
        mixed_x : linearly-interpolated images
        y_a, y_b: original and permuted labels
        lam     : mixing coefficient
    """
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixed loss: lam * L(y_a) + (1-lam) * L(y_b)."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

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
        mixup_alpha: float = 0.4,
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
        self.mixup_alpha = mixup_alpha

    # ── Single epoch helpers ──────────────────────────────────────────────────

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        loop = tqdm(self.train_loader, desc=f"[{self.version_name}] Train", leave=False)
        for images, labels in loop:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            if self.use_mixup:
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
            all_labels.extend(labels.cpu().numpy())

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

    # ── Main training loop ────────────────────────────────────────────────────

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
        }
        best_metrics_snapshot = {}

        for epoch in range(epochs):
            print(f"\n{'─'*60}")
            print(f"  [{self.version_name}]  Epoch {epoch+1}/{epochs}")
            print(f"{'─'*60}")

            train_loss, train_m = self.train_epoch()
            val_loss,   val_m   = self.validate()

            # ── Record history ────────────────────────────────────────────────
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

            # ── LR scheduler ──────────────────────────────────────────────────
            if self.lr_scheduler:
                self.lr_scheduler.step(val_loss)

            # ── Console summary ───────────────────────────────────────────────
            auc_str = f"{val_m['auc_macro']:.4f}" if 'auc_macro' in val_m and not np.isnan(val_m['auc_macro']) else "N/A "
            print(f"  Train  | Loss: {train_loss:.4f} | Acc: {train_m['accuracy']:.4f} | "
                  f"P: {train_m['precision_macro']:.4f} | R: {train_m['recall_macro']:.4f} | "
                  f"F1: {train_m['f1_macro']:.4f}")
            print(f"  Val    | Loss: {val_loss:.4f} | Acc: {val_m['accuracy']:.4f} | "
                  f"P: {val_m['precision_macro']:.4f} | R: {val_m['recall_macro']:.4f} | "
                  f"F1: {val_m['f1_macro']:.4f} | AUC: {auc_str}")

            # ── Checkpointing (best val F1) ───────────────────────────────────
            if val_m["f1_macro"] > best_f1:
                best_f1 = val_m["f1_macro"]
                best_cm = val_m["confusion_matrix"]
                torch.save(self.model.state_dict(), model_path)
                print(f"  ✓ Saved best model  →  Val F1 = {best_f1:.4f}")
                patience_counter = 0
                best_metrics_snapshot = {
                    "epoch": epoch + 1,
                    "val_loss": round(val_loss, 6),
                    "val_accuracy": round(val_m["accuracy"], 6),
                    "val_precision_macro": round(val_m["precision_macro"], 6),
                    "val_recall_macro": round(val_m["recall_macro"], 6),
                    "val_f1_macro": round(val_m["f1_macro"], 6),
                    # None → JSON null (NaN is invalid JSON)
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

        # ── Post-training: save artefacts ─────────────────────────────────────
        print(f"\n  Saving artefacts to  {save_dir} …")

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