import os
import numpy as np
from PIL import Image
from typing import List, Optional
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class KneeDataset(Dataset):
    """
    ImageFolder-style dataset for Knee OA severity classification.
    Folder structure expected:
        root_dir/
            train/  0/  1/  2/  3/  4/   (class sub-folders by severity)
            val/    ...
            test/   ...
    """

    def __init__(self, root_dir: str, split: str = "train", transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.image_paths: List[str] = []
        self.labels: List[int] = []

        if not os.path.exists(self.root_dir):
            print(f"[WARNING] Directory not found: {self.root_dir}")
            self.classes = []
            return

        # Only count actual subdirectories as classes (ignore .DS_Store, Thumbs.db, etc.)
        self.classes = sorted(
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        )
        for label, cls in enumerate(self.classes):
            cls_dir = os.path.join(self.root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(label)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    # ── Class-weight helpers ──────────────────────────────────────────────────

    def get_sample_weights(self):
        """Per-sample weights for WeightedRandomSampler (inverse class freq)."""
        if not self.labels:
            return np.array([], dtype=np.float64), np.zeros(len(self.classes), dtype=np.float64)
        counts = np.bincount(self.labels, minlength=len(self.classes))
        class_weights = np.where(counts > 0, 1.0 / counts, 0.0)
        sample_weights = class_weights[self.labels]
        return sample_weights, class_weights

    def get_class_counts(self) -> np.ndarray:
        """Raw count per class — needed by BalancedSoftmaxLoss."""
        if not self.labels:
            return np.zeros(max(len(self.classes), 1), dtype=np.float32)
        return np.bincount(self.labels, minlength=len(self.classes)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ─────────────────────────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    use_sampler: bool = True,
    num_workers: int = 4,
):
    """
    Build train / val / test DataLoaders.

    Args:
        data_dir:     Root directory that contains train/ val/ test/ sub-folders.
        batch_size:   Samples per batch.
        img_size:     Resize target (224 for ResNet50, 456 for EfficientNet-B5).
        use_sampler:  If True, use WeightedRandomSampler to balance batches.
                      Set False for the true baseline experiment.
        num_workers:  DataLoader parallelism.
                      On Windows, multiprocessing requires the entry point to be
                      guarded by `if __name__ == '__main__':` (already the case
                      in run_all.py / run_experiment.py). Set to 0 if you hit
                      BrokenPipeError or DataLoader hang issues.

    Returns:
        train_loader, val_loader, test_loader,
        class_weights (np.ndarray, inverse-freq),
        class_counts  (np.ndarray, raw counts)
    """

    # Auto-detect common dataset structures
    candidate = os.path.join(data_dir, "Data")
    actual_root = candidate if os.path.isdir(candidate) else data_dir

    # ImageNet normalization works well for transfer learning
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),  # slightly larger for random crop
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),                 # valid: L/R knee are mirror images
        # NOTE: RandomVerticalFlip is intentionally OMITTED — it would invert
        # femur/tibia orientation, producing anatomically impossible images.
        transforms.RandomRotation(10),                     # small rotations only
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = KneeDataset(actual_root, split="train", transform=train_transform)
    val_ds   = KneeDataset(actual_root, split="val",   transform=val_test_transform)
    test_ds  = KneeDataset(actual_root, split="test",  transform=val_test_transform)

    sample_weights, class_weights = train_ds.get_sample_weights()
    class_counts = train_ds.get_class_counts()

    # ── Train loader ──────────────────────────────────────────────────────────
    # persistent_workers speeds up epoch transitions when num_workers > 0
    _pw = num_workers > 0

    if use_sampler and len(sample_weights) > 0:
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, persistent_workers=_pw,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, persistent_workers=_pw,
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_weights, class_counts