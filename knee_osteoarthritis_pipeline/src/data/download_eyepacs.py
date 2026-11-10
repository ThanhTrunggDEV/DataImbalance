"""
Download EyePACS Diabetic Retinopathy dataset from HuggingFace.
Preprocess to match Knee OA pipeline: 224x224, folder structure data_dr/{train,val,test}/{0-4}/.
"""
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm


PIPELINE_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = PIPELINE_ROOT / "data_dr"
TARGET_SIZE = (224, 224)
RANDOM_STATE = 42

SPLIT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}
VALID_SPLIT = SPLIT_RATIOS["val"] / (SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"])


def main():
    print("=" * 60)
    print("Downloading EyePACS from HuggingFace (bumbledeep/eyepacs) ...")
    print("=" * 60)

    ds = load_dataset("bumbledeep/eyepacs", split="train")
    N = len(ds)
    print(f"Total images: {N}")

    images = ds["image"]
    labels = ds["label_code"]

    print("\nClass distribution (before split):")
    unique, counts = np.unique(labels, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Grade {cls}: {cnt:>6} ({100*cnt/N:.1f}%)")

    labels = np.array(labels)

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        range(N), labels,
        test_size=(1 - SPLIT_RATIOS["train"]),
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(SPLIT_RATIOS["test"] / (SPLIT_RATIOS["val"] + SPLIT_RATIOS["test"])),
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    split_indices = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    print(f"\nSplit sizes:")
    for split_name, indices in split_indices.items():
        split_labels = labels[list(indices)]
        unique_s, counts_s = np.unique(split_labels, return_counts=True)
        total_s = len(indices)
        print(f"  {split_name}: {total_s:>5} images")
        for cls, cnt in zip(unique_s, counts_s):
            print(f"    Grade {cls}: {cnt:>5} ({100*cnt/total_s:.1f}%)")

    for split_name, indices in split_indices.items():
        split_dir = OUTPUT_DIR / split_name
        print(f"\nSaving {split_name} ({len(indices)} images) to {split_dir} ...")

        for cls in range(5):
            cls_dir = split_dir / str(cls)
            cls_dir.mkdir(parents=True, exist_ok=True)

        for idx in tqdm(indices, desc=f"  {split_name}"):
            img = images[idx]
            label = int(labels[idx])

            if img.mode != "RGB":
                img = img.convert("RGB")

            img = img.resize(TARGET_SIZE, Image.BILINEAR)

            cls_dir = split_dir / str(label)
            fname = f"eyepacs_{idx:06d}.png"
            img.save(cls_dir / fname)

    print("\nSummary:")
    total_saved = 0
    for split_name in ["train", "val", "test"]:
        n_files = sum(1 for _ in (OUTPUT_DIR / split_name).rglob("*.png"))
        total_saved += n_files
        print(f"  {split_name}: {n_files} images")
    print(f"  Total: {total_saved} images")
    print(f"  Location: {OUTPUT_DIR}")
    print("\nDone!")


if __name__ == "__main__":
    main()
