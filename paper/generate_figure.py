import os
import random
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from datasets import load_dataset
import numpy as np

# 1. Get EyePACS images (streaming)
eyepacs_images = {}
print("Loading EyePACS (streaming)...")
ds = load_dataset("bumbledeep/eyepacs", split="train", streaming=True)
for item in ds:
    label = item["label_code"]
    if label not in eyepacs_images:
        img = item["image"].convert("RGB")
        # Center crop somewhat
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) / 2
        top = (h - min_dim) / 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        eyepacs_images[label] = img
        print(f"Found DR {label}")
    if len(eyepacs_images) == 5:
        break

# 2. Get Knee OA images (local)
koa_images = {}
koa_dir = Path(r"d:\Coding Space\LearningML\DataImbalance\knee_osteoarthritis_pipeline\data\train")
print("Loading Knee OA...")
for i in range(5):
    folder = koa_dir / str(i)
    files = list(folder.glob("*.png"))
    if files:
        # pick a random one or the first one
        img_path = files[10] if len(files) > 10 else files[0]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) / 2
        top = (h - min_dim) / 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        koa_images[i] = img
        print(f"Found KOA {i}")

# 3. Stitch them together
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i in range(5):
    # Top Row: Knee OA
    ax = axes[0, i]
    if i in koa_images:
        ax.imshow(koa_images[i])
    ax.set_title(f"Knee OA: KL Grade {i}")
    ax.axis("off")
    
    # Bottom Row: EyePACS
    ax = axes[1, i]
    if i in eyepacs_images:
        ax.imshow(eyepacs_images[i])
    ax.set_title(f"EyePACS: DR Grade {i}")
    ax.axis("off")

plt.tight_layout()
out_path = r"d:\Coding Space\LearningML\DataImbalance\paper\dataset_samples.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved to {out_path}")
