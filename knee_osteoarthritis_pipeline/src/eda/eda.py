import os
from pathlib import Path
import argparse
import csv

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


def resolve_data_root_from_src_relative():
    # Default: repo_root/data where this script lives at src/eda/eda.py
    src_dir = Path(__file__).resolve().parent.parent
    repo_root = src_dir.parent
    data_root = repo_root / 'data'
    return str(data_root)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def save_csv_summary(summary_rows, out_path):
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'class', 'count'])
        for row in summary_rows:
            writer.writerow(row)


def save_pixel_stats_csv(stats, out_path):
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['channel', 'mean', 'std'])
        for channel_name, mean_value, std_value in zip(['R', 'G', 'B'], stats['mean'], stats['std']):
            writer.writerow([channel_name, mean_value, std_value])


def make_image_grid(image_paths, out_file, thumb_size=(224, 224), cols=5):
    imgs = []
    for p in image_paths:
        try:
            im = Image.open(p).convert('RGB')
            im.thumbnail(thumb_size)
            imgs.append(im)
        except Exception:
            continue
    if not imgs:
        return
    rows = (len(imgs) + cols - 1) // cols
    w, h = thumb_size
    grid = Image.new('RGB', (cols * w, rows * h), color=(255, 255, 255))
    for idx, im in enumerate(imgs):
        x = (idx % cols) * w
        y = (idx // cols) * h
        grid.paste(im, (x, y))
    grid.save(out_file)


def compute_pixel_stats(image_paths, max_images=None):
    sum_pix = np.zeros(3, dtype=np.float64)
    sum_pix_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    seen = 0
    for p in image_paths:
        try:
            im = Image.open(p).convert('RGB')
            arr = np.asarray(im, dtype=np.float32) / 255.0
            h, w, c = arr.shape
            pixels = h * w
            sum_pix += arr.reshape(-1, 3).sum(axis=0)
            sum_pix_sq += (arr.reshape(-1, 3) ** 2).sum(axis=0)
            total_pixels += pixels
            seen += 1
            if max_images and seen >= max_images:
                break
        except Exception:
            continue
    if total_pixels == 0:
        return None
    mean = sum_pix / total_pixels
    var = (sum_pix_sq / total_pixels) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 0.0))
    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'total_pixels': int(total_pixels),
        'images_used': int(seen)
    }


def build_count_matrix(summary_rows, splits):
    classes = sorted({row[1] for row in summary_rows}, key=lambda x: int(x) if str(x).isdigit() else str(x))
    matrix = np.zeros((len(splits), len(classes)), dtype=np.int64)
    lookup = {(split, cls): count for split, cls, count in summary_rows}
    for i, split in enumerate(splits):
        for j, cls in enumerate(classes):
            matrix[i, j] = int(lookup.get((split, cls), 0))
    return classes, matrix


def plot_class_distribution(summary_rows, splits, out_dir):
    classes, matrix = build_count_matrix(summary_rows, splits)

    x = np.arange(len(classes))
    width = 0.18
    plt.figure(figsize=(12, 6))
    for idx, split in enumerate(splits):
        plt.bar(x + (idx - (len(splits) - 1) / 2) * width, matrix[idx], width=width, label=split)
    plt.xticks(x, classes)
    plt.yscale('log')
    plt.xlabel('Class')
    plt.ylabel('Số lượng mẫu (log scale)')
    plt.title('Phân bố lớp theo split')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'plots' / 'class_distribution_by_split.png', dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='YlOrRd', cbar_kws={'label': 'Số lượng'})
    plt.yticks(np.arange(len(splits)) + 0.5, splits, rotation=0)
    plt.xticks(np.arange(len(classes)) + 0.5, classes, rotation=0)
    plt.title('Heatmap số lượng mẫu theo split/lớp')
    plt.xlabel('Class')
    plt.ylabel('Split')
    plt.tight_layout()
    plt.savefig(out_dir / 'plots' / 'class_count_heatmap.png', dpi=200)
    plt.close()

    train_idx = splits.index('train') if 'train' in splits else 0
    train_counts = matrix[train_idx]
    total = max(train_counts.sum(), 1)
    class_pct = train_counts / total * 100.0
    plt.figure(figsize=(10, 5))
    bars = plt.bar(classes, train_counts, color=sns.color_palette('crest', n_colors=len(classes)))
    plt.yscale('log')
    plt.xlabel('Class')
    plt.ylabel('Số lượng mẫu (log scale)')
    plt.title('Mất cân bằng lớp trên train')
    for bar, pct in zip(bars, class_pct):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / 'plots' / 'train_class_imbalance.png', dpi=200)
    plt.close()

    with open(out_dir / 'class_counts_matrix.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['split'] + classes)
        for split, row in zip(splits, matrix):
            writer.writerow([split] + row.tolist())


def add_split_totals_plot(summary_rows, splits, out_dir):
    split_totals = []
    for split in splits:
        total = sum(count for s, _, count in summary_rows if s == split)
        split_totals.append(total)

    plt.figure(figsize=(8, 4))
    bars = plt.bar(splits, split_totals, color=sns.color_palette('flare', n_colors=len(splits)))
    plt.ylabel('Tổng số ảnh')
    plt.title('Tổng số ảnh theo split')
    for bar, total in zip(bars, split_totals):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(total), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / 'plots' / 'split_totals.png', dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run EDA for Knee OA dataset')
    parser.add_argument('--data-root', default=None, help='Path to data root (overrides default ./data)')
    parser.add_argument('--out', default='outputs/eda', help='Output folder for EDA results')
    parser.add_argument('--samples-per-class', type=int, default=6, help='Number of sample images per class for gallery')
    parser.add_argument('--max-stats-images', type=int, default=2000, help='Max images to use for pixel stats (train)')
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent.parent
    data_root = args.data_root or resolve_data_root_from_src_relative()
    print(f"[EDA] Data root: {data_root}")

    out_dir = Path(args.out)
    ensure_dir(out_dir)
    ensure_dir(out_dir / 'plots')
    ensure_dir(out_dir / 'galleries')

    IMG_SIZE = 224
    splits = ['train', 'val', 'test', 'auto_test']
    summary_rows = []
    # Collect counts and galleries by scanning folders (no torch dependency)
    for split in splits:
        split_dir = Path(data_root) / split
        classes = []
        counts = []
        if split_dir.exists() and split_dir.is_dir():
            # class folders are subdirectories
            classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
            for cls_name in classes:
                cls_dir = split_dir / cls_name
                imgs = [str(p) for p in cls_dir.iterdir() if p.suffix.lower() in ('.png', '.jpg', '.jpeg')]
                counts.append(len(imgs))
                summary_rows.append([split, cls_name, len(imgs)])
                sample_paths = imgs[:args.samples_per_class]
                out_file = out_dir / 'galleries' / f'{split}_class_{cls_name}.png'
                make_image_grid(sample_paths, str(out_file), thumb_size=(IMG_SIZE // 2, IMG_SIZE // 2), cols=min(5, max(1, len(sample_paths))))
        else:
            # empty
            classes = []
            counts = []

    # Save summary CSV and plots
    save_csv_summary(summary_rows, out_dir / 'summary.csv')
    plot_class_distribution(summary_rows, splits, out_dir)
    add_split_totals_plot(summary_rows, splits, out_dir)

    # Pixel stats on train (scan files directly)
    train_dir = Path(data_root) / 'train'
    train_image_paths = []
    if train_dir.exists() and train_dir.is_dir():
        for cls_dir in train_dir.iterdir():
            if cls_dir.is_dir():
                for p in cls_dir.iterdir():
                    if p.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                        train_image_paths.append(str(p))
    train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()]) if train_dir.exists() and train_dir.is_dir() else []
    print(f"[EDA] Train classes: {train_classes}")
    pixel_stats = compute_pixel_stats(train_image_paths, max_images=args.max_stats_images)
    if pixel_stats:
        save_pixel_stats_csv(pixel_stats, out_dir / 'pixel_stats.csv')
        mean = np.array(pixel_stats['mean'])
        std = np.array(pixel_stats['std'])
        plt.figure(figsize=(7, 4))
        x = np.arange(3)
        plt.bar(x - 0.15, mean, width=0.3, label='Mean', color='#4C78A8')
        plt.bar(x + 0.15, std, width=0.3, label='Std', color='#F58518')
        plt.xticks(x, ['R', 'G', 'B'])
        plt.ylim(0, max(1.0, float(max(mean.max(), std.max()) * 1.25)))
        plt.ylabel('Giá trị (0-1)')
        plt.legend()
        plt.title('Mean và std theo kênh màu (train)')
        plt.tight_layout()
        plt.savefig(out_dir / 'plots' / 'channel_mean_std.png')
        plt.close()

    if summary_rows:
        classes, matrix = build_count_matrix(summary_rows, splits)
        plt.figure(figsize=(12, 5))
        for idx, split in enumerate(splits):
            plt.plot(classes, matrix[idx], marker='o', linewidth=2, label=split)
        plt.yscale('log')
        plt.xlabel('Class')
        plt.ylabel('Số lượng mẫu (log scale)')
        plt.title('Đường xu hướng số lượng lớp theo split')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'plots' / 'class_trend_line.png', dpi=200)
        plt.close()

    print(f"[EDA] Done. Outputs written to: {out_dir}")


if __name__ == '__main__':
    main()
