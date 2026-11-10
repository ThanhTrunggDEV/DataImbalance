EDA helper

Run the EDA script to produce summaries and plots in `outputs/eda/`.

What you will get:
- `outputs/eda/plots/class_distribution_by_split.png` - grouped bar chart of class counts by split
- `outputs/eda/plots/class_count_heatmap.png` - heatmap for quick imbalance inspection
- `outputs/eda/plots/train_class_imbalance.png` - train-split imbalance view
- `outputs/eda/plots/split_totals.png` - total images per split
- `outputs/eda/plots/channel_mean_std.png` - RGB mean/std summary
- `outputs/eda/plots/class_trend_line.png` - line chart across splits
- `outputs/eda/galleries/*.png` - sample image galleries per split/class

Usage:

```bash
python -u src/eda/eda.py
```

Options:
- `--data-root`: override data root (default: `./data`)
- `--out`: output directory (default: `outputs/eda`)
- `--samples-per-class`: number of example images per class
- `--max-stats-images`: max images to sample for pixel statistics

Notes:
- The script writes `summary.csv` and `pixel_stats.csv` for tabular export.
- Visual plots are the main output; JSON files are no longer the primary artifact.
