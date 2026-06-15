#!/usr/bin/env python3
"""
generate_report.py — Generate weekly HTML report from pipeline results.

Usage:
    python generate_report.py
    python generate_report.py --outputdir ../reports/weekly
    python generate_report.py --outputdir . --output index.html
"""

import argparse
import base64
import csv
import io
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from configs.config import (
    VERSIONS,
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, MIXUP_ALPHA, FOCAL_GAMMA,
    SUPCON_TEMPERATURE, SUPCON_PROJECT_DIM, SUPCON_EPOCHS, IMG_SIZE,
    EARLY_STOPPING_PATIENCE, NUM_CLASSES, SUPCON_IMG_SIZE,
)

CLASS_NAMES = [f"Grade {i}" for i in range(NUM_CLASSES)]
METRIC_LABELS = [
    ("f1_macro",        "F1 Macro"),
    ("accuracy",        "Accuracy"),
    ("precision_macro", "Precision Macro"),
    ("recall_macro",    "Recall Macro"),
    ("auc_macro",       "AUC Macro"),
]

VERSION_METHOD = {
    "v1_baseline":               "Cross-Entropy (baseline)",
    "v2_mixup":                  "Mixup + CE",
    "v3_balanced_softmax":       "Balanced Softmax",
    "v4_mixup_balanced_softmax": "Mixup + Balanced Softmax",
    "v5_focal_loss":             "Focal Loss (gamma=2)",
    "v6_adjacent_ce":            "Adjacent Mixup + CE",
    "v7_adjacent_balanced":      "Adjacent Mixup + Balanced Softmax",
    "v8_rule_ce":                "Rule-based Mixup + CE",
    "v9_rule_balanced":          "Rule-based Mixup + Balanced Softmax",
    "v10_supcon":                "SupCon + Finetune",
    "v11_owmixup_ce":           "OWMix + CE (τ=1.0)",
    "v11_owmixup_ce_t05":       "OWMix + CE (τ=0.5)",
    "v11_owmixup_ce_t15":       "OWMix + CE (τ=1.5)",
    "v11_owmixup_ce_t20":       "OWMix + CE (τ=2.0)",
    "v12_owmixup_balanced":     "OWMix + BalSoft (τ=1.0)",
    "v12_owmixup_balanced_t05": "OWMix + BalSoft (τ=0.5)",
    "v12_owmixup_balanced_t15": "OWMix + BalSoft (τ=1.5)",
    "v12_owmixup_balanced_t20": "OWMix + BalSoft (τ=2.0)",
}

# ── Helpers ─────────────────────────────────────────────────────────────────

def load_test_metrics(results_dir, version_name):
    path = os.path.join(results_dir, version_name, "test_metrics.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Normalize per_class keys: handle both "Grade 0" and "0" formats
        if "per_class" in data:
            normalized = {}
            for k, v in data["per_class"].items():
                grade = k.replace("Grade ", "")
                normalized[grade] = v
            data["per_class"] = normalized
        return data
    return None


def load_metrics_history(results_dir, version_name):
    path = os.path.join(results_dir, version_name, "metrics.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def load_classification_report(results_dir, version_name):
    path = os.path.join(results_dir, version_name, "test_classification_report.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return None


def load_summary_csv(results_dir):
    path = os.path.join(results_dir, "comparison", "all_versions_summary.csv")
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    return rows


def load_eda_summary(outputs_dir):
    path = os.path.join(outputs_dir, "eda", "summary.csv")
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)
    return rows


def img_to_base64(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    ext = Path(path).suffix.lower().lstrip(".")
    if ext == "png":
        mime = "image/png"
    elif ext in ("jpg", "jpeg"):
        mime = "image/jpeg"
    else:
        mime = "image/png"
    return f"data:{mime};base64,{b64}"


def copy_assets(output_dir, results_dir, outputs_dir):
    """Copy images to output directory and return relative paths."""
    assets = os.path.join(output_dir, "report_assets")
    plots_dir = os.path.join(assets, "plots")
    comp_dir = os.path.join(assets, "comparison")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)

    # Comparison images
    comp_map = {}
    for fname in ["metrics_comparison.png", "loss_f1_curves_comparison.png", "per_class_f1_heatmap.png"]:
        src = os.path.join(results_dir, "comparison", fname)
        if os.path.exists(src):
            dst = os.path.join(comp_dir, fname)
            shutil.copy2(src, dst)
            comp_map[fname] = os.path.relpath(dst, output_dir)

    # EDA plots
    eda_src = os.path.join(outputs_dir, "eda", "plots")
    eda_map = {}
    if os.path.exists(eda_src):
        for fname in os.listdir(eda_src):
            if fname.endswith(".png"):
                src = os.path.join(eda_src, fname)
                dst = os.path.join(plots_dir, fname)
                shutil.copy2(src, dst)
                eda_map[fname] = os.path.relpath(dst, output_dir)

    # Per-version images: embed as base64 in a dict for the HTML generator
    version_images = {}
    for vcfg in VERSIONS:
        vn = vcfg["name"]
        ver_dir = os.path.join(results_dir, vn)
        version_images[vn] = {}
        for iname in ["training_history.png", "test_confusion_matrix.png"]:
            src = os.path.join(ver_dir, iname)
            if os.path.exists(src):
                dst = os.path.join(assets, f"{vn}_{iname}")
                shutil.copy2(src, dst)
                version_images[vn][iname] = os.path.relpath(dst, output_dir)

    return comp_map, eda_map, version_images


# ── HTML Components ──────────────────────────────────────────────────────────

def build_styles():
    return """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', -apple-system, Roboto, Arial, sans-serif;
    background: #f5f6fa; color: #2d3436;
    line-height: 1.65; font-size: 15px;
    max-width: 1100px; margin: 0 auto; padding: 30px 20px 60px;
}
h1 { font-size: 26px; font-weight: 700; color: #1a1a2e; margin-bottom: 4px; }
h2 {
    font-size: 20px; font-weight: 600; color: #1a1a2e;
    border-bottom: 2px solid #dfe6e9; padding-bottom: 6px;
    margin-top: 40px; margin-bottom: 16px;
}
h3 { font-size: 17px; font-weight: 600; color: #2d3436; margin: 24px 0 10px; }
p { margin-bottom: 12px; }
table {
    width: 100%; border-collapse: collapse; margin: 14px 0;
    font-size: 14px;
}
th, td { padding: 8px 10px; text-align: center; border: 1px solid #dfe6e9; }
th { background: #1a1a2e; color: #fff; font-weight: 600; }
tr:nth-child(even) { background: #f8f9fa; }
tr:hover { background: #eef2f7; }
.meta { color: #636e72; font-size: 13px; margin-bottom: 24px; }
.section-desc { color: #636e72; font-size: 14px; margin-bottom: 14px; }
.version-badge {
    display: inline-block; padding: 3px 10px; border-radius: 4px;
    font-size: 12px; font-weight: 600; margin-bottom: 8px;
}
.best-row { background: #e8f5e9 !important; font-weight: 600; }
.top3-row { background: #e3f2fd !important; }
.card {
    background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    padding: 20px; margin-bottom: 20px; overflow-x: auto;
}
.version-card {
    background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    padding: 20px; margin-bottom: 28px;
}
.img-wrap { text-align: center; margin: 16px 0; }
.img-wrap img { max-width: 100%; height: auto; border-radius: 6px; border: 1px solid #eee; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.figure-caption { text-align: center; font-size: 13px; color: #636e72; margin-top: 6px; }
.exec-summary {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #fff; border-radius: 10px; padding: 24px 28px; margin-bottom: 28px;
}
.exec-summary h2 { color: #fff; border-color: rgba(255,255,255,0.2); }
.exec-summary .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: 16px; margin-top: 14px; }
.exec-summary .stat-item { text-align: center; background: rgba(255,255,255,0.08); border-radius: 8px; padding: 14px 8px; }
.exec-summary .stat-value { font-size: 28px; font-weight: 700; color: #00b894; }
.exec-summary .stat-label { font-size: 12px; color: rgba(255,255,255,0.7); margin-top: 2px; }
.insight-box {
    background: #fff3cd; border-left: 4px solid #ffc107;
    padding: 12px 16px; border-radius: 4px; margin: 12px 0;
    font-size: 14px;
}
.insight-box strong { color: #856404; }
.rec-box {
    background: #d4edda; border-left: 4px solid #28a745;
    padding: 12px 16px; border-radius: 4px; margin: 12px 0;
    font-size: 14px;
}
.rec-box strong { color: #155724; }
pre {
    background: #2d3436; color: #dfe6e9; padding: 14px 16px;
    border-radius: 6px; font-size: 13px; overflow-x: auto;
    font-family: 'Cascadia Code', 'Fira Code', 'Consolas', monospace;
}
@media (max-width: 768px) { .grid-2 { grid-template-columns: 1fr; } }
@media print {
    body { background: #fff; padding: 0; }
    .card, .version-card, .exec-summary { box-shadow: none; border: 1px solid #ddd; }
    .exec-summary { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    .best-row, .top3-row { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
}
.small-tbl { font-size: 13px; }
a { color: #0984e3; text-decoration: none; }
a:hover { text-decoration: underline; }
.toc { background: #fff; border-radius: 8px; padding: 16px 20px; margin-bottom: 28px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
.toc ul { list-style: none; columns: 2; column-gap: 24px; }
.toc li { margin: 4px 0; }
.toc a { color: #2d3436; }
.toc a:hover { color: #0984e3; }
.highlight-na { color: #b2bec3; font-style: italic; }
"""


def build_toc():
    sections = [
        ("exec",     "Tổng quan kết quả"),
        ("dataset",  "Thông tin dữ liệu"),
        ("config",   "Cấu hình thực nghiệm"),
        ("compare",  "So sánh các phương pháp"),
        ("figures",  "Biểu đồ so sánh"),
        ("perclass", "Phân tích từng lớp"),
        ("versions", "Chi tiết từng phiên bản"),
        ("insights", "Nhận xét & Đề xuất"),
        ("appendix", "Phụ lục — Per-class chi tiết"),
    ]
    html = '<div class="toc"><h3>Mục lục</h3><ul>'
    for anchor, title in sections:
        html += f'<li><a href="#{anchor}">{title}</a></li>'
    html += "</ul></div>"
    return html


def build_executive_summary(versions, summary_rows, results_dir):
    """Executive summary with key findings (using TEST metrics for ranking)."""

    # Load test metrics for all versions that have them
    test_rows = []
    for vcfg in VERSIONS:
        tm = load_test_metrics(results_dir, vcfg["name"])
        if tm:
            test_rows.append({
                "version": vcfg["name"],
                "display": vcfg["display"],
                "test_f1_macro": tm.get("test_f1_macro", 0) or 0,
                "test_accuracy": tm.get("test_accuracy", 0) or 0,
                "test_precision_macro": tm.get("test_precision_macro", 0) or 0,
                "test_recall_macro": tm.get("test_recall_macro", 0) or 0,
                "test_auc_macro": tm.get("test_auc_macro", 0) or 0,
            })

    if not test_rows:
        return "<div class='exec-summary'><h2>1. Tổng quan kết quả</h2><p>Chưa có dữ liệu test.</p></div>"

    # Sort by test F1 macro descending
    sorted_versions = sorted(test_rows, key=lambda r: r["test_f1_macro"], reverse=True)
    best = sorted_versions[0]
    top3 = sorted_versions[:3]

    best_f1 = best["test_f1_macro"]
    best_acc = best["test_accuracy"]
    best_auc = best["test_auc_macro"]
    best_name = best["display"]

    # Baseline F1 for comparison
    baseline_f1 = None
    for r in sorted_versions:
        if r["version"] == "v1_baseline":
            baseline_f1 = r["test_f1_macro"]
            break
    improvement = ""
    if baseline_f1:
        delta = (best_f1 - baseline_f1) / baseline_f1 * 100
        improvement = f"Cải thiện {delta:+.1f}% so với Baseline (F1={baseline_f1:.4f})."

    html = f"""<div class="exec-summary">
<h2 id="exec">1. Tổng quan kết quả</h2>
<p>Báo cáo tuần — so sánh {len(sorted_versions)} phương pháp xử lý mất cân bằng lớp
trên bài toán phân loại độ nặng thoái hóa khớp gối (KL-grade 0–4).</p>

<div class="stat-grid">
<div class="stat-item"><div class="stat-value">{best_f1:.4f}</div><div class="stat-label">F1 Macro (best, test)</div></div>
<div class="stat-item"><div class="stat-value">{best_acc:.4f}</div><div class="stat-label">Accuracy (best, test)</div></div>
<div class="stat-item"><div class="stat-value">{best_auc:.4f}</div><div class="stat-label">AUC Macro (best, test)</div></div>
<div class="stat-item"><div class="stat-value">{best_name}</div><div class="stat-label">Best method</div></div>
</div>
<p><strong>Kết quả chính (trên test set):</strong> Phương pháp <strong>{best_name}</strong> đạt F1 Macro cao nhất với <strong>{best_f1:.4f}</strong>.
{improvement}</p>"""

    # Top-3 comparison
    html += '<p style="margin-top:12px;"><strong>Top 3 trên test set (F1 Macro):</strong></p>'
    html += '<table><tr><th>Hạng</th><th>Phương pháp</th><th>F1 Macro</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>AUC</th></tr>'
    for i, r in enumerate(top3):
        rank_icon = ["🥇", "🥈", "🥉"][i]
        html += f"<tr class='{'best-row' if i==0 else 'top3-row'}'>"
        html += f"<td>{rank_icon}</td>"
        html += f"<td>{r['display']}</td>"
        html += f"<td>{r['test_f1_macro']:.4f}</td>"
        html += f"<td>{r['test_accuracy']:.4f}</td>"
        html += f"<td>{r['test_precision_macro']:.4f}</td>"
        html += f"<td>{r['test_recall_macro']:.4f}</td>"
        html += f"<td>{r['test_auc_macro']:.4f}</td>"
        html += "</tr>"
    html += "</table>"

    # Find worst-performing class across all versions
    class_f1s = {c: [] for c in range(NUM_CLASSES)}
    for vcfg in VERSIONS:
        tm = load_test_metrics(results_dir, vcfg["name"])
        if tm and "per_class" in tm:
            for k, v in tm["per_class"].items():
                class_f1s[int(k)].append(v["f1"])
    if class_f1s:
        avg_f1s = {c: sum(v)/len(v) for c, v in class_f1s.items() if v}
        worst = min(avg_f1s, key=avg_f1s.get)
        worst_val = avg_f1s[worst]
        html += f'<p style="margin-top:10px;"><strong>Lưu ý:</strong> Lớp {worst} (Grade {worst}) có F1 trung bình thấp nhất ({worst_val:.3f}) qua các phương pháp — đây là lớp khó phân loại nhất, cần tập trung cải thiện.</p>'

    html += "</div>"
    return html


def build_dataset_section(outputs_dir):
    """Class distribution table + EDA plots."""
    eda_rows = load_eda_summary(outputs_dir)
    if not eda_rows:
        return "<h2 id='dataset'>2. Thông tin dữ liệu</h2><p>Chưa có dữ liệu EDA.</p>"

    splits = ["train", "val", "test"]
    classes = sorted(set(r["class"] for r in eda_rows if r["split"] in splits))

    html = '<h2 id="dataset">2. Thông tin dữ liệu</h2>'
    html += '<p class="section-desc">Phân phối dữ liệu Knee Osteoarthritis — 5 lớp KL-grade (0: Bình thường, 4: Nặng nhất).</p>'

    # Build matrix: first compute totals per split, then percentages
    html += '<div class="card"><table>'
    html += "<tr><th>Grade</th>" + "".join(f"<th>{s.upper()}</th>" for s in splits) + "<th>Tổng</th></tr>"

    # Pre-compute totals per split
    total_per_split = {s: sum(int(r['count']) for r in eda_rows if r['split'] == s) for s in splits}
    # Pre-compute row totals per class
    class_row_totals = {}
    for c in classes:
        class_row_totals[c] = sum(int(r['count']) for r in eda_rows if r['class'] == c)

    for c in classes:
        html += f"<tr><td><strong>Grade {c}</strong></td>"
        row_total = 0
        for s in splits:
            count = 0
            for r in eda_rows:
                if r["split"] == s and r["class"] == c:
                    count = int(r["count"])
                    break
            row_total += count
            split_total = total_per_split.get(s, 0)
            pct = f" ({count/split_total*100:.1f}%)" if split_total > 0 else ""
            html += f"<td>{count}{pct}</td>"
        html += f"<td><strong>{row_total}</strong></td></tr>"

    html += "<tr style='font-weight:700; background:#dfe6e9;'>"
    html += "<td>Tổng</td>"
    grand = 0
    for s in splits:
        split_total = total_per_split.get(s, 0)
        html += f"<td>{split_total}</td>"
        grand += split_total
    html += f"<td>{grand}</td></tr>"
    html += "</table></div>"

    # Imbalance insight
    train_counts = {}
    for r in eda_rows:
        if r["split"] == "train":
            train_counts[int(r["class"])] = int(r["count"])
    if train_counts:
        max_c = max(train_counts, key=train_counts.get)
        min_c = min(train_counts, key=train_counts.get)
        ratio = train_counts[max_c] / train_counts[min_c]
        html += f'<p>Train set mất cân bằng: lớp {max_c} gấp {ratio:.1f}x lớp {min_c} ({train_counts[max_c]} vs {train_counts[min_c]} ảnh).</p>'

    return html


def build_config_section():
    """Hyperparameters table."""
    html = '<h2 id="config">3. Cấu hình thực nghiệm</h2>'
    html += '<div class="card">'
    html += '<p class="section-desc">Các tham số chung cho tất cả phiên bản (trừ SupCon có số epoch riêng).</p>'
    html += "<table class='small-tbl'>"
    params = [
        ("Kích thước ảnh đầu vào", f"{IMG_SIZE}x{IMG_SIZE}"),
        ("Batch size", str(BATCH_SIZE)),
        ("Số epoch tối đa", str(EPOCHS)),
        ("Learning rate", f"{LR}"),
        ("Weight decay", f"{WEIGHT_DECAY}"),
        ("Optimizer", "AdamW"),
        ("Scheduler", "ReduceLROnPlateau (patience=3, factor=0.5)"),
        ("Early stopping", f"patience={EARLY_STOPPING_PATIENCE} trên val F1"),
        ("Mixup alpha", str(MIXUP_ALPHA)),
        ("Focal gamma", str(FOCAL_GAMMA)),
        ("SupCon temperature", str(SUPCON_TEMPERATURE)),
        ("SupCon project dim", str(SUPCON_PROJECT_DIM)),
        ("SupCon pretrain epochs", str(SUPCON_EPOCHS)),
        ("Backbone", "ResNet50 (ImageNet pretrained)"),
        ("Dropout", "0.5"),
    ]
    html += "<tr><th style='width:40%;'>Tham số</th><th>Giá trị</th></tr>"
    for label, value in params:
        html += f"<tr><td>{label}</td><td>{value}</td></tr>"
    html += "</table></div>"

    # Version-specific config table
    html += '<div class="card">'
    html += "<p class='section-desc'>Đặc điểm riêng từng phiên bản.</p>"
    html += "<table class='small-tbl'>"
    html += "<tr><th>Phiên bản</th><th>Kỹ thuật</th><th>Loss function</th><th>Sampler</th></tr>"
    for vcfg in VERSIONS:
        vn = vcfg["name"]
        loss_map = {
            "cross_entropy": "CrossEntropy",
            "balanced_softmax": "BalancedSoftmax",
            "focal": "Focal Loss",
            "supcon": "SupCon → BalancedSoftmax",
        }
        loss = loss_map.get(vcfg["loss_type"], vcfg["loss_type"])
        sampler = "WeightedRandomSampler" if vcfg.get("use_sampler") else "—"
        mixup_str = vcfg.get("mixup_mode", "standard") if vcfg.get("use_mixup") else "—"
        html += f"<tr><td><strong>{vn}</strong></td><td>{VERSION_METHOD.get(vn, '')}</td><td>{loss}</td><td>{sampler}</td></tr>"
    html += "</table></div>"
    return html


def build_comparison_table(summary_rows):
    """Full metrics comparison table."""
    if not summary_rows:
        return "<h2 id='compare'>4. So sánh các phương pháp</h2><p>Chưa có dữ liệu.</p>"

    html = '<h2 id="compare">4. So sánh các phương pháp</h2>'
    html += '<p class="section-desc">Bảng xếp hạng các phương pháp theo F1 Macro trên validation set.</p>'
    html += '<div class="card">'

    sorted_rows = sorted(summary_rows, key=lambda r: float(r.get("val_f1_macro", 0)), reverse=True)

    html += "<table>"
    html += "<tr><th>#</th><th>Phương pháp</th><th>F1 Macro</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>AUC</th><th>Epoch</th></tr>"
    for i, r in enumerate(sorted_rows):
        cls = " best-row" if i == 0 else (" top3-row" if i < 3 else "")
        html += f"<tr class='{cls}'>"
        html += f"<td>{i+1}</td>"
        html += f"<td style='text-align:left;'>{r['display']}</td>"
        for mk in ["val_f1_macro", "val_accuracy", "val_precision_macro", "val_recall_macro", "val_auc_macro"]:
            val = float(r.get(mk, 0))
            html += f"<td>{val:.4f}</td>"
        html += f"<td>{r.get('epoch', '—')}</td>"
        html += "</tr>"
    html += "</table>"
    html += "</div>"
    return html


def build_figures_section(comp_map):
    """Comparison charts."""
    html = '<h2 id="figures">5. Biểu đồ so sánh</h2>'
    html += '<p class="section-desc">Trực quan hóa kết quả so sánh giữa các phương pháp.</p>'

    fig_mapping = [
        ("metrics_comparison.png",     "So sánh các metrics giữa tất cả phiên bản (validation set)"),
        ("loss_f1_curves_comparison.png", "Đường cong validation Loss và F1 Macro qua các epoch"),
        ("per_class_f1_heatmap.png",   "Heatmap F1-score cho từng lớp trên test set"),
    ]

    for fname, caption in fig_mapping:
        img_path = comp_map.get(fname)
        if img_path:
            html += f'<div class="card img-wrap"><img src="{img_path}" alt="{caption}" />'
            html += f'<div class="figure-caption">{caption}</div></div>'

    return html


def build_per_class_analysis(results_dir):
    """Per-class F1 analysis table."""
    html = '<h2 id="perclass">6. Phân tích từng lớp</h2>'
    html += '<p class="section-desc">F1-score cho từng lớp (Grade 0–4) trên test set — so sánh giữa các phương pháp.</p>'
    html += '<div class="card">'

    # Build matrix
    html += "<table>"
    html += "<tr><th>Phương pháp</th>"
    for c in range(NUM_CLASSES):
        html += f"<th>Grade {c}</th>"
    html += "<th>F1 Macro</th></tr>"

    # Sort by F1 macro descending
    class_f1s_all = {}
    for vcfg in VERSIONS:
        vn = vcfg["name"]
        tm = load_test_metrics(results_dir, vn)
        if not tm or "per_class" not in tm:
            continue
        row_f1s = {int(k): v["f1"] for k, v in tm["per_class"].items()}
        class_f1s_all[vn] = row_f1s

    sorted_versions = sorted(class_f1s_all.keys(),
        key=lambda vn: sum(class_f1s_all[vn].values()) / len(class_f1s_all[vn]),
        reverse=True)

    for vn in sorted_versions:
        rf = class_f1s_all[vn]
        macro_f1 = sum(rf.values()) / len(rf)
        row_best = vn == sorted_versions[0]
        html += f"<tr class='{'best-row' if row_best else ''}'>"
        html += f"<td style='text-align:left;'>{VERSION_METHOD.get(vn, vn)}</td>"
        for c in range(NUM_CLASSES):
            f1 = rf.get(c, 0)
            cls_highlight = ""
            if f1 >= 0.8: cls_highlight = " style='color:#00b894; font-weight:600;'"
            elif f1 < 0.35: cls_highlight = " style='color:#d63031; font-weight:600;'"
            elif f1 < 0.5: cls_highlight = " style='color:#e17055;'"
            html += f"<td{cls_highlight}>{f1:.4f}</td>"
        html += f"<td><strong>{macro_f1:.4f}</strong></td>"
        html += "</tr>"

    html += "</table></div>"

    # Worst class analysis
    html += '<h3>6.1. Lớp khó phân loại nhất</h3>'
    avg_f1s = {}
    for c in range(NUM_CLASSES):
        vals = [class_f1s_all[vn].get(c, 0) for vn in sorted_versions]
        avg_f1s[c] = sum(vals) / len(vals) if vals else 0

    worst = min(avg_f1s, key=avg_f1s.get)
    second_worst = sorted(avg_f1s.items(), key=lambda x: x[1])[1][0]
    best_class = max(avg_f1s, key=avg_f1s.get)

    html += '<div class="insight-box">'
    html += f"<strong>⚠ Lớp Grade {worst}</strong> — F1 trung bình thấp nhất ({avg_f1s[worst]:.3f}). "
    html += f"Đây là lớp {['Bình thường', 'Nghi ngờ', 'Nhẹ', 'Trung bình', 'Nặng'][worst]} "
    html += f"với rất ít mẫu huấn luyện. "
    html += f"Lớp Grade {best_class} có F1 cao nhất ({avg_f1s[best_class]:.3f}) do có nhiều mẫu nhất."
    html += "</div>"

    return html


def build_version_detail(results_dir, comp_map):
    """Detailed section for each version."""
    html = '<h2 id="versions">7. Chi tiết từng phiên bản</h2>'
    html += '<p class="section-desc">Training history, confusion matrix và classification report cho mỗi phương pháp.</p>'

    for vcfg in VERSIONS:
        vn = vcfg["name"]
        display = vcfg["display"]
        metrics = load_metrics_history(results_dir, vn)
        test_metrics = load_test_metrics(results_dir, vn)
        cls_report = load_classification_report(results_dir, vn)

        if not metrics and not test_metrics:
            continue

        html += f'<div class="version-card" id="ver-{vn}">'
        html += f'<h3 style="margin-top:0;">{display} ({vn})</h3>'

        # Summary stats
        if test_metrics:
            html += "<div style='display:flex; flex-wrap:wrap; gap:8px; margin-bottom:12px;'>"
            for key, label in METRIC_LABELS:
                val = test_metrics.get(f"test_{key}", None)
                if val is not None:
                    html += f"<span style='background:#dfe6e9; padding:2px 10px; border-radius:4px; font-size:13px;'><strong>{label}:</strong> {val:.4f}</span>"
            html += "</div>"

        # Training history + confusion matrix side by side
        html += '<div class="grid-2">'

        # Training history
        history_img = None
        history_path = os.path.join(results_dir, vn, "training_history.png")
        if os.path.exists(history_path):
            b64 = img_to_base64(history_path)
            if b64:
                html += f'<div class="img-wrap"><img src="{b64}" alt="{vn} training history" /><div class="figure-caption">Training history</div></div>'
            else:
                html += "<div></div>"
        else:
            html += "<div></div>"

        # Confusion matrix
        cm_img = None
        cm_path = os.path.join(results_dir, vn, "test_confusion_matrix.png")
        if os.path.exists(cm_path):
            b64 = img_to_base64(cm_path)
            if b64:
                html += f'<div class="img-wrap"><img src="{b64}" alt="{vn} confusion matrix" /><div class="figure-caption">Test confusion matrix</div></div>'
            else:
                html += "<div></div>"
        else:
            html += "<div></div>"

        html += "</div>"

        # Classification report
        if cls_report:
            html += "<pre>" + cls_report.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + "</pre>"
        else:
            html += "<p>Chưa có classification report.</p>"

        html += "</div>"

    return html


def build_insights(results_dir, summary_rows, outputs_dir):
    """Generate insights and recommendations (using TEST metrics for ranking)."""

    # Load test metrics
    test_rows = {}
    for vcfg in VERSIONS:
        tm = load_test_metrics(results_dir, vcfg["name"])
        if tm:
            test_rows[vcfg["name"]] = tm

    if not test_rows:
        return "<h2 id='insights'>8. Nhận xét & Đề xuất</h2><p>Chưa đủ dữ liệu để phân tích.</p>"

    sorted_names = sorted(test_rows.keys(), key=lambda vn: test_rows[vn].get("test_f1_macro", 0), reverse=True)
    best_vn = sorted_names[0]
    best = test_rows[best_vn]
    best_f1 = best.get("test_f1_macro", 0)
    best_display = next((v["display"] for v in VERSIONS if v["name"] == best_vn), best_vn)

    # Baseline F1
    baseline_f1 = 0
    for vn in sorted_names:
        if vn == "v1_baseline":
            baseline_f1 = test_rows[vn].get("test_f1_macro", 0)
            break

    html = '<h2 id="insights">8. Nhận xét & Đề xuất</h2>'

    # Key findings
    html += "<h3>8.1. Nhận xét chính</h3>"
    html += '<div class="card">'

    # Compare best vs baseline
    if baseline_f1 > 0:
        html += f'<div class="insight-box"><strong>📊 So với Baseline:</strong> '
        delta = best_f1 - baseline_f1
        html += f'Phương pháp tốt nhất ({best_display}) cải thiện F1 Macro từ {baseline_f1:.4f} lên {best_f1:.4f} (tăng {delta:+.4f}).'
        html += "</div>"

    # Per-class analysis
    class_f1s = {c: [] for c in range(NUM_CLASSES)}
    for vcfg in VERSIONS:
        tm = load_test_metrics(results_dir, vcfg["name"])
        if tm and "per_class" in tm:
            for k, v in tm["per_class"].items():
                class_f1s[int(k)].append(v["f1"])
    if class_f1s:
        avg_f1s = {c: sum(v)/len(v) for c, v in class_f1s.items() if v}
        worst = min(avg_f1s, key=avg_f1s.get)
        best_c = max(avg_f1s, key=avg_f1s.get)
        html += f'<div class="insight-box"><strong>🔍 Phân tích lớp:</strong> '
        html += f'Grade {worst} là lớp khó nhất (F1 trung bình {avg_f1s[worst]:.3f}) — '
        html += f'Grade {best_c} là lớp dễ nhất (F1 trung bình {avg_f1s[best_c]:.3f}). '
        html += f'Khoảng cách giữa lớp mạnh nhất và yếu nhất là {avg_f1s[best_c]-avg_f1s[worst]:.3f}.'
        html += "</div>"

    html += "</div>"

    # Recommendations
    html += "<h3>8.2. Đề xuất cho tuần sau</h3>"
    html += '<div class="card">'

    reco = [
        "Thử nghiệm <strong>weighted sampling</strong> kết hợp với Focal Loss để cải thiện Grade 1 (lớp nghi ngờ — dễ nhầm với Grade 0 và 2).",
        "Áp dụng <strong>class-aware augmentation</strong> (ví dụ: oversampling cho Grade 1 và 4 kết hợp với augmentation mạnh hơn).",
        "Thử nghiệm <strong>Ordinal regression</strong> (CORN, CORAL) — tận dụng tính chất có thứ tự của KL-grade.",
        "Cân nhắc dùng <strong>EfficientNet-B5</strong> (đã có trong codebase) làm backbone stronger.",
        "Thử nghiệm <strong>ensemble</strong> top-3 phương pháp (ví dụ: voting giữa baseline, focal loss và supcon).",
        "Phân tích <strong>error case</strong> trên Grade 1 misclassified samples để hiểu rõ nguyên nhân.",
    ]
    for r in reco:
        html += f'<div class="rec-box">💡 {r}</div>'

    html += "</div>"

    # Best configuration summary
    html += "<h3>8.3. Cấu hình đề xuất</h3>"
    html += '<div class="card">'
    html += f"<p>Dựa trên kết quả hiện tại, cấu hình đề xuất cho production pipeline:</p>"
    html += "<table>"
    html += "<tr><th>Thành phần</th><th>Giá trị</th></tr>"
    html += f"<tr><td>Phương pháp chính</td><td><strong>{best_display}</strong> (F1={best_f1:.4f})</td></tr>"
    html += "<tr><td>Loss function</td><td>Xem config phần trên</td></tr>"
    html += "<tr><td>Mixup</td><td>Khuyến khích sử dụng Mixup hoặc biến thể</td></tr>"
    html += "<tr><td>Lưu ý</td><td>Cần xử lý riêng Grade 1 (lớp có F1 thấp nhất)</td></tr>"
    html += "</table>"
    html += "</div>"

    return html


def build_appendix(results_dir):
    """Per-class metrics for all versions."""
    html = '<h2 id="appendix">9. Phụ lục — Per-class metrics chi tiết</h2>'
    html += '<p class="section-desc">Precision, Recall, F1 cho từng lớp trên test set — tất cả phiên bản.</p>'

    for vcfg in VERSIONS:
        vn = vcfg["name"]
        tm = load_test_metrics(results_dir, vn)
        if not tm or "per_class" not in tm:
            continue

        html += f'<div class="card">'
        html += f"<h4 style='margin-top:0;'>{vcfg['display']} ({vn})</h4>"
        html += "<table>"
        html += "<tr><th>Grade</th><th>Precision</th><th>Recall</th><th>F1-score</th><th>Support</th></tr>"
        for c in range(NUM_CLASSES):
            pc = tm["per_class"].get(str(c), {})
            html += f"<tr><td>{c}</td><td>{pc.get('precision', 0):.4f}</td><td>{pc.get('recall', 0):.4f}</td><td>{pc.get('f1', 0):.4f}</td><td>{pc.get('support', '—')}</td></tr>"
        html += "</table>"
        html += "</div>"

    return html


# ── Generate ─────────────────────────────────────────────────────────────────

def generate_report(results_dir, outputs_dir, output_path, dataset=""):
    """Main generator."""
    results_dir = os.path.abspath(results_dir)
    outputs_dir = os.path.abspath(outputs_dir)
    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)

    print(f"[Report] Loading from: {results_dir}")
    print(f"[Report] Output to:   {output_path}")

    # Detect dataset from results_dir parent if not explicitly set
    if not dataset:
        parent_dir = os.path.basename(os.path.dirname(results_dir.rstrip(os.sep)))
        if parent_dir in ("koa", "eyepacs"):
            dataset = parent_dir

    # Load data
    summary_rows = load_summary_csv(results_dir)
    print(f"         Summary: {len(summary_rows)} versions")

    # Copy assets
    comp_map, eda_map, version_images = copy_assets(output_dir, results_dir, outputs_dir)
    print(f"         Assets:  {len(comp_map)} comparison + {len(eda_map)} EDA images")

    dataset_title = dataset.upper() if dataset else "Knee OA"
    dataset_label = f" — {dataset_title.upper()}" if dataset else ""

    # Build HTML
    html_parts = [
        "<!DOCTYPE html><html lang='vi'><head>",
        "<meta charset='UTF-8'><meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"<title>Weekly Report{dataset_label} — Imbalance Pipeline</title>",
        f"<style>{build_styles()}</style>",
        "</head><body>",

        # Header
        f"<h1>📋 Báo cáo thực nghiệm{dataset_label}</h1>",
        f"<p class='meta'>Ngày: {datetime.now().strftime('%d/%m/%Y %H:%M')} &nbsp;|&nbsp; "
        f"Số phiên bản: {len(summary_rows)} &nbsp;|&nbsp; "
        f"Pipeline: Multi-version imbalance strategies</p>",

        build_toc(),
        build_executive_summary(VERSIONS, summary_rows, results_dir),
        build_dataset_section(outputs_dir),
        build_config_section(),
        build_comparison_table(summary_rows),
        build_figures_section(comp_map),
        build_per_class_analysis(results_dir),
        build_version_detail(results_dir, comp_map),
        build_insights(results_dir, summary_rows, outputs_dir),
        build_appendix(results_dir),

        "<hr style='margin:40px 0; border:none; border-top:1px solid #dfe6e9;' />",
        f"<p style='text-align:center; color:#636e72; font-size:13px;'>"
        f"Generated by generate_report.py — {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>",

        "</body></html>",
    ]

    html_content = "\n".join(html_parts)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\n=> Report generated: {output_path}")
    print(f"  Open in browser to view.")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate weekly HTML report from pipeline results.")
    parser.add_argument("--results-dir", default=os.path.join(_SCRIPT_DIR, "../results"),
                        help="Path to results directory (default: ../results)")
    parser.add_argument("--outputs-dir", default=os.path.join(_SCRIPT_DIR, "../outputs"),
                        help="Path to outputs directory (default: ../outputs)")
    parser.add_argument("--outputdir", default=os.path.join(_SCRIPT_DIR, ".."),
                        help="Directory to save the report (default: project root)")
    parser.add_argument("--output", default="weekly_report.html",
                        help="Output filename (default: weekly_report.html)")
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset subdirectory under results_dir (e.g. koa, eyepacs)")
    args = parser.parse_args()

    results_dir = args.results_dir
    if args.dataset:
        results_dir = os.path.join(results_dir, args.dataset)
    if not os.path.isdir(results_dir):
        print(f"[ERROR] Results directory not found: {results_dir}")
        sys.exit(1)

    output_path = os.path.join(args.outputdir, args.output)
    generate_report(results_dir, args.outputs_dir, output_path, dataset=args.dataset)


if __name__ == "__main__":
    main()
