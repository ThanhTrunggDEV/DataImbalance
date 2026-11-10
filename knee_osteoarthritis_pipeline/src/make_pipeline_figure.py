"""
Draw the OWMM pipeline schematic (Fig. fig:method in paper/manuscript.tex).

Pure matplotlib — no data or checkpoints required. Renders the four coupled
mechanisms described in the Methodology as a left-to-right flow:
  (1) ordinal-weighted partner selection  (Gaussian kernel x effective-number weight)
  (2) manifold interpolation at ResNet layer3
  (3) ordinally-smoothed soft target
  (4) tempered prior gamma on the log-frequency logit shift

Usage (from knee_osteoarthritis_pipeline/src):
    python make_pipeline_figure.py                 # -> paper/figs/pipeline_owmm.png
    python make_pipeline_figure.py --out foo.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PIPELINE_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_OUT = os.path.join(PIPELINE_ROOT, "..", "paper", "figs", "pipeline_owmm.png")

# palette
C_BACKBONE = "#dbe7f3"
C_MIX      = "#f6d9b8"
C_SELECT   = "#d8ecd8"
C_LOSS     = "#f3d3d8"
C_EDGE     = "#3a3a3a"
C_ACCENT   = "#b5480f"


def _box(ax, x, y, w, h, text, face, fontsize=8.5, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.1, edgecolor=C_EDGE, facecolor=face, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, zorder=3,
            fontweight="bold" if bold else "normal")


def _arrow(ax, x0, y0, x1, y1, color=C_EDGE, style="-|>", lw=1.3, ls="-"):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=13,
        linewidth=lw, color=color, linestyle=ls, zorder=1,
        shrinkA=1, shrinkB=1))


def draw(out_path: str):
    fig, ax = plt.subplots(figsize=(11.0, 4.3))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.3)
    ax.axis("off")

    ymid = 2.55  # main flow row

    # 1. mini-batch (imbalanced)
    _box(ax, 0.15, ymid, 1.35, 0.9,
         "Mini-batch\n$(\\mathbf{X},\\mathbf{y})$\nimbalanced", C_BACKBONE, bold=True)

    # 2. backbone -> layer3 features
    _box(ax, 1.9, ymid, 1.55, 0.9,
         "Backbone\n$f_\\theta$ to $\\mathtt{layer3}$\n$\\rightarrow \\mathbf{h}_i,\\mathbf{h}_j$", C_BACKBONE)
    _arrow(ax, 1.5, ymid + 0.45, 1.9, ymid + 0.45)

    # 3. partner selection (below), feeds the mix box
    _box(ax, 1.75, 0.35, 2.05, 1.15,
         "Ordinal-weighted\npartner selection\n"
         "$p_{ij}\\!\\propto\\!e^{-(y_i-y_j)^2/\\tau^2} w_{y_i} w_{y_j}$\n"
         "eff.-number weights $w_c$", C_SELECT, fontsize=8.0)

    # 4. manifold mix
    _box(ax, 3.85, ymid, 1.7, 0.9,
         "Manifold mix\n$\\mathbf{h}\\!=\\!\\lambda\\mathbf{h}_i\\!+\\!(1\\!-\\!\\lambda)\\mathbf{h}_j$\n"
         "$\\lambda\\!\\sim\\!\\mathrm{Beta}(\\alpha,\\alpha)$", C_MIX, bold=True)
    _arrow(ax, 3.45, ymid + 0.45, 3.85, ymid + 0.45)
    _arrow(ax, 2.9, 1.5, 4.4, ymid, color=C_ACCENT, ls="--", lw=1.2)  # partner -> mix

    # 5. remaining layers + head -> logits
    _box(ax, 5.95, ymid, 1.6, 0.9,
         "$\\mathtt{layer4}$ + head\n$\\rightarrow$ logits $\\mathbf{z}$", C_BACKBONE)
    _arrow(ax, 5.55, ymid + 0.45, 5.95, ymid + 0.45)

    # 6. tempered prior shift
    _box(ax, 7.95, ymid, 1.55, 0.9,
         "Tempered prior\n$\\mathbf{z}+\\gamma\\log\\mathbf{n}$\n$\\gamma\\in[0,1]$", C_MIX, bold=True)
    _arrow(ax, 7.55, ymid + 0.45, 7.95, ymid + 0.45)

    # 7. soft ordinal target (below-right), feeds loss
    _box(ax, 7.9, 0.55, 2.0, 0.95,
         "Soft ordinal target\n$\\tilde{\\mathbf{y}}$: two-hot $(\\lambda)$\n"
         "Gaussian-smoothed $\\sigma$", C_SELECT, fontsize=8.0)

    # 8. loss
    _box(ax, 9.85, ymid, 1.0, 0.9,
         "$\\mathcal{L}=$\nCE$(\\cdot,\\tilde{\\mathbf{y}})$", C_LOSS, bold=True)
    _arrow(ax, 9.5, ymid + 0.45, 9.85, ymid + 0.45)
    _arrow(ax, 8.9, 1.5, 10.15, ymid, color=C_ACCENT, ls="--", lw=1.2)  # target -> loss

    # backprop arrow
    _arrow(ax, 10.35, ymid, 10.35, 4.05, color=C_ACCENT, lw=1.1)
    _arrow(ax, 10.35, 4.05, 2.7, 4.05, color=C_ACCENT, lw=1.1)
    _arrow(ax, 2.7, 4.05, 2.7, ymid + 0.9, color=C_ACCENT, lw=1.1)
    ax.text(6.4, 4.16, "backpropagation", ha="center", va="center",
            fontsize=8, color=C_ACCENT, style="italic")

    fig.tight_layout(pad=0.4)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Draw the OWMM pipeline schematic.")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    path = os.path.abspath(draw(args.out))
    print(f"Saved: {path}")
    print("Reference in manuscript.tex as: figs/pipeline_owmm.png")


if __name__ == "__main__":
    main()
