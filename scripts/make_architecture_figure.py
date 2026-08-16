"""
Generate the thesis architecture figure as vector PDF + 300 dpi PNG.

    python scripts/make_architecture_figure.py --out docs/figures

Produces ``architecture.pdf`` (for \\includegraphics) and ``architecture.png``.
Pure matplotlib -- no graphviz or LaTeX dependency, so it runs anywhere the
project runs.
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Colour-blind-safe palette (Okabe-Ito), legible in greyscale print.
PALETTE = {
    "data":    "#E8F1F8",
    "encoder": "#FDF0E3",
    "method":  "#EAF4EA",
    "eval":    "#F3ECF7",
    "output":  "#FBEAEA",
    "accent":  "#0072B2",
    "edge":    "#4A4A4A",
    "text":    "#1A1A1A",
    "novel":   "#D55E00",
}


def box(ax, x, y, w, h, label, facecolor, *, fontsize=8.2, bold=False,
        edgecolor=None, lw=1.0, zorder=3, style="round,pad=0.012,rounding_size=0.014"):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle=style,
        facecolor=facecolor, edgecolor=edgecolor or PALETTE["edge"],
        linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, label,
        ha="center", va="center", fontsize=fontsize,
        color=PALETTE["text"], zorder=zorder + 1,
        fontweight="bold" if bold else "normal", linespacing=1.45,
    )
    return patch


TITLE_BAND = 0.030   # vertical space reserved for a group's heading


def group(ax, x, y, w, h, title, facecolor):
    """Draw a group container. Boxes must stay below ``y + h - TITLE_BAND``."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.016",
        facecolor=facecolor, edgecolor="#B8B8B8", linewidth=1.1,
        linestyle="-", zorder=1,
    ))
    ax.text(x + 0.014, y + h - 0.010, title, ha="left", va="top",
            fontsize=9.4, fontweight="bold", color=PALETTE["accent"], zorder=2)


def arrow(ax, start, end, *, style="-|>", color=None, lw=1.3, ls="-",
          connection="arc3,rad=0.0", zorder=6):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=11,
        color=color or PALETTE["edge"], linewidth=lw, linestyle=ls,
        connectionstyle=connection, zorder=zorder,
        shrinkA=2, shrinkB=2,
    ))


def build_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.4, 9.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---------------------------------------------------------- data layer --
    g1_y, g1_h = 0.800, 0.170
    group(ax, 0.02, g1_y, 0.96, g1_h, "1 · Data layer", PALETTE["data"])
    r1_y, r1_h = 0.812, 0.098
    box(ax, 0.045, r1_y, 0.200, r1_h,
        "Dataset registry\n5 species × {fresh, rotten}\n6,978 images", "#FFFFFF")
    box(ax, 0.275, r1_y, 0.210, r1_h,
        "LOSO species splitter\n5 folds, seeded\nleakage assertions", "#FFFFFF")
    box(ax, 0.515, r1_y, 0.235, r1_h,
        "Deterministic episode bank\nfrozen (support, query) specs\n"
        "nested shots  1 " + chr(8838) + " 3 " + chr(8838) + " 5 " + chr(8838) + " 10",
        "#FFFFFF", edgecolor=PALETTE["accent"], lw=1.9, bold=True)
    box(ax, 0.780, r1_y, 0.175, r1_h,
        "Shared across\nEVERY method\n" + chr(8658) + " paired tests valid",
        "#FFFFFF", fontsize=7.8)

    mid1 = r1_y + r1_h / 2
    arrow(ax, (0.245, mid1), (0.275, mid1))
    arrow(ax, (0.485, mid1), (0.515, mid1))
    arrow(ax, (0.750, mid1), (0.780, mid1), color=PALETTE["accent"], lw=1.7)

    # --------------------------------------------- encoder / adaptation -----
    g2_y, g2_h = 0.520, 0.240
    group(ax, 0.02, g2_y, 0.465, g2_h, "2 · Encoder layer", PALETTE["encoder"])
    group(ax, 0.505, g2_y, 0.475, g2_h,
          "3 · Adaptation layer (FewShotMethod)", PALETTE["method"])

    ra_y, rb_y, rh = 0.638, 0.556, 0.068
    box(ax, 0.042, ra_y, 0.200, rh,
        "ResNet-18 / 50, EfficientNet\nfine-tuned, freeze depth k", "#FFFFFF", fontsize=7.8)
    box(ax, 0.262, ra_y, 0.200, rh,
        "DINOv2 ViT-S/14\nfrozen + feature cache", "#FFFFFF", fontsize=7.8)
    box(ax, 0.042, rb_y, 0.200, rh,
        "CLIP ViT-B/16 image tower\nfrozen + feature cache", "#FFFFFF", fontsize=7.8)
    box(ax, 0.262, rb_y, 0.200, rh,
        "CLIP text tower\n\"a photo of a {q} {species}\"", "#FFFFFF",
        fontsize=7.8, edgecolor=PALETTE["novel"], lw=1.7)

    box(ax, 0.523, ra_y, 0.215, rh,
        "zero-shot supervised (K=0)\nTHE PIVOTAL CONTROL", "#FFFFFF",
        fontsize=7.6, bold=True, edgecolor=PALETTE["accent"], lw=1.8)
    box(ax, 0.752, ra_y, 0.212, rh,
        "nearest centroid / linear\nprobe / support fine-tune", "#FFFFFF", fontsize=7.6)
    box(ax, 0.523, rb_y, 0.215, rh,
        "Siamese / Matching /\nProtoNet " + chr(177) + " temperature", "#FFFFFF", fontsize=7.6)
    box(ax, 0.752, rb_y, 0.212, rh,
        "SAP (proposed)\n" + chr(945) + "·visual + (1" + chr(8722) + chr(945) + ")·text",
        "#FFFFFF", fontsize=7.6, bold=True, edgecolor=PALETTE["novel"], lw=1.9)
    ax.text(0.7425, 0.535,
            "train-time regularisers:  supervised contrastive  |  species-adversarial GRL",
            ha="center", va="center", fontsize=7.3, style="italic", color="#555555")

    arrow(ax, (0.253, g1_y), (0.253, g2_y + g2_h))
    arrow(ax, (0.462, ra_y + rh / 2), (0.523, ra_y + rh / 2))
    # Text tower feeds SAP specifically. Routed through the inter-row gap so it
    # does not clip the Siamese/Matching box.
    arrow(ax, (0.462, rb_y + rh / 2), (0.752, rb_y + rh / 2),
          color=PALETTE["novel"], lw=1.6, connection="arc3,rad=-0.32")

    # ---------------------------------------------------- evaluation layer --
    g3_y, g3_h = 0.250, 0.245
    group(ax, 0.02, g3_y, 0.96, g3_h, "4 · Evaluation layer", PALETTE["eval"])
    r3_y, r3_h = 0.330, 0.115
    box(ax, 0.042, r3_y, 0.215, r3_h,
        "Metric suite\nAcc  BalAcc  P/R/F1\nSens  Spec  ROC-AUC\n"
        "PR-AUC  " + chr(954) + "  MCC  ECE", "#FFFFFF", fontsize=7.7)
    box(ax, 0.277, r3_y, 0.225, r3_h,
        "Intervals, 3 units\nover episodes (t)\nover images (cluster boot.)\n"
        "over folds (t, n=5)", "#FFFFFF", fontsize=7.7,
        edgecolor=PALETTE["accent"], lw=1.6)
    box(ax, 0.522, r3_y, 0.215, r3_h,
        "Paired statistics\nsame episodes only\nWilcoxon (Pratt ties)\n"
        "Holm" + chr(8211) + "Bonferroni", "#FFFFFF", fontsize=7.7,
        edgecolor=PALETTE["accent"], lw=1.6)
    box(ax, 0.757, r3_y, 0.205, r3_h,
        "Cost\nparams / FLOPs\ntrain wall-clock\ninference latency",
        "#FFFFFF", fontsize=7.7)
    ax.text(0.5, 0.283,
            "every interval carries its ci_method, n and unit " + chr(8212) +
            " two differently-scoped intervals can never share a column",
            ha="center", va="center", fontsize=7.4, style="italic", color="#555555")

    arrow(ax, (0.253, g2_y), (0.253, g3_y + g3_h))
    arrow(ax, (0.742, g2_y), (0.742, g3_y + g3_h))

    # ------------------------------------------------------------- outputs --
    g4_y, g4_h = 0.030, 0.190
    group(ax, 0.02, g4_y, 0.96, g4_h, "5 · Serialized artifacts", PALETTE["output"])
    r4_y, r4_h = 0.062, 0.098
    box(ax, 0.042, r4_y, 0.215, r4_h,
        "per_query.csv.gz\nSOURCE OF TRUTH\nevery prediction", "#FFFFFF",
        fontsize=7.7, bold=True, edgecolor=PALETTE["accent"], lw=1.8)
    box(ax, 0.277, r4_y, 0.215, r4_h,
        "metrics.json\ncomparisons.json\nper_episode.csv", "#FFFFFF", fontsize=7.7)
    box(ax, 0.522, r4_y, 0.215, r4_h,
        "config.yaml   env.json\nstatus.json   run.log\ncheckpoints/",
        "#FFFFFF", fontsize=7.7)
    box(ax, 0.757, r4_y, 0.205, r4_h,
        "figures/  tables/\nregenerated ONLY\nfrom artifacts", "#FFFFFF", fontsize=7.7)

    arrow(ax, (0.253, g3_y), (0.253, g4_y + g4_h))
    mid4 = r4_y + r4_h / 2
    arrow(ax, (0.257, mid4), (0.277, mid4))
    arrow(ax, (0.492, mid4), (0.522, mid4))
    arrow(ax, (0.737, mid4), (0.757, mid4))

    # Feedback: tables/figures are recomputed from per_query without a GPU.
    # Straight, in the clear channel below the boxes, so it crosses nothing.
    arrow(ax, (0.859, 0.048), (0.150, 0.048),
          color=PALETTE["accent"], lw=1.2, ls=(0, (4, 3)), style="-|>")
    ax.text(0.962, g4_y + g4_h - 0.014,
            "any new metric recomputed on CPU in seconds " + chr(8212) + " no retraining",
            ha="right", va="top", fontsize=7.2, style="italic",
            color=PALETTE["accent"])

    # ---------------------------------------------------------- title/key ---
    fig.suptitle(
        "Cross-Species Quality Grading: system architecture",
        fontsize=13.2, fontweight="bold", y=0.985, color=PALETTE["text"],
    )
    handles = [
        mpatches.Patch(facecolor="#FFFFFF", edgecolor=PALETTE["accent"],
                       linewidth=1.8, label="methodological fix"),
        mpatches.Patch(facecolor="#FFFFFF", edgecolor=PALETTE["novel"],
                       linewidth=1.8, label="proposed contribution (SAP)"),
    ]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.995, 0.028),
              fontsize=7.6, frameon=False, ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.965])
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description="Render the thesis architecture figure.")
    parser.add_argument("--out", default="docs/figures", help="Output directory")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig = build_figure()
    pdf = out / "architecture.pdf"
    png = out / "architecture.png"
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"wrote {pdf}")
    print(f"wrote {png}  ({args.dpi} dpi)")
    print("\nLaTeX usage:")
    print(r"  \begin{figure}[t]\centering")
    print(r"    \includegraphics[width=\textwidth]{figures/architecture.pdf}")
    print(r"    \caption{System architecture for cross-species quality grading.}")
    print(r"    \label{fig:architecture}")
    print(r"  \end{figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
