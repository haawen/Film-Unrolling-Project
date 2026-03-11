"""
Visualization utilities for nnU-Net predictions.

Usage:
    python Scripts/visualize.py                          # visualize all validation cases
    python Scripts/visualize.py --case Mickey_1026       # single case
    python Scripts/visualize.py --save                   # save to disk instead of showing
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import tifffile

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent / "nnUNet_data"
RAW_DIR = BASE / "nnUNet_raw" / "Dataset501_MickeyScroll"
RESULTS_DIR = (
    BASE / "nnUNet_results" / "Dataset501_MickeyScroll"
    / "nnUNetTrainerProgress__nnUNetPlans__2d"
)
IMAGES_DIR = RAW_DIR / "imagesTr"
LABELS_DIR = RAW_DIR / "labelsTr"
OUTPUT_DIR = BASE.parent / "visualizations"

# ─── Label config ────────────────────────────────────────────────────────────
LABEL_NAMES = {0: "background", 1: "foreground_1", 2: "foreground_2"}
LABEL_COLORS = ["black", "#2196F3", "#FF5722"]  # black, blue, orange-red
CMAP = mcolors.ListedColormap(LABEL_COLORS)
NORM = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=3)


def plot_prediction(case_id: str, fold: int = 0, save: bool = False):
    """Show image, ground-truth label, prediction, and per-class probabilities."""
    image_path = IMAGES_DIR / f"{case_id}_0000.tif"
    label_path = LABELS_DIR / f"{case_id}.tif"
    val_dir = RESULTS_DIR / f"fold_{fold}" / "validation"
    pred_path = val_dir / f"{case_id}.tif"
    prob_path = val_dir / f"{case_id}.npz"

    if not pred_path.exists():
        print(f"  Skipping {case_id}: no prediction found for fold {fold}")
        return

    image = tifffile.imread(image_path)
    label = tifffile.imread(label_path)
    pred = tifffile.imread(pred_path)

    n_classes = len(LABEL_NAMES)
    has_probs = prob_path.exists()
    ncols = 3 + n_classes if has_probs else 3

    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    fig.suptitle(f"{case_id}  (fold {fold})", fontsize=14, fontweight="bold")

    # 1) Raw image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Image")

    # 2) Ground-truth label
    axes[1].imshow(label, cmap=CMAP, norm=NORM, interpolation="nearest")
    axes[1].set_title("Ground Truth")

    # 3) Prediction
    axes[2].imshow(pred, cmap=CMAP, norm=NORM, interpolation="nearest")
    axes[2].set_title("Prediction")

    # 4+) Per-class probability maps
    if has_probs:
        probs = np.load(prob_path)["softmax"]  # shape: (C, H, W)
        for c in range(n_classes):
            axes[3 + c].imshow(probs[c], cmap="magma", vmin=0, vmax=1)
            axes[3 + c].set_title(f"P({LABEL_NAMES[c]})")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{case_id}_fold{fold}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")
    else:
        plt.show()


def plot_overlay(case_id: str, fold: int = 0, alpha: float = 0.4, save: bool = False):
    """Overlay prediction on the raw image with transparency."""
    image_path = IMAGES_DIR / f"{case_id}_0000.tif"
    val_dir = RESULTS_DIR / f"fold_{fold}" / "validation"
    pred_path = val_dir / f"{case_id}.tif"

    if not pred_path.exists():
        print(f"  Skipping {case_id}: no prediction found for fold {fold}")
        return

    image = tifffile.imread(image_path)
    pred = tifffile.imread(pred_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{case_id}  (fold {fold}) — Overlay", fontsize=14, fontweight="bold")

    # Image only
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Image")

    # Image + prediction overlay
    axes[1].imshow(image, cmap="gray")
    mask = np.ma.masked_where(pred == 0, pred)
    axes[1].imshow(mask, cmap=CMAP, norm=NORM, alpha=alpha, interpolation="nearest")
    axes[1].set_title("Prediction Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"{case_id}_fold{fold}_overlay.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")
    else:
        plt.show()


def get_validation_cases(fold: int = 0) -> list[str]:
    """List all case IDs that have validation predictions for a given fold."""
    val_dir = RESULTS_DIR / f"fold_{fold}" / "validation"
    return sorted(p.stem for p in val_dir.glob("*.tif"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize nnU-Net predictions")
    parser.add_argument("--case", type=str, default=None, help="Single case ID (e.g. Mickey_1026)")
    parser.add_argument("--fold", type=int, default=0, help="Fold number (default: 0)")
    parser.add_argument("--save", action="store_true", help="Save figures instead of showing")
    parser.add_argument("--overlay", action="store_true", help="Also plot overlay view")
    args = parser.parse_args()

    cases = [args.case] if args.case else get_validation_cases(args.fold)
    print(f"Visualizing {len(cases)} validation case(s) from fold {args.fold}...")

    for case_id in cases:
        print(f"  {case_id}")
        plot_prediction(case_id, fold=args.fold, save=args.save)
        if args.overlay:
            plot_overlay(case_id, fold=args.fold, save=args.save)
