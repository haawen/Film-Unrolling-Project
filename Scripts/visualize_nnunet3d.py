"""
Generate high-quality overlay and mask visualizations for nnU-Net 3D predictions.

Produces per-case, per-slice images at full resolution so you can zoom in and inspect.
- Overlay: grayscale image + semi-transparent coloured prediction
- Mask only: prediction labels with vibrant colours on black background
- Ground truth overlay and mask for comparison

Usage:
    python Scripts/visualize_nnunet3d.py
    python Scripts/visualize_nnunet3d.py --slices 5 10 15   # specific Z slices
    python Scripts/visualize_nnunet3d.py --cases Mickey3D_1263 Mickey3D_1547
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "nnUNet_data" / "nnUNet_raw" / "Dataset502_MickeyScroll3D"
PRED_DIR = (
    PROJECT_DIR
    / "nnUNet_data"
    / "nnUNet_results"
    / "Dataset502_MickeyScroll3D"
    / "nnUNetTrainerProgress__nnUNetPlans__3d_fullres"
    / "fold_0"
    / "validation"
)
OUT_DIR = PROJECT_DIR / "visualizations" / "nnunet3d_hq"

# Validation cases (fold 0)
VAL_CASES = ["Mickey3D_1263", "Mickey3D_1547", "Mickey3D_1669", "Mickey3D_1774", "Mickey3D_1847"]

# ─── Colour scheme ───────────────────────────────────────────────────────────
# Class 0 = yellow background, Class 1 = blue, Class 2 = red
COLORS = {
    0: np.array([255, 255, 0]),   # yellow (background)
    1: np.array([0, 80, 255]),    # blue
    2: np.array([255, 0, 0]),     # red
}
OVERLAY_ALPHA = 0.45


def load_volume(path: Path) -> np.ndarray:
    return nib.load(str(path)).get_fdata()


def make_overlay(image_slice: np.ndarray, mask_slice: np.ndarray, alpha: float) -> np.ndarray:
    """Overlay coloured mask on grayscale image."""
    img = image_slice.astype(np.float64)
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip((img - p1) / max(p99 - p1, 1e-8), 0, 1)
    rgb = np.stack([img, img, img], axis=-1)

    for cls, color in COLORS.items():
        region = mask_slice == cls
        if region.any():
            c = color / 255.0
            rgb[region] = rgb[region] * (1 - alpha) + c * alpha

    return (rgb * 255).astype(np.uint8)


def make_mask_image(mask_slice: np.ndarray) -> np.ndarray:
    """Vibrant mask with yellow background, blue class 1, red class 2."""
    h, w = mask_slice.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in COLORS.items():
        rgb[mask_slice == cls] = color
    return rgb


def save_figure(arr: np.ndarray, path: Path, title: str = ""):
    """Save array as a high-res image with minimal margins."""
    h, w = arr.shape[:2]
    # Use 1:1 pixel mapping at 150 DPI → large but zoomable files
    dpi = 150
    fig, ax = plt.subplots(1, 1, figsize=(w / dpi, h / dpi), dpi=dpi)
    ax.imshow(arr, interpolation="nearest")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8, pad=2)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cases = args.cases if args.cases else VAL_CASES
    print(f"Generating high-quality visualizations for {len(cases)} cases")

    for case_id in cases:
        img_path = RAW_DIR / "imagesTr" / f"{case_id}_0000.nii.gz"
        lbl_path = RAW_DIR / "labelsTr" / f"{case_id}.nii.gz"
        pred_path = PRED_DIR / f"{case_id}.nii.gz"

        if not pred_path.exists():
            print(f"  {case_id}: prediction not found, skipping")
            continue

        print(f"  Loading {case_id}...")
        img_vol = load_volume(img_path)   # (Z, H, W) or (H, W, Z)
        lbl_vol = load_volume(lbl_path).astype(np.int32)
        pred_vol = load_volume(pred_path).astype(np.int32)

        # Crop pred to label shape if needed
        s = lbl_vol.shape
        pred_vol = pred_vol[: s[0], : s[1], : s[2]]

        nz = img_vol.shape[2] if img_vol.ndim == 3 else 1
        print(f"    Volume shape: {img_vol.shape}, pred shape: {pred_vol.shape}, nslices(dim2)={nz}")

        # Determine which axis is Z (the thin one, ~20 slices)
        # For these volumes: shape is (H, W, Z) where Z ~ 20
        min_axis = int(np.argmin(img_vol.shape))
        print(f"    Thin axis (Z): {min_axis} with size {img_vol.shape[min_axis]}")

        n_slices = img_vol.shape[min_axis]

        # Pick slices to visualize
        if args.slices:
            slice_indices = [s for s in args.slices if s < n_slices]
        else:
            # Pick 3 representative slices: 25%, 50%, 75%
            slice_indices = [
                n_slices // 4,
                n_slices // 2,
                3 * n_slices // 4,
            ]

        case_dir = OUT_DIR / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        for z in slice_indices:
            # Extract 2D slices along the thin axis
            if min_axis == 0:
                img_s = img_vol[z, :, :]
                lbl_s = lbl_vol[z, :, :]
                pred_s = pred_vol[z, :, :]
            elif min_axis == 1:
                img_s = img_vol[:, z, :]
                lbl_s = lbl_vol[:, z, :]
                pred_s = pred_vol[:, z, :]
            else:
                img_s = img_vol[:, :, z]
                lbl_s = lbl_vol[:, :, z]
                pred_s = pred_vol[:, :, z]

            tag = f"z{z:03d}"

            # 1. Prediction overlay on image
            overlay_pred = make_overlay(img_s, pred_s, OVERLAY_ALPHA)
            save_figure(overlay_pred, case_dir / f"{tag}_pred_overlay.png",
                        f"{case_id} z={z} — nnU-Net 3D prediction overlay")

            # 2. Prediction mask only
            mask_pred = make_mask_image(pred_s)
            save_figure(mask_pred, case_dir / f"{tag}_pred_mask.png",
                        f"{case_id} z={z} — nnU-Net 3D prediction mask")

            # 3. Ground truth overlay on image
            overlay_gt = make_overlay(img_s, lbl_s, OVERLAY_ALPHA)
            save_figure(overlay_gt, case_dir / f"{tag}_gt_overlay.png",
                        f"{case_id} z={z} — Ground truth overlay")

            # 4. Ground truth mask only
            mask_gt = make_mask_image(lbl_s)
            save_figure(mask_gt, case_dir / f"{tag}_gt_mask.png",
                        f"{case_id} z={z} — Ground truth mask")

            # 5. Side-by-side: GT overlay | Pred overlay (single wide figure)
            h, w = img_s.shape
            dpi = 150
            fig, axes = plt.subplots(1, 2, figsize=(2 * w / dpi, h / dpi), dpi=dpi)
            axes[0].imshow(overlay_gt, interpolation="nearest")
            axes[0].set_title("Ground Truth", fontsize=10)
            axes[0].axis("off")
            axes[1].imshow(overlay_pred, interpolation="nearest")
            axes[1].set_title("nnU-Net 3D Prediction", fontsize=10)
            axes[1].axis("off")
            fig.suptitle(f"{case_id}  z={z}", fontsize=12, y=0.99)
            fig.savefig(str(case_dir / f"{tag}_comparison.png"), dpi=dpi,
                        bbox_inches="tight", pad_inches=0.02)
            plt.close(fig)

            print(f"    Saved z={z}: overlay, mask, GT, comparison")

    print(f"\nAll visualizations saved to: {OUT_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="+", default=None, help="Case IDs to visualize")
    p.add_argument("--slices", type=int, nargs="+", default=None, help="Z slice indices")
    main(p.parse_args())
