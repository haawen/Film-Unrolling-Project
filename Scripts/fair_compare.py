"""
Fair head-to-head comparison of ALL models on the SAME 3D validation data.

For the 2D model (nnU-Net 2D), we extract each Z-slice from the 3D volumes,
run inference per-slice, and reassemble into a 3D volume before computing
metrics — so every model is evaluated on identical voxels.

Steps:
  1. Extract 2D slices from 3D validation volumes → temp input dir
  2. Run nnU-Net 2D prediction on those slices (done externally via SLURM)
  3. Reassemble 2D predictions into 3D volumes
  4. Compute per-case Dice/IoU for every model on the same 3D volumes
  5. Print tables + generate comparison figures

Usage:
    # Step 1: Prepare slices (run before SLURM job)
    python Scripts/fair_compare.py --prepare

    # Step 2: After nnU-Net 2D inference completes, run comparison
    python Scripts/fair_compare.py --compare --save
"""

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parent.parent))
NNUNET_RAW = BASE / "nnUNet_data" / "nnUNet_raw"
NNUNET_RESULTS = BASE / "nnUNet_data" / "nnUNet_results"
MONAI_RESULTS = BASE / "monai_results"

DS_3D = "Dataset502_MickeyScroll3D"
DS_2D = "Dataset501_MickeyScroll"
DS_2D_FROM_3D = "Dataset503_MickeyScroll2Dfrom3D"

FAIR_DIR = BASE / "fair_comparison"
SLICES_INPUT_DIR = FAIR_DIR / "slices_input"     # 2D slices for nnU-Net predict
SLICES_OUTPUT_DIR = FAIR_DIR / "slices_output"    # nnU-Net 2D predictions
REASSEMBLED_DIR = FAIR_DIR / "nnunet2d_on_3d"     # reassembled 3D predictions

NUM_CLASSES = 3
LABEL_NAMES = {1: "foreground_1", 2: "foreground_2"}

# ─── Which 3D validation cases to use ───────────────────────────────────────

def get_3d_val_cases(fold: int = 0) -> list[str]:
    """Get 3D validation case IDs from nnU-Net 3D predictions (ground truth source)."""
    pred_dir = (NNUNET_RESULTS / DS_3D /
                "nnUNetTrainerProgress__nnUNetPlans__3d_fullres" /
                f"fold_{fold}" / "validation")
    if pred_dir.exists():
        return sorted([p.name.replace(".nii.gz", "") for p in pred_dir.glob("*.nii.gz")])

    # Fallback: check MONAI predictions
    for model in ["UNet3D", "SwinUNETR"]:
        pred_dir = MONAI_RESULTS / DS_3D / model / f"fold_{fold}" / "validation"
        if pred_dir.exists():
            cases = [p.name.replace(".nii.gz", "") for p in pred_dir.glob("*.nii.gz")]
            return sorted(cases)
    return []


# ─── Step 1: Extract 2D slices ──────────────────────────────────────────────

def prepare_slices(fold: int = 0):
    """Extract each Z-slice from 3D val volumes as .tif files for nnU-Net 2D."""
    import tifffile

    cases = get_3d_val_cases(fold)
    if not cases:
        print("No 3D validation cases found!")
        return

    raw3d = NNUNET_RAW / DS_3D
    SLICES_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {}  # case_id → {n_slices, shape}

    for case_id in cases:
        img_path = raw3d / "imagesTr" / f"{case_id}_0000.nii.gz"
        if not img_path.exists():
            print(f"  WARNING: {img_path} not found, skipping")
            continue

        vol = nib.load(str(img_path)).get_fdata()
        n_z = vol.shape[0]
        manifest[case_id] = {"n_slices": n_z, "shape": list(vol.shape)}

        for z in range(n_z):
            # nnU-Net 2D expects: {case_id}_0000.tif
            slice_name = f"{case_id}_z{z:03d}_0000.tif"
            tifffile.imwrite(str(SLICES_INPUT_DIR / slice_name), vol[z].astype(np.float32))

        print(f"  {case_id}: {n_z} slices extracted ({vol.shape})")

    # Save manifest for reassembly
    manifest_path = FAIR_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")
    print(f"Slices saved to {SLICES_INPUT_DIR}")
    print(f"Total slices: {sum(m['n_slices'] for m in manifest.values())}")
    print(f"\nNext: run nnU-Net 2D prediction on {SLICES_INPUT_DIR}")


# ─── Step 2: Reassemble 2D predictions into 3D ──────────────────────────────

def reassemble_predictions():
    """Stack 2D nnU-Net predictions back into 3D volumes."""
    import tifffile

    manifest_path = FAIR_DIR / "manifest.json"
    if not manifest_path.exists():
        print("No manifest.json found. Run --prepare first.")
        return

    manifest = json.loads(manifest_path.read_text())
    REASSEMBLED_DIR.mkdir(parents=True, exist_ok=True)

    raw3d = NNUNET_RAW / DS_3D

    for case_id, info in manifest.items():
        n_z = info["n_slices"]
        shape = tuple(info["shape"])
        slices = []

        for z in range(n_z):
            pred_name = f"{case_id}_z{z:03d}.tif"
            pred_path = SLICES_OUTPUT_DIR / pred_name
            if not pred_path.exists():
                print(f"  WARNING: {pred_path} not found!")
                slices.append(np.zeros(shape[1:], dtype=np.uint8))
            else:
                slices.append(tifffile.imread(str(pred_path)).astype(np.uint8))

        vol_pred = np.stack(slices, axis=0)

        # Get affine from original volume
        ref_path = raw3d / "imagesTr" / f"{case_id}_0000.nii.gz"
        ref_nii = nib.load(str(ref_path))

        pred_nii = nib.Nifti1Image(vol_pred, affine=ref_nii.affine)
        out_path = REASSEMBLED_DIR / f"{case_id}.nii.gz"
        nib.save(pred_nii, str(out_path))
        print(f"  {case_id}: reassembled {vol_pred.shape}, classes={np.unique(vol_pred)}")

    print(f"Reassembled volumes saved to {REASSEMBLED_DIR}")


# ─── Step 2b: Reassemble nnU-Net 2D (from Dataset503) val predictions ────────

def reassemble_2d_from_3d(fold: int = 0):
    """Reassemble nnU-Net 2D (Dataset503) validation predictions into 3D volumes.

    nnU-Net stores val predictions in:
      nnUNet_results/Dataset503_.../nnUNetTrainerProgress__nnUNetPlans__2d/fold_0/validation/
    as .tif files like Mickey2D_0836_z000.tif.
    We stack them back into 3D NIfTI volumes matching the 3D case IDs.
    """
    import re
    import tifffile

    val_dir = (NNUNET_RESULTS / DS_2D_FROM_3D /
               "nnUNetTrainerProgress__nnUNetPlans__2d" /
               f"fold_{fold}" / "validation")

    if not val_dir.exists():
        print(f"  No validation predictions found at {val_dir}")
        return

    REASSEMBLED_2DFROM3D_DIR.mkdir(parents=True, exist_ok=True)
    raw3d = NNUNET_RAW / DS_3D

    # Group prediction files by their source 3D volume
    # Filename: Mickey2D_{volid}_z{ZZZ}.tif
    vol_slices: dict[str, dict[int, Path]] = {}
    for pred_file in sorted(val_dir.glob("Mickey2D_*.tif")):
        m = re.match(r"Mickey2D_(\d+)_z(\d+)\.tif", pred_file.name)
        if not m:
            continue
        vol_id = m.group(1)
        z_idx = int(m.group(2))
        case_3d = f"Mickey3D_{vol_id}"
        vol_slices.setdefault(case_3d, {})[z_idx] = pred_file

    if not vol_slices:
        # Try .npz format (nnU-Net with --npz)
        for pred_file in sorted(val_dir.glob("Mickey2D_*.npz")):
            m = re.match(r"Mickey2D_(\d+)_z(\d+)\.npz", pred_file.name)
            if not m:
                continue
            vol_id = m.group(1)
            z_idx = int(m.group(2))
            case_3d = f"Mickey3D_{vol_id}"
            vol_slices.setdefault(case_3d, {})[z_idx] = pred_file

    for case_3d, slices_dict in sorted(vol_slices.items()):
        n_z = max(slices_dict.keys()) + 1
        # Read reference for affine and shape
        ref_path = raw3d / "imagesTr" / f"{case_3d}_0000.nii.gz"
        ref_nii = nib.load(str(ref_path))

        assembled = []
        for z in range(n_z):
            if z not in slices_dict:
                print(f"    WARNING: missing z={z} for {case_3d}")
                assembled.append(np.zeros(ref_nii.shape[1:3], dtype=np.uint8))
                continue
            p = slices_dict[z]
            if p.suffix == ".npz":
                assembled.append(np.load(str(p))["softmax"].argmax(0).astype(np.uint8))
            else:
                assembled.append(tifffile.imread(str(p)).astype(np.uint8))

        vol = np.stack(assembled, axis=0)
        out_path = REASSEMBLED_2DFROM3D_DIR / f"{case_3d}.nii.gz"
        nib.save(nib.Nifti1Image(vol, affine=ref_nii.affine), str(out_path))
        print(f"  {case_3d}: reassembled {vol.shape}, classes={np.unique(vol)}")

    print(f"  Saved to {REASSEMBLED_2DFROM3D_DIR}")


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(pred: np.ndarray, label: np.ndarray) -> dict:
    """Compute per-class Dice, IoU, Precision, Recall."""
    results = {}
    for cls in range(1, NUM_CLASSES):
        p = (pred == cls)
        r = (label == cls)
        tp = float(np.logical_and(p, r).sum())
        fp = float(np.logical_and(p, ~r).sum())
        fn = float(np.logical_and(~p, r).sum())

        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

        results[str(cls)] = {"Dice": dice, "IoU": iou, "Precision": prec, "Recall": rec}
    return results


# ─── Step 3: Fair comparison ─────────────────────────────────────────────────

MODEL_SOURCES = {
    "nnU-Net 2D (matched)": ("nnunet2d_from3d", "nnUNetTrainerProgress__nnUNetPlans__2d"),
    "nnU-Net 2D (orig)": ("reassembled", None),
    "nnU-Net 3D": ("nnunet3d", "nnUNetTrainerProgress__nnUNetPlans__3d_fullres"),
    "SwinUNETR": ("monai", "SwinUNETR"),
    "UNet3D": ("monai", "UNet3D"),
}


REASSEMBLED_2DFROM3D_DIR = FAIR_DIR / "nnunet2d_from3d"  # reassembled from Dataset503


def get_prediction_path(model_key: str, case_id: str, fold: int = 0) -> Path | None:
    source_type, source_name = MODEL_SOURCES[model_key]

    if source_type == "reassembled":
        p = REASSEMBLED_DIR / f"{case_id}.nii.gz"
    elif source_type == "nnunet2d_from3d":
        p = REASSEMBLED_2DFROM3D_DIR / f"{case_id}.nii.gz"
    elif source_type == "nnunet3d":
        p = (NNUNET_RESULTS / DS_3D / source_name /
             f"fold_{fold}" / "validation" / f"{case_id}.nii.gz")
    elif source_type == "monai":
        p = (MONAI_RESULTS / DS_3D / source_name /
             f"fold_{fold}" / "validation" / f"{case_id}.nii.gz")
    else:
        return None

    return p if p.exists() else None


def run_comparison(save: bool = False, fold: int = 0):
    """Compare all models on the same 3D validation volumes."""
    cases = get_3d_val_cases(fold)
    if not cases:
        print("No validation cases found!")
        return

    raw3d = NNUNET_RAW / DS_3D

    # Check which models have predictions available
    available_models = []
    for model_key in MODEL_SOURCES:
        has_any = any(get_prediction_path(model_key, c, fold) is not None for c in cases)
        if has_any:
            available_models.append(model_key)
        else:
            print(f"  Skipping {model_key}: no predictions found")

    if not available_models:
        print("No model predictions found!")
        return

    print(f"\nFair comparison on {len(cases)} 3D validation volumes:")
    print(f"  Models: {', '.join(available_models)}")
    print(f"  Cases:  {', '.join(cases)}")

    # Compute metrics for each model × case
    # results[model][case_id] = {cls: {Dice, IoU, Prec, Rec}}
    results: dict[str, dict[str, dict]] = {m: {} for m in available_models}

    for case_id in cases:
        lbl_path = raw3d / "labelsTr" / f"{case_id}.nii.gz"
        if not lbl_path.exists():
            print(f"  WARNING: label {lbl_path} not found, skipping")
            continue

        label = np.round(nib.load(str(lbl_path)).get_fdata()).astype(int)

        for model_key in available_models:
            pred_path = get_prediction_path(model_key, case_id, fold)
            if pred_path is None:
                continue

            pred = np.round(nib.load(str(pred_path)).get_fdata()).astype(int)

            # Handle shape mismatch (padding from inference)
            if pred.shape != label.shape:
                pred = pred[:label.shape[0], :label.shape[1], :label.shape[2]]

            metrics = compute_metrics(pred, label)
            results[model_key][case_id] = metrics

    # ── Print table ──────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("  FAIR COMPARISON — All models evaluated on identical 3D validation volumes")
    print("=" * 110)

    header = f"{'Model':<22} {'Case':<18} {'Class':<15} {'Dice':>8} {'IoU':>8} {'Prec':>8} {'Rec':>8}"
    print(header)
    print("-" * len(header))

    for model_key in available_models:
        for case_id in cases:
            if case_id not in results[model_key]:
                continue
            for cls in sorted(results[model_key][case_id].keys()):
                m = results[model_key][case_id][cls]
                cls_name = LABEL_NAMES.get(int(cls), f"class_{cls}")
                print(f"{model_key:<22} {case_id:<18} {cls_name:<15} "
                      f"{m['Dice']:>8.4f} {m['IoU']:>8.4f} "
                      f"{m['Precision']:>8.4f} {m['Recall']:>8.4f}")

    # ── Aggregate means ──────────────────────────────────────────────────
    print("\n" + "-" * 110)
    print(f"{'Model':<22} {'FG Dice':>10} {'FG IoU':>10} {'FG Prec':>10} {'FG Rec':>10} {'Cases':>8}")
    print("-" * 75)

    model_fg = {}
    for model_key in available_models:
        all_dice, all_iou, all_prec, all_rec = [], [], [], []
        for case_id, case_metrics in results[model_key].items():
            for cls, m in case_metrics.items():
                all_dice.append(m["Dice"])
                all_iou.append(m["IoU"])
                all_prec.append(m["Precision"])
                all_rec.append(m["Recall"])

        fg_dice = float(np.nanmean(all_dice)) if all_dice else 0
        fg_iou = float(np.nanmean(all_iou)) if all_iou else 0
        fg_prec = float(np.nanmean(all_prec)) if all_prec else 0
        fg_rec = float(np.nanmean(all_rec)) if all_rec else 0
        n_cases = len(results[model_key])

        model_fg[model_key] = {"Dice": fg_dice, "IoU": fg_iou, "Precision": fg_prec, "Recall": fg_rec}
        print(f"{model_key:<22} {fg_dice:>10.4f} {fg_iou:>10.4f} {fg_prec:>10.4f} {fg_rec:>10.4f} {n_cases:>8}")

    print("=" * 110)

    # ── Save results JSON ────────────────────────────────────────────────
    FAIR_DIR.mkdir(parents=True, exist_ok=True)
    out = {"per_case": {m: {c: results[m][c] for c in results[m]} for m in available_models},
           "aggregate": model_fg}
    with open(FAIR_DIR / "fair_comparison_results.json", "w") as f:
        json.dump(out, f, indent=2)

    # ── Plots ────────────────────────────────────────────────────────────
    _plot_fair_comparison(available_models, results, model_fg, cases, save)
    _plot_fair_overlays(available_models, cases, fold, save)


# ─── Plots ───────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PALETTE = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
OVERLAY_COLORS = {
    1: np.array([0, 200, 255], dtype=np.uint8),   # cyan
    2: np.array([255, 80, 0], dtype=np.uint8),     # orange-red
}
OVERLAY_ALPHA = 0.45


def _save_or_show(fig, filename: str, save: bool):
    out_dir = BASE / "visualizations"
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
    else:
        plt.show()


def _plot_fair_comparison(models, results, model_fg, cases, save):
    """Grouped bar chart of FG Dice per model (fair comparison)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: per-class mean Dice
    n_models = len(models)
    classes = sorted(LABEL_NAMES.keys())
    n_cls = len(classes)
    x = np.arange(n_models)
    width = 0.35

    for ax, metric, title in [(axes[0], "Dice", "Dice Score"), (axes[1], "IoU", "IoU")]:
        for j, cls in enumerate(classes):
            vals = []
            for model_key in models:
                case_vals = [results[model_key][c][str(cls)][metric]
                             for c in results[model_key] if str(cls) in results[model_key][c]]
                vals.append(float(np.nanmean(case_vals)) if case_vals else 0)
            offset = (j - (n_cls - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width,
                         label=LABEL_NAMES[cls], color=PALETTE[j], alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9, rotation=15, ha="right")
        all_vals = [v for m in models for cls in classes
                    for c in results[m]
                    for v in [results[m][c].get(str(cls), {}).get(metric, 0)]]
        ymin = max(0, min(v for v in all_vals if v > 0) - 0.03) if all_vals else 0
        ax.set_ylim(ymin, 1.02)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Fair Comparison — Same 3D Validation Data", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, "fair_comparison.png", save)

    # Box plot: per-case FG Dice
    fig2, ax2 = plt.subplots(figsize=(max(8, 2.5 * n_models), 5))
    box_data = []
    for model_key in models:
        dices = []
        for case_id, case_metrics in results[model_key].items():
            for cls, m in case_metrics.items():
                dices.append(m["Dice"])
        box_data.append(dices)

    bp = ax2.boxplot(box_data, tick_labels=models, patch_artist=True,
                     medianprops=dict(color="black"))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax2.set_ylabel("Dice")
    ax2.set_title("Per-Case Foreground Dice — Fair Comparison", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save_or_show(fig2, "fair_per_case.png", save)


def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(img, [2, 98])
    if p98 - p2 < 1e-6:
        p2, p98 = float(img.min()), float(img.max())
    if p98 - p2 < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    clipped = np.clip((img.astype(np.float32) - p2) / (p98 - p2), 0, 1)
    return (clipped * 255).astype(np.uint8)


def _make_overlay(gray: np.ndarray, seg: np.ndarray, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    gray_u8 = _normalize_to_uint8(gray)
    rgb = np.stack([gray_u8, gray_u8, gray_u8], axis=-1).astype(np.float32)
    for cls, color in OVERLAY_COLORS.items():
        mask = seg == cls
        if mask.any():
            rgb[mask] = rgb[mask] * (1 - alpha) + color.astype(np.float32) * alpha
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _plot_fair_overlays(models, cases, fold, save, max_cases=3):
    """Side-by-side overlays: Raw | GT | model1 | model2 | ... for same cases."""
    raw3d = NNUNET_RAW / DS_3D
    n_models = len(models)

    # Select representative cases
    selected = cases[:max_cases]
    n_cases = len(selected)
    n_cols = 2 + n_models  # raw + GT + one per model

    fig, axes = plt.subplots(n_cases, n_cols, figsize=(5 * n_cols, 5 * n_cases), squeeze=False)

    for row, case_id in enumerate(selected):
        img_path = raw3d / "imagesTr" / f"{case_id}_0000.nii.gz"
        lbl_path = raw3d / "labelsTr" / f"{case_id}.nii.gz"

        img_vol = nib.load(str(img_path)).get_fdata()
        lbl_vol = np.round(nib.load(str(lbl_path)).get_fdata()).astype(int)
        z_mid = img_vol.shape[0] // 2

        img_slice = img_vol[z_mid]
        lbl_slice = lbl_vol[z_mid]

        # Raw
        axes[row, 0].imshow(_normalize_to_uint8(img_slice), cmap="gray")
        axes[row, 0].set_title(f"Raw — {case_id}", fontsize=9)

        # GT
        axes[row, 1].imshow(_make_overlay(img_slice, lbl_slice))
        axes[row, 1].set_title("Ground Truth", fontsize=9)

        # Each model
        for col, model_key in enumerate(models, start=2):
            pred_path = get_prediction_path(model_key, case_id, fold)
            if pred_path is not None:
                pred_vol = np.round(nib.load(str(pred_path)).get_fdata()).astype(int)
                if pred_vol.shape != lbl_vol.shape:
                    pred_vol = pred_vol[:lbl_vol.shape[0], :lbl_vol.shape[1], :lbl_vol.shape[2]]
                pred_slice = pred_vol[z_mid]
                axes[row, col].imshow(_make_overlay(img_slice, pred_slice))
            else:
                axes[row, col].text(0.5, 0.5, "N/A", ha="center", va="center",
                                   transform=axes[row, col].transAxes, fontsize=14)
            axes[row, col].set_title(model_key, fontsize=9)

        for ax in axes[row]:
            ax.axis("off")

    legend_elements = [
        Patch(facecolor=np.array(OVERLAY_COLORS[1]) / 255, label="foreground_1"),
        Patch(facecolor=np.array(OVERLAY_COLORS[2]) / 255, label="foreground_2"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Fair Comparison — All Models on Same Validation Data (mid Z-slice)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, "fair_overlay_comparison.png", save)


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fair model comparison on identical data")
    p.add_argument("--prepare", action="store_true",
                   help="Extract 2D slices from 3D val volumes")
    p.add_argument("--reassemble", action="store_true",
                   help="Reassemble 2D predictions into 3D volumes")
    p.add_argument("--compare", action="store_true",
                   help="Run comparison (requires all predictions)")
    p.add_argument("--save", action="store_true",
                   help="Save figures instead of showing")
    p.add_argument("--fold", type=int, default=0)
    args = p.parse_args()

    if args.prepare:
        print("Extracting 2D slices from 3D validation volumes...")
        prepare_slices(args.fold)
    elif args.reassemble:
        print("Reassembling 2D predictions into 3D volumes...")
        reassemble_predictions()
        reassemble_2d_from_3d(args.fold)
    elif args.compare:
        print("Running fair comparison...")
        reassemble_predictions()
        reassemble_2d_from_3d(args.fold)
        run_comparison(save=args.save, fold=args.fold)
    else:
        p.print_help()
