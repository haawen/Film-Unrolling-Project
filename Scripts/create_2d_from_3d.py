"""
Create a 2D nnU-Net dataset (Dataset503) by extracting Z-slices from 3D volumes.

This ensures the 2D model is trained on the SAME data as the 3D models,
making comparisons fair. The train/val split is derived from the 3D split:
if a 3D volume is in the val set, ALL its slices go to the 2D val set.

Input:  Dataset502_MickeyScroll3D  (25 NIfTI volumes, ~20 slices each)
Output: Dataset503_MickeyScroll2Dfrom3D (500 TIF slices with matched splits)

Usage:
    python Scripts/create_2d_from_3d.py
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import tifffile

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE = Path(__file__).resolve().parent.parent
NNUNET_RAW = BASE / "nnUNet_data" / "nnUNet_raw"
NNUNET_PREPROCESSED = BASE / "nnUNet_data" / "nnUNet_preprocessed"

SRC_DATASET = "Dataset502_MickeyScroll3D"
DST_DATASET = "Dataset503_MickeyScroll2Dfrom3D"
DST_ID = 503

LABELS = {"background": 0, "foreground_1": 1, "foreground_2": 2}


def create_dataset(args):
    src_dir = NNUNET_RAW / SRC_DATASET
    dst_dir = NNUNET_RAW / DST_DATASET
    images_out = dst_dir / "imagesTr"
    labels_out = dst_dir / "labelsTr"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Discover 3D cases
    label_files = sorted((src_dir / "labelsTr").glob("*.nii.gz"))
    print(f"Found {len(label_files)} 3D volumes in {SRC_DATASET}")

    num_slices = 0
    slice_to_volume = {}  # maps 2D case_id -> 3D case_id (for split mapping)

    for lbl_path in label_files:
        case_3d = lbl_path.name.replace(".nii.gz", "")
        img_path = src_dir / "imagesTr" / f"{case_3d}_0000.nii.gz"

        if not img_path.exists():
            print(f"  WARNING: {img_path} not found, skipping")
            continue

        img_vol = nib.load(str(img_path)).get_fdata().astype(np.float32)
        lbl_vol = np.round(nib.load(str(lbl_path)).get_fdata()).astype(np.uint8)

        n_z = img_vol.shape[0]
        # Derive 2D case prefix from 3D case id: Mickey3D_0836 -> Mickey2D_0836
        base_id = case_3d.replace("Mickey3D_", "")

        for z in range(n_z):
            case_2d = f"Mickey2D_{base_id}_z{z:03d}"
            img_out = images_out / f"{case_2d}_0000.tif"
            lbl_out = labels_out / f"{case_2d}.tif"

            if not img_out.exists() or args.force:
                tifffile.imwrite(str(img_out), img_vol[z])
                tifffile.imwrite(str(lbl_out), lbl_vol[z])

            slice_to_volume[case_2d] = case_3d
            num_slices += 1

        print(f"  {case_3d}: {n_z} slices -> {images_out.name}/")

    # Write dataset.json
    dataset_json = {
        "channel_names": {"0": "XRay"},
        "labels": LABELS,
        "numTraining": num_slices,
        "file_ending": ".tif",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    with open(dst_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    # Save slice-to-volume mapping (needed for creating matched splits)
    with open(dst_dir / "slice_to_volume.json", "w") as f:
        json.dump(slice_to_volume, f, indent=2)

    print(f"\nCreated {num_slices} 2D slices in {dst_dir}")

    # ── Create matched 5-fold splits ────────────────────────────────────────
    # Use the same split as the 3D nnU-Net model, mapping volume->slices
    create_matched_splits(slice_to_volume, dst_dir)

    print(f"\nNext steps:")
    print(f"  1. nnUNetv2_plan_and_preprocess -d {DST_ID} --verify_dataset_integrity -c 2d")
    print(f"  2. nnUNetv2_train {DST_ID} 2d 0 -tr nnUNetTrainerProgress")


def create_matched_splits(slice_to_volume: dict, dst_dir: Path):
    """Create 5-fold splits where slices from the same volume stay together."""
    # Get all unique 3D case IDs, sorted for determinism
    all_volumes = sorted(set(slice_to_volume.values()))
    n = len(all_volumes)

    # Create 5-fold split at the volume level (matches sklearn KFold with seed 42)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Build reverse mapping: volume -> list of 2D case IDs
    vol_to_slices = {}
    for slice_id, vol_id in slice_to_volume.items():
        vol_to_slices.setdefault(vol_id, []).append(slice_id)

    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_volumes)):
        train_vols = {all_volumes[i] for i in train_idx}
        val_vols = {all_volumes[i] for i in val_idx}

        train_slices = []
        val_slices = []
        for vol_id in train_vols:
            train_slices.extend(vol_to_slices[vol_id])
        for vol_id in val_vols:
            val_slices.extend(vol_to_slices[vol_id])

        splits.append({
            "train": sorted(train_slices),
            "val": sorted(val_slices),
        })
        print(f"  Fold {fold_idx}: {len(train_slices)} train slices "
              f"({len(train_idx)} vols), {len(val_slices)} val slices "
              f"({len(val_idx)} vols)")
        if fold_idx == 0:
            print(f"    Val volumes: {sorted(val_vols)}")

    # Save splits in nnU-Net format in the preprocessed dir
    prep_dir = NNUNET_PREPROCESSED / DST_DATASET
    prep_dir.mkdir(parents=True, exist_ok=True)
    with open(prep_dir / "splits_final.json", "w") as f:
        json.dump(splits, f, indent=2)
    # Also save in raw dir for reference
    with open(dst_dir / "splits_final.json", "w") as f:
        json.dump(splits, f, indent=2)
    print(f"  Splits saved to {prep_dir / 'splits_final.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Create 2D nnU-Net dataset from 3D volumes")
    p.add_argument("--force", action="store_true", help="Overwrite existing files")
    create_dataset(p.parse_args())
