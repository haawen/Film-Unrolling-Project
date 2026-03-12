"""
Convert 3D HDF5 volumes to nnU-Net / MONAI NIfTI format.

Input (01_Mickey_3d/):
    volume_XXXX-YYYY.h5               → key 'volume',        shape (Z, H, W), uint16
    volume_XXXX-YYYY_Probabilities.h5  → key 'exported_data', shape (Z, H, W, 3), float32

Output (nnUNet_data/nnUNet_raw/Dataset502_MickeyScroll3D/):
    imagesTr/Mickey3D_XXXX_0000.nii.gz   (single-channel, float32, z-y-x)
    labelsTr/Mickey3D_XXXX.nii.gz        (uint8 segmentation via argmax)
    dataset.json

Usage:
    python Scripts/convert_3d_data.py                  # convert all
    python Scripts/convert_3d_data.py --dataset-id 502 # custom dataset ID
"""

import argparse
import json
import re
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np

# ─── Defaults ────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
HDF5_DIR = BASE_DIR / "01_Mickey_3d"
NNUNET_RAW = BASE_DIR / "nnUNet_data" / "nnUNet_raw"

DATASET_ID = 502
DATASET_PREFIX = "MickeyScroll3D"

LABELS = {
    "background": 0,
    "foreground_1": 1,
    "foreground_2": 2,
}

# Simple 1mm isotropic affine (no real-world coordinates needed)
AFFINE = np.eye(4)


def convert(args):
    dataset_name = f"Dataset{args.dataset_id:03d}_{DATASET_PREFIX}"
    dataset_dir = NNUNET_RAW / dataset_name
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Discover volume files (exclude Probabilities files)
    vol_files = sorted([
        f for f in HDF5_DIR.glob("volume_*.h5")
        if "Probabilities" not in f.name
    ])

    print(f"Found {len(vol_files)} volumes in {HDF5_DIR}")
    print(f"Output → {dataset_dir}\n")

    num_converted = 0
    for i, vol_file in enumerate(vol_files):
        # Extract start slice number from filename, e.g. volume_0836-0855 → 0836
        m = re.search(r"volume_(\d+)-(\d+)", vol_file.stem)
        if not m:
            print(f"  Skipping {vol_file.name} (unexpected name)")
            continue
        case_id = f"Mickey3D_{m.group(1)}"

        prob_file = vol_file.parent / f"{vol_file.stem}_Probabilities.h5"
        if not prob_file.exists():
            print(f"  WARNING: No probability file for {vol_file.name}, skipping.")
            continue

        img_out = images_dir / f"{case_id}_0000.nii.gz"
        lbl_out = labels_dir / f"{case_id}.nii.gz"

        if img_out.exists() and lbl_out.exists() and not args.force:
            print(f"  [{i+1}/{len(vol_files)}] {case_id} — already exists, skipping.")
            num_converted += 1
            continue

        # Read image: (Z, H, W) uint16
        with h5py.File(vol_file, "r") as f:
            image = f["volume"][:].astype(np.float32)

        # Read probabilities: (Z, H, W, 3) float32 → argmax → uint8 label
        with h5py.File(prob_file, "r") as f:
            probs = f["exported_data"][:]
        label = np.argmax(probs, axis=-1).astype(np.uint8)

        # NIfTI expects (X, Y, Z) but nnU-Net handles axes via dataset.json
        # We store as (Z, H, W) — nibabel will write axes in order
        img_nii = nib.Nifti1Image(image, AFFINE)
        lbl_nii = nib.Nifti1Image(label, AFFINE)

        nib.save(img_nii, str(img_out))
        nib.save(lbl_nii, str(lbl_out))
        del image, probs, label  # free memory

        num_converted += 1
        print(
            f"  [{i+1}/{len(vol_files)}] {case_id} — "
            f"saved  image {img_out.name}  label {lbl_out.name}"
        )

    # Write dataset.json
    dataset_json = {
        "channel_names": {"0": "XRay"},
        "labels": LABELS,
        "numTraining": num_converted,
        "file_ending": ".nii.gz",
    }
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\nConverted {num_converted} volumes → {dataset_dir}")
    print(f"dataset.json written.")
    print(f"\nNext steps:")
    print(f"  1. Run preprocessing:")
    print(f"     nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity")
    print(f"  2. Train:")
    print(f"     nnUNetv2_train {args.dataset_id} 3d_fullres 0 -tr nnUNetTrainerProgress")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Convert 3D HDF5 volumes to nnU-Net NIfTI")
    p.add_argument("--dataset-id", type=int, default=DATASET_ID)
    p.add_argument("--force", action="store_true", help="Overwrite existing files")
    convert(p.parse_args())
