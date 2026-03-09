"""
nnU-Net v2 Training Pipeline for Mickey HDF5 Scroll Data
=========================================================

This script handles the full pipeline:
  1. Environment setup (installs nnU-Net v2, sets env vars)
  2. Data conversion (HDF5 -> nnU-Net-compatible .tif format)
  3. Experiment planning & preprocessing
  4. Model training (2D U-Net, since data is 2D)

Data:
  - Raw images:  single-channel uint16 (3063x3062) in .h5 'image' dataset
  - Labels:      3-channel float32 probability maps in .h5 'exported_data' dataset
                 -> Converted to integer segmentation maps via argmax
                 -> Label 0 = background, Label 1 = class 1, Label 2 = class 2

Usage:
  python Scripts/train_nnunet.py --step all        # Run everything end-to-end
  python Scripts/train_nnunet.py --step install     # Install nnU-Net v2 only
  python Scripts/train_nnunet.py --step convert     # Convert data only
  python Scripts/train_nnunet.py --step preprocess  # Plan + preprocess only
  python Scripts/train_nnunet.py --step train       # Train only
  python Scripts/train_nnunet.py --step train --fold 0  # Train a specific fold
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Base directory of the project
BASE_DIR = Path(r"D:\M_thesis")

# Source HDF5 data
HDF5_SUBSET_DIR = BASE_DIR / "scannedrolls" / "01_Mickey_hdf_subset"

# nnU-Net directory structure
NNUNET_BASE = BASE_DIR / "nnUNet_data"
NNUNET_RAW = NNUNET_BASE / "nnUNet_raw"
NNUNET_PREPROCESSED = NNUNET_BASE / "nnUNet_preprocessed"
NNUNET_RESULTS = NNUNET_BASE / "nnUNet_results"

# Dataset configuration
DATASET_ID = 501
DATASET_NAME = f"Dataset{DATASET_ID:03d}_MickeyScroll"

# Labels in the probability maps (3 channels -> 3 classes)
LABELS = {
    "background": 0,
    "foreground_1": 1,
    "foreground_2": 2,
}

# Training configuration
UNET_CONFIG = "2d"  # 2D U-Net since our data is 2D
NUM_FOLDS = 5        # Standard 5-fold cross-validation
CUSTOM_TRAINER = "nnUNetTrainerProgress"  # Custom trainer with tqdm + checkpointing
CUSTOM_TRAINER_SRC = BASE_DIR / "Scripts" / "custom_trainer.py"


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Install nnU-Net v2 with CUDA-enabled PyTorch
# ──────────────────────────────────────────────────────────────────────────────

def step_install():
    """Install PyTorch with CUDA support and nnU-Net v2."""
    print("=" * 70)
    print("STEP 1: Installing nnU-Net v2 with CUDA-enabled PyTorch")
    print("=" * 70)

    # Check if CUDA-enabled PyTorch is already available
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  PyTorch {torch.__version__} with CUDA already installed.")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  PyTorch {torch.__version__} found but WITHOUT CUDA support.")
            print("  Reinstalling PyTorch with CUDA...")
            # Install PyTorch with CUDA 12.4 (compatible with NVIDIA driver 591.x)
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu124",
                "--force-reinstall"
            ])
            print("  PyTorch with CUDA 12.4 installed successfully.")
    except ImportError:
        print("  PyTorch not found. Installing with CUDA 12.4...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu124"
        ])

    # Install nnU-Net v2
    try:
        import nnunetv2
        from importlib.metadata import version
        print(f"  nnU-Net v2 already installed (version {version('nnunetv2')}).")
    except (ImportError, Exception):
        print("  Installing nnU-Net v2...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "nnunetv2"
        ])
        print("  nnU-Net v2 installed successfully.")

    # Install additional dependencies
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "h5py", "scikit-image", "tifffile"
    ])

    # Verify installation
    print("\n  Verifying installation...")
    subprocess.check_call([
        sys.executable, "-c",
        "import torch; import nnunetv2; "
        "from importlib.metadata import version; "
        "print(f'  PyTorch: {torch.__version__}'); "
        "print(f'  CUDA available: {torch.cuda.is_available()}'); "
        "print(f'  nnU-Net v2: {version(\"nnunetv2\")}')"
    ])
    print("  Installation complete!\n")


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Convert HDF5 data to nnU-Net format
# ──────────────────────────────────────────────────────────────────────────────

def step_convert():
    """Convert HDF5 image/label pairs to nnU-Net dataset format."""
    import h5py
    import numpy as np
    from skimage.io import imsave

    print("=" * 70)
    print("STEP 2: Converting HDF5 data to nnU-Net format")
    print("=" * 70)

    # Set up nnU-Net environment variables (needed for paths module)
    _set_env_vars()

    # Create directories (automatically created if they don't exist)
    dataset_dir = NNUNET_RAW / DATASET_NAME
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    
    print(f"  Created directories (if needed):")
    print(f"    {dataset_dir}")
    print(f"    {images_tr}")
    print(f"    {labels_tr}\n")

    # Find all raw HDF5 files (those WITHOUT "-image_Probabilities" in the name)
    raw_files = sorted([
        f for f in HDF5_SUBSET_DIR.glob("*.h5")
        if "Probabilities" not in f.name
    ])

    print(f"  Found {len(raw_files)} image-label pairs in {HDF5_SUBSET_DIR}")
    print(f"  Output: {dataset_dir}\n")

    num_converted = 0
    for i, raw_file in enumerate(raw_files):
        # Derive case identifier from filename
        # e.g., "01_Mickey_Stitch_Stitch_Export_0835.h5" -> "Mickey_0835"
        stem = raw_file.stem  # "01_Mickey_Stitch_Stitch_Export_0835"
        case_num = stem.split("_")[-1]  # "0835"
        case_id = f"Mickey_{case_num}"

        # Corresponding probability file
        prob_file = raw_file.parent / f"{stem}-image_Probabilities.h5"
        if not prob_file.exists():
            print(f"  WARNING: No probability file for {raw_file.name}, skipping.")
            continue

        # Output paths (nnU-Net naming convention)
        # Image: {CASE_ID}_{CHANNEL:04d}.tif  (single channel -> 0000)
        # Label: {CASE_ID}.tif
        img_out = images_tr / f"{case_id}_0000.tif"
        lbl_out = labels_tr / f"{case_id}.tif"

        if img_out.exists() and lbl_out.exists():
            print(f"  [{i+1}/{len(raw_files)}] {case_id} — already converted, skipping.")
            num_converted += 1
            continue

        # Read raw image
        with h5py.File(raw_file, "r") as f:
            image = f["image"][:]  # shape: (H, W), dtype: uint16

        # Read probability map and convert to segmentation label
        with h5py.File(prob_file, "r") as f:
            probs = f["exported_data"][:]  # shape: (H, W, 3), dtype: float32

        # Argmax to get integer segmentation map
        label = np.argmax(probs, axis=2).astype(np.uint8)

        # Save as TIFF (lossless, supports uint16 for images and uint8 for labels)
        imsave(str(img_out), image, check_contrast=False)
        imsave(str(lbl_out), label, check_contrast=False)

        num_converted += 1
        print(f"  [{i+1}/{len(raw_files)}] {case_id} — "
              f"image {image.shape} ({image.dtype}), "
              f"label {label.shape} (classes: {np.unique(label).tolist()})")

    # Create dataset.json
    dataset_json = {
        "channel_names": {
            "0": "XRay"   # single-channel grayscale micro-CT / X-ray scan
        },
        "labels": LABELS,
        "numTraining": num_converted,
        "file_ending": ".tif",
        "overwrite_image_reader_writer": "NaturalImage2DIO"
    }

    dataset_json_path = dataset_dir / "dataset.json"
    with open(dataset_json_path, "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\n  Converted {num_converted} cases.")
    print(f"  dataset.json written to {dataset_json_path}")
    print(f"  Dataset structure:")
    print(f"    {dataset_dir}/")
    print(f"      imagesTr/  ({len(list(images_tr.iterdir()))} files)")
    print(f"      labelsTr/  ({len(list(labels_tr.iterdir()))} files)")
    print(f"      dataset.json")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Experiment planning and preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def step_preprocess():
    """Run nnU-Net experiment planning and preprocessing."""
    print("=" * 70)
    print("STEP 3: Experiment planning and preprocessing")
    print("=" * 70)

    _set_env_vars()

    print(f"  Dataset ID: {DATASET_ID}")
    print(f"  Running nnUNetv2_plan_and_preprocess...\n")

    subprocess.check_call([
        sys.executable, "-c",
        "from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry; plan_and_preprocess_entry()",
        "-d", str(DATASET_ID),
        "--verify_dataset_integrity",
        "-c", UNET_CONFIG,
        "--clean",
    ], env=_get_env())

    print("\n  Preprocessing complete!")
    print(f"  Preprocessed data: {NNUNET_PREPROCESSED / DATASET_NAME}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Model training
# ──────────────────────────────────────────────────────────────────────────────

def _install_custom_trainer():
    """Copy custom trainer into nnU-Net package so it gets auto-discovered."""
    import nnunetv2
    variants_dir = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants"
    dest = variants_dir / "custom_trainer.py"
    if not dest.exists() or dest.read_text() != CUSTOM_TRAINER_SRC.read_text():
        shutil.copy2(CUSTOM_TRAINER_SRC, dest)
        print(f"  Installed custom trainer -> {dest}")
    else:
        print(f"  Custom trainer already installed.")


def step_train(fold: int = None, continue_training: bool = False):
    """Train the 2D U-Net model.

    Args:
        fold: Specific fold to train (0-4), or None to train all folds.
        continue_training: If True, continue from last checkpoint.
    """
    print("=" * 70)
    print("STEP 4: Training 2D U-Net")
    print("=" * 70)

    _set_env_vars()
    _install_custom_trainer()

    folds = [fold] if fold is not None else list(range(NUM_FOLDS))

    for f in folds:
        print(f"\n  Training fold {f}/{NUM_FOLDS - 1}...")
        print(f"  Config: {UNET_CONFIG}")
        print(f"  Trainer: {CUSTOM_TRAINER}")
        print(f"  Dataset: {DATASET_NAME} (ID: {DATASET_ID})")

        cmd = [
            sys.executable, "-c",
            "from nnunetv2.run.run_training import run_training_entry; run_training_entry()",
            str(DATASET_ID),
            UNET_CONFIG,
            str(f),
            "-tr", CUSTOM_TRAINER,
            "--npz",  # save softmax outputs for later ensemble selection
        ]

        if continue_training:
            cmd.append("--c")

        print(f"  Command: {' '.join(cmd)}\n")

        subprocess.check_call(cmd, env=_get_env())

        print(f"\n  Fold {f} training complete!")

    print(f"\n  Results saved to: {NNUNET_RESULTS / DATASET_NAME}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────

def _set_env_vars():
    """Set nnU-Net environment variables and ensure directories exist."""
    # Create base directories if they don't exist
    NNUNET_BASE.mkdir(parents=True, exist_ok=True)
    NNUNET_RAW.mkdir(parents=True, exist_ok=True)
    NNUNET_PREPROCESSED.mkdir(parents=True, exist_ok=True)
    NNUNET_RESULTS.mkdir(parents=True, exist_ok=True)
    
    # Set environment variables
    os.environ["nnUNet_raw"] = str(NNUNET_RAW)
    os.environ["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    os.environ["nnUNet_results"] = str(NNUNET_RESULTS)
    # Recommended data augmentation workers for RTX 3060 Ti
    os.environ["nnUNet_n_proc_DA"] = "12"


def _get_env():
    """Get environment dict with nnU-Net paths set."""
    env = os.environ.copy()
    env["nnUNet_raw"] = str(NNUNET_RAW)
    env["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    env["nnUNet_results"] = str(NNUNET_RESULTS)
    env["nnUNet_n_proc_DA"] = "12"
    return env


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="nnU-Net v2 training pipeline for Mickey scroll data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  install     Install PyTorch (CUDA) + nnU-Net v2
  convert     Convert HDF5 data to nnU-Net format
  preprocess  Run experiment planning & preprocessing
  train       Train the 2D U-Net model
  all         Run all steps sequentially

Examples:
  python Scripts/train_nnunet.py --step all
  python Scripts/train_nnunet.py --step train --fold 0
  python Scripts/train_nnunet.py --step train --fold 0 --continue
        """,
    )
    parser.add_argument(
        "--step",
        choices=["install", "convert", "preprocess", "train", "all"],
        default="all",
        help="Which step(s) to run (default: all)",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        choices=[0, 1, 2, 3, 4],
        help="Specific fold to train (default: all 5 folds)",
    )
    parser.add_argument(
        "--continue",
        dest="continue_training",
        action="store_true",
        help="Continue training from last checkpoint",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  nnU-Net v2 Training Pipeline — Mickey Scroll Dataset")
    print("=" * 70)
    print(f"  Base directory:    {BASE_DIR}")
    print(f"  HDF5 source:      {HDF5_SUBSET_DIR}")
    print(f"  nnUNet_raw:       {NNUNET_RAW}")
    print(f"  nnUNet_preproc:   {NNUNET_PREPROCESSED}")
    print(f"  nnUNet_results:   {NNUNET_RESULTS}")
    print(f"  Dataset:          {DATASET_NAME}")
    print(f"  Config:           {UNET_CONFIG}")
    print(f"  Step:             {args.step}")
    if args.fold is not None:
        print(f"  Fold:             {args.fold}")
    print("=" * 70 + "\n")

    if args.step in ("install", "all"):
        step_install()

    if args.step in ("convert", "all"):
        step_convert()

    if args.step in ("preprocess", "all"):
        step_preprocess()

    if args.step in ("train", "all"):
        step_train(fold=args.fold, continue_training=args.continue_training)

    print("=" * 70)
    print("  Pipeline complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
