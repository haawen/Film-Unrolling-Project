"""
Generate predictions from trained MONAI models (UNet3D / SwinUNETR).

Loads the best checkpoint, runs sliding-window inference on validation cases,
and saves predictions as .nii.gz files alongside summary.json.

Usage:
    python Scripts/predict_monai.py --model unet3d    --fold 0
    python Scripts/predict_monai.py --model swinunetr --fold 0
"""

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet, SwinUNETR
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    SpatialPadd,
)

# ─── Paths ───────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parent.parent))
NNUNET_RAW = Path(os.environ.get("nnUNet_raw", PROJECT_DIR / "nnUNet_data" / "nnUNet_raw"))
NNUNET_PREPROCESSED = Path(
    os.environ.get("nnUNet_preprocessed", PROJECT_DIR / "nnUNet_data" / "nnUNet_preprocessed")
)
RESULTS_BASE = Path(os.environ.get("MONAI_RESULTS", PROJECT_DIR / "monai_results"))

DATASET_NAME_DEFAULT = "Dataset502_MickeyScroll3D"
NUM_CLASSES = 3

# Default training patch sizes per model (must match training config)
MODEL_PATCH_SIZES = {
    "unet3d": [16, 256, 256],
    "swinunetr": [32, 256, 256],
}


# ─── Reuse helpers from train_monai ─────────────────────────────────────────

def discover_cases(raw_dir: Path) -> list[dict]:
    images_dir = raw_dir / "imagesTr"
    labels_dir = raw_dir / "labelsTr"
    label_files = sorted(labels_dir.glob("*.nii.gz"))
    data = []
    for lbl in label_files:
        case_id = lbl.name.replace(".nii.gz", "")
        img = images_dir / f"{case_id}_0000.nii.gz"
        if img.exists():
            data.append({"image": str(img), "label": str(lbl), "case_id": case_id})
    return data


def load_splits(fold: int, data_list: list[dict], dataset_name: str):
    splits_file = NNUNET_PREPROCESSED / dataset_name / "splits_final.json"
    if splits_file.exists():
        splits = json.loads(splits_file.read_text())
        if fold < len(splits):
            val_ids = set(splits[fold]["val"])
            val = [d for d in data_list if d["case_id"] in val_ids]
            if val:
                return val
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    indices = list(range(len(data_list)))
    _, val_idx = list(kf.split(indices))[fold]
    return [data_list[i] for i in val_idx]


def build_model(name: str, patch_size: tuple[int, ...]) -> torch.nn.Module:
    if name == "unet3d":
        return UNet(
            spatial_dims=3, in_channels=1, out_channels=NUM_CLASSES,
            channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
            num_res_units=2, norm="INSTANCE",
        )
    elif name == "swinunetr":
        return SwinUNETR(
            in_channels=1, out_channels=NUM_CLASSES,
            feature_size=48, spatial_dims=3, use_checkpoint=False,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


# ─── Inference transforms ───────────────────────────────────────────────────

def inference_transforms(patch_size: tuple[int, ...]):
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size, method="end"),
        EnsureTyped(keys=["image", "label"]),
    ])


# ─── Main ───────────────────────────────────────────────────────────────────

def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_size = tuple(args.patch_size)

    # Discover validation cases
    dataset_name = args.dataset
    raw_dir = NNUNET_RAW / dataset_name
    data_list = discover_cases(raw_dir)
    val_data = load_splits(args.fold, data_list, dataset_name)
    print(f"Fold {args.fold}: {len(val_data)} validation cases")

    # Load model
    model_display = "UNet3D" if args.model == "unet3d" else "SwinUNETR"
    out_dir = RESULTS_BASE / dataset_name / model_display / f"fold_{args.fold}"
    ckpt_path = out_dir / "checkpoints" / "best.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best checkpoint at {ckpt_path}")

    model = build_model(args.model, patch_size).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded {model_display} from {ckpt_path}")

    # Prepare transforms
    xform = inference_transforms(patch_size)

    # Output directory for predictions
    pred_dir = out_dir / "validation"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving predictions to {pred_dir}")

    with torch.no_grad():
        for i, case in enumerate(val_data):
            case_id = case["case_id"]
            sample = xform(case)
            image = sample["image"].unsqueeze(0).to(device)

            # Sliding window inference
            output = sliding_window_inference(
                image, roi_size=patch_size, sw_batch_size=2,
                predictor=model, overlap=0.5,
            )
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

            # Load original image to get affine for NIfTI header
            ref_nii = nib.load(case["image"])
            # pred may have been padded — crop back to original shape
            orig_shape = ref_nii.shape
            pred_cropped = pred[:orig_shape[0], :orig_shape[1], :orig_shape[2]]

            pred_nii = nib.Nifti1Image(pred_cropped, affine=ref_nii.affine)
            save_path = pred_dir / f"{case_id}.nii.gz"
            nib.save(pred_nii, str(save_path))

            print(f"  [{i+1}/{len(val_data)}] {case_id} → {save_path.name}  "
                  f"shape={pred_cropped.shape}  classes={np.unique(pred_cropped)}")

    print(f"\nDone. {len(val_data)} predictions saved to {pred_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MONAI prediction: 3D U-Net / Swin UNETR")
    p.add_argument("--model", required=True, choices=["unet3d", "swinunetr"])
    p.add_argument("--dataset", default=DATASET_NAME_DEFAULT)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--patch_size", type=int, nargs=3, default=None,
                   help="Override inference patch size (default: use training patch size)")
    args = p.parse_args()
    if args.patch_size is None:
        args.patch_size = MODEL_PATCH_SIZES.get(args.model, [32, 192, 192])
    predict(args)
