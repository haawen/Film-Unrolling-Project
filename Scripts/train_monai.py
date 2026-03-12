"""
MONAI-based training for 3D U-Net and Swin UNETR.

Uses the same data and cross-validation splits as nnU-Net so that
results are directly comparable.

Models:
    unet3d      — standard 3D U-Net (encoder–decoder with skip connections)
    swinunetr   — Swin UNETR (shifted-window vision-transformer encoder)

Usage:
    python Scripts/train_monai.py --model unet3d    --fold 0
    python Scripts/train_monai.py --model swinunetr --fold 0
    python Scripts/train_monai.py --model unet3d    --fold 0 --resume
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

from monai.data import (
    CacheDataset,
    DataLoader,
    decollate_batch,
    list_data_collate,
)
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import UNet, SwinUNETR
from monai.transforms import (
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
from monai.utils import set_determinism

# ─── Paths ───────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", Path(__file__).resolve().parent.parent))
NNUNET_RAW = Path(os.environ.get("nnUNet_raw", PROJECT_DIR / "nnUNet_data" / "nnUNet_raw"))
NNUNET_PREPROCESSED = Path(
    os.environ.get("nnUNet_preprocessed", PROJECT_DIR / "nnUNet_data" / "nnUNet_preprocessed")
)
RESULTS_BASE = Path(os.environ.get("MONAI_RESULTS", PROJECT_DIR / "monai_results"))

DATASET_NAME_DEFAULT = "Dataset502_MickeyScroll3D"
NUM_CLASSES = 3  # background=0, foreground_1=1, foreground_2=2

# ─── Data helpers ────────────────────────────────────────────────────────────


def discover_cases(raw_dir: Path) -> list[dict]:
    """Build [{image, label, case_id}, …] from nnU-Net raw directory."""
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


def load_splits(fold: int, data_list: list[dict], dataset_name: str) -> tuple[list[dict], list[dict]]:
    """Return (train, val) using nnU-Net splits if available, else KFold."""
    splits_file = NNUNET_PREPROCESSED / dataset_name / "splits_final.json"
    if splits_file.exists():
        splits = json.loads(splits_file.read_text())
        if fold < len(splits):
            train_ids = set(splits[fold]["train"])
            val_ids = set(splits[fold]["val"])
            train = [d for d in data_list if d["case_id"] in train_ids]
            val = [d for d in data_list if d["case_id"] in val_ids]
            if train and val:
                return train, val

    # Fallback: deterministic 5-fold split
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    indices = list(range(len(data_list)))
    train_idx, val_idx = list(kf.split(indices))[fold]
    return [data_list[i] for i in train_idx], [data_list[i] for i in val_idx]


# ─── Transforms ──────────────────────────────────────────────────────────────


def train_transforms(patch_size: tuple[int, ...]):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"], nonzero=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=2,
                neg=1,
                num_samples=4,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(1, 2)),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


def val_transforms():
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"], nonzero=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )


# ─── Model factory ──────────────────────────────────────────────────────────


def build_model(name: str, patch_size: tuple[int, ...]) -> torch.nn.Module:
    if name == "unet3d":
        return UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=NUM_CLASSES,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="INSTANCE",
        )
    elif name == "swinunetr":
        return SwinUNETR(
            in_channels=1,
            out_channels=NUM_CLASSES,
            feature_size=48,
            spatial_dims=3,
            use_checkpoint=True,
        )
    else:
        raise ValueError(f"Unknown model: {name}")


# ─── Metrics → nnU-Net–compatible summary.json ──────────────────────────────


def compute_iou(pred: np.ndarray, ref: np.ndarray) -> float:
    intersection = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    return float(intersection / union) if union > 0 else float("nan")


def evaluate_predictions(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    patch_size: tuple[int, ...],
    sw_batch_size: int = 2,
) -> dict:
    """Run sliding-window inference and compute per-case + mean metrics."""
    model.eval()
    post_pred = AsDiscreted(keys="pred", argmax=True, to_onehot=NUM_CLASSES)
    post_label = AsDiscreted(keys="label", to_onehot=NUM_CLASSES)

    dice_metric = DiceMetric(include_background=False, reduction="none")

    per_case = []
    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            preds = sliding_window_inference(
                images,
                roi_size=patch_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=0.5,
            )

            # Decollate for per-sample processing
            batch_out = [{"pred": p, "label": l} for p, l in zip(preds, labels)]
            batch_out = [post_pred(d) for d in batch_out]
            batch_out = [post_label(d) for d in batch_out]

            for sample in batch_out:
                pred_oh = sample["pred"].cpu().numpy()  # (C, ...)
                label_oh = sample["label"].cpu().numpy()

                case_metrics = {}
                for cls in range(1, NUM_CLASSES):
                    p = pred_oh[cls] > 0.5
                    r = label_oh[cls] > 0.5
                    tp = float(np.logical_and(p, r).sum())
                    fp = float(np.logical_and(p, ~r).sum())
                    fn = float(np.logical_and(~p, r).sum())
                    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else float("nan")
                    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float("nan")
                    case_metrics[str(cls)] = {
                        "Dice": dice,
                        "IoU": iou,
                        "FP": fp,
                        "FN": fn,
                        "TP": tp,
                    }
                per_case.append({"metrics": case_metrics})

    # Aggregate means
    mean_metrics = {}
    for cls in range(1, NUM_CLASSES):
        cls_key = str(cls)
        vals = [c["metrics"][cls_key] for c in per_case if cls_key in c["metrics"]]
        mean_metrics[cls_key] = {
            m: float(np.nanmean([v[m] for v in vals])) for m in ["Dice", "IoU", "FP", "FN", "TP"]
        }

    fg_dice = np.nanmean([mean_metrics[str(c)]["Dice"] for c in range(1, NUM_CLASSES)])
    fg_iou = np.nanmean([mean_metrics[str(c)]["IoU"] for c in range(1, NUM_CLASSES)])

    return {
        "mean": mean_metrics,
        "foreground_mean": {"Dice": float(fg_dice), "IoU": float(fg_iou)},
        "metric_per_case": per_case,
    }


# ─── Training loop ──────────────────────────────────────────────────────────


def train(args):
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve patch size
    patch_size = tuple(args.patch_size)
    print(f"Patch size: {patch_size}")

    # ── Data ─────────────────────────────────────────────────────────────
    dataset_name = args.dataset
    raw_dir = NNUNET_RAW / dataset_name
    data_list = discover_cases(raw_dir)
    if not data_list:
        raise FileNotFoundError(f"No cases found in {raw_dir}")
    print(f"Found {len(data_list)} cases in {dataset_name}")

    train_data, val_data = load_splits(args.fold, data_list, dataset_name)
    print(f"Fold {args.fold}: {len(train_data)} train, {len(val_data)} val")

    train_ds = CacheDataset(train_data, transform=train_transforms(patch_size), cache_rate=0.5)
    val_ds = CacheDataset(val_data, transform=val_transforms(), cache_rate=1.0)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=list_data_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = build_model(args.model, patch_size).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}  |  Parameters: {n_params:,}")

    # ── Output directory ─────────────────────────────────────────────────
    model_display = "UNet3D" if args.model == "unet3d" else "SwinUNETR"
    out_dir = RESULTS_BASE / dataset_name / model_display / f"fold_{args.fold}"
    ckpt_dir = out_dir / "checkpoints"
    val_dir = out_dir / "validation"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # ── Optimizer / scheduler / loss ─────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    scaler = GradScaler()

    # ── Resume ───────────────────────────────────────────────────────────
    start_epoch = 0
    best_dice = 0.0
    log = {"train_loss": [], "val_dice": [], "epoch_time": []}

    ckpt_path = ckpt_dir / "latest.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt.get("best_dice", 0.0)
        log = ckpt.get("log", log)
        print(f"Resumed from epoch {start_epoch}, best Dice {best_dice:.4f}")

    # ── Training ─────────────────────────────────────────────────────────
    print(f"\nStarting training for {args.epochs} epochs …")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        n_steps = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_steps += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_steps, 1)
        elapsed = time.time() - t0
        log["train_loss"].append(avg_loss)
        log["epoch_time"].append(elapsed)

        # Validate every N epochs or on last epoch
        val_dice = 0.0
        if (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1:
            summary = evaluate_predictions(model, val_loader, device, patch_size)
            val_dice = summary["foreground_mean"]["Dice"]
            log["val_dice"].append(val_dice)

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(model.state_dict(), ckpt_dir / "best.pt")

            # Save validation summary
            with open(val_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
        else:
            log["val_dice"].append(None)

        # Logging
        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch:>4d}/{args.epochs}  "
            f"loss={avg_loss:.4f}  val_dice={val_dice:.4f}  "
            f"best={best_dice:.4f}  lr={lr:.2e}  time={elapsed:.0f}s"
        )

        # Checkpointing
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_dice": best_dice,
                    "log": log,
                },
                ckpt_path,
            )

    # ── Final evaluation with best weights ───────────────────────────────
    print("\nRunning final evaluation with best weights …")
    best_weights = ckpt_dir / "best.pt"
    if best_weights.exists():
        model.load_state_dict(torch.load(best_weights, map_location=device, weights_only=True))
    final_summary = evaluate_predictions(model, val_loader, device, patch_size)
    final_summary["model"] = model_display
    final_summary["fold"] = args.fold
    final_summary["n_parameters"] = n_params
    final_summary["epochs_trained"] = args.epochs
    final_summary["best_fg_dice"] = best_dice

    with open(val_dir / "summary.json", "w") as f:
        json.dump(final_summary, f, indent=2)

    # Save training log
    with open(out_dir / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nDone.  Best foreground Dice = {best_dice:.4f}")
    print(f"Results saved to {out_dir}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MONAI training: 3D U-Net / Swin UNETR")
    p.add_argument("--model", required=True, choices=["unet3d", "swinunetr"])
    p.add_argument("--dataset", default=DATASET_NAME_DEFAULT,
                   help=f"Dataset directory name (default: {DATASET_NAME_DEFAULT})")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patch_size", type=int, nargs=3, default=[16, 192, 192],
                   help="Z Y X patch size (default: 16 192 192)")
    p.add_argument("--val_interval", type=int, default=10,
                   help="Validate every N epochs")
    p.add_argument("--save_every", type=int, default=25)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    train(p.parse_args())
