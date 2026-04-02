"""
Evaluation script for trained INR models.

Produces:
- PSNR and SSIM metrics
- Side-by-side visualization (ground truth vs reconstruction)
- Segmentation accuracy and confusion matrix (if seg head was used)
- Difference map
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim

from .data import (
    load_2d_slice,
    load_3d_volume,
    CoordinateDataset2D,
    CoordinateDataset3D,
)
from .models import build_model


@torch.no_grad()
def reconstruct_image(model, dataset, device, batch_size=2**18):
    """Query model at all dataset coordinates and reshape to image/volume."""
    model.eval()
    coords_all = torch.from_numpy(dataset.coords).to(device)
    n = coords_all.shape[0]

    intensity_parts = []
    seg_parts = []
    has_seg = hasattr(model, "use_seg_head") and model.use_seg_head

    for i in range(0, n, batch_size):
        batch = coords_all[i : i + batch_size]
        out = model(batch)
        intensity_parts.append(out["intensity"].squeeze(-1).cpu())
        if has_seg:
            seg_parts.append(out["seg_logits"].argmax(dim=-1).cpu())

    intensity = torch.cat(intensity_parts).numpy()
    seg_pred = torch.cat(seg_parts).numpy() if has_seg else None

    if isinstance(dataset, CoordinateDataset2D):
        intensity = intensity.reshape(dataset.H, dataset.W)
        if seg_pred is not None:
            seg_pred = seg_pred.reshape(dataset.H, dataset.W)
    else:
        intensity = intensity.reshape(dataset.D, dataset.H, dataset.W)
        if seg_pred is not None:
            seg_pred = seg_pred.reshape(dataset.D, dataset.H, dataset.W)

    return intensity, seg_pred


def compute_metrics(gt, pred, seg_gt=None, seg_pred=None):
    """Compute PSNR, SSIM, and optionally segmentation accuracy."""
    mse = np.mean((gt - pred) ** 2)
    psnr = -10.0 * np.log10(mse) if mse > 0 else float("inf")

    # SSIM — for 3D, compute per-slice and average
    if gt.ndim == 2:
        ssim_val = ssim(gt, pred, data_range=1.0)
    else:
        ssim_vals = []
        for z in range(gt.shape[0]):
            ssim_vals.append(ssim(gt[z], pred[z], data_range=1.0))
        ssim_val = np.mean(ssim_vals)

    metrics = {"psnr": float(psnr), "ssim": float(ssim_val), "mse": float(mse)}

    if seg_gt is not None and seg_pred is not None:
        acc = np.mean(seg_gt == seg_pred)
        # Per-class accuracy
        per_class = {}
        for c in range(3):
            mask = seg_gt == c
            if mask.sum() > 0:
                per_class[f"class_{c}_acc"] = float(np.mean(seg_pred[mask] == c))
        metrics["seg_accuracy"] = float(acc)
        metrics["seg_per_class"] = per_class

    return metrics


def save_visualizations(gt, pred, seg_gt, seg_pred, out_dir, mode):
    """Save comparison figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)

    if mode == "2d":
        slices = [(gt, pred, seg_gt, seg_pred, "")]
    else:
        # Pick 3 representative slices for 3D
        D = gt.shape[0]
        indices = [0, D // 2, D - 1]
        slices = [(gt[z], pred[z],
                   seg_gt[z] if seg_gt is not None else None,
                   seg_pred[z] if seg_pred is not None else None,
                   f"_z{z}") for z in indices]

    for gt_s, pred_s, sgt, spred, suffix in slices:
        # Intensity comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(gt_s, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")

        axes[1].imshow(pred_s, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("INR Reconstruction")
        axes[1].axis("off")

        diff = np.abs(gt_s - pred_s)
        axes[2].imshow(diff, cmap="hot", vmin=0, vmax=0.1)
        axes[2].set_title(f"Abs Difference (max={diff.max():.4f})")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(out_dir / f"comparison{suffix}.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Segmentation comparison
        if sgt is not None and spred is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(sgt, cmap="tab10", vmin=0, vmax=2)
            axes[0].set_title("Ground Truth Segmentation")
            axes[0].axis("off")

            axes[1].imshow(spred, cmap="tab10", vmin=0, vmax=2)
            axes[1].set_title("INR Segmentation")
            axes[1].axis("off")

            plt.tight_layout()
            plt.savefig(out_dir / f"segmentation{suffix}.png", dpi=150,
                        bbox_inches="tight")
            plt.close()


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load config from training
    log_path = Path(args.model_dir) / "train_log.json"
    with open(log_path) as f:
        train_log = json.load(f)
    config = train_log["config"]

    mode = config["mode"]
    use_seg = config["use_seg"]
    arch = config["arch"]

    # Load data
    probs_path = config["probs"] if use_seg else None
    if mode == "2d":
        image, seg = load_2d_slice(config["image"], probs_path)
        dataset = CoordinateDataset2D(image, seg)
        coord_dim = 2
    else:
        volume, seg = load_3d_volume(config["image"], probs_path)
        dataset = CoordinateDataset3D(volume, seg)
        coord_dim = 3
        image = volume  # for metrics

    # Load model
    model = build_model(
        arch=arch,
        coord_dim=coord_dim,
        use_seg_head=use_seg,
        n_fourier=config.get("n_fourier", 256),
        sigma=config.get("sigma", 10.0),
        hidden_dim=config.get("hidden_dim", 256),
        n_layers=config.get("n_layers", 4),
        omega_0=config.get("omega_0", 30.0),
    ).to(device)

    ckpt_path = Path(args.model_dir) / "best_model.pt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f"Loaded model from {ckpt_path}")

    # Reconstruct
    print("Reconstructing full image/volume...")
    pred, seg_pred = reconstruct_image(model, dataset, device)

    # Metrics
    metrics = compute_metrics(image, pred, seg, seg_pred)
    print(f"\nMetrics:")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    if "seg_accuracy" in metrics:
        print(f"  Seg Accuracy: {metrics['seg_accuracy']:.4f}")
        for k, v in metrics["seg_per_class"].items():
            print(f"    {k}: {v:.4f}")

    # Save
    out_dir = Path(args.model_dir) / "eval"
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    save_visualizations(image, pred, seg, seg_pred, out_dir, mode)
    print(f"\nEvaluation outputs saved to {out_dir}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained INR model")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing best_model.pt and train_log.json")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
