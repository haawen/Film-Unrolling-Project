"""
Training script for INR on CT reconstruction data.

Supports 2D slices and 3D volumes, with and without segmentation head.

Usage:
    # 2D, intensity only
    python -m unwrapping.inr.train --mode 2d --image path/to/slice.h5 --arch fourier

    # 2D, with segmentation
    python -m unwrapping.inr.train --mode 2d --image path/to/slice.h5 \
        --probs path/to/probs.h5 --use-seg --arch fourier

    # 3D, intensity only
    python -m unwrapping.inr.train --mode 3d --image path/to/volume.h5 --arch siren

    # 3D, with segmentation
    python -m unwrapping.inr.train --mode 3d --image path/to/volume.h5 \
        --probs path/to/probs.h5 --use-seg --arch siren
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import (
    load_2d_slice,
    load_3d_volume,
    CoordinateDataset2D,
    CoordinateDataset3D,
    RandomCoordinateSampler,
)
from .models import build_model


def compute_psnr(mse: float) -> float:
    """Compute PSNR from MSE (data range [0, 1])."""
    if mse <= 0:
        return float("inf")
    return -10.0 * torch.log10(torch.tensor(mse)).item()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load data ----
    print(f"Loading {args.mode} data...")
    probs_path = args.probs if args.use_seg else None

    if args.mode == "2d":
        image, seg = load_2d_slice(args.image, probs_path)
        dataset = CoordinateDataset2D(image, seg)
        coord_dim = 2
        print(f"  Image shape: {image.shape}, {dataset.n_pixels:,} pixels")
    else:
        volume, seg = load_3d_volume(args.image, probs_path)
        dataset = CoordinateDataset3D(volume, seg)
        coord_dim = 3
        print(f"  Volume shape: {volume.shape}, {dataset.n_voxels:,} voxels")

    sampler = RandomCoordinateSampler(dataset, batch_size=args.batch_size, device=str(device))

    # ---- Build model ----
    model = build_model(
        arch=args.arch,
        coord_dim=coord_dim,
        use_seg_head=args.use_seg,
        n_fourier=args.n_fourier,
        sigma=args.sigma,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        omega_0=args.omega_0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {args.arch}, {n_params:,} parameters, seg_head={args.use_seg}")

    # ---- Optimizer & scheduler ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    # ---- Training loop ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = {"config": vars(args), "steps": [], "psnr": [], "loss": [],
           "seg_acc": [] if args.use_seg else None}

    seg_weight = args.seg_weight
    best_psnr = 0.0
    t0 = time.time()

    model.train()
    for step in range(1, args.steps + 1):
        batch = sampler.sample()
        if args.use_seg:
            coords, targets, seg_labels = batch
        else:
            coords, targets = batch

        out = model(coords)
        intensity_pred = out["intensity"].squeeze(-1)

        # Intensity loss (MSE)
        loss_mse = F.mse_loss(intensity_pred, targets)
        loss = loss_mse

        # Segmentation loss (cross-entropy)
        if args.use_seg:
            seg_logits = out["seg_logits"]
            loss_seg = F.cross_entropy(seg_logits, seg_labels)
            loss = loss + seg_weight * loss_seg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # ---- Logging ----
        if step % args.log_every == 0 or step == 1:
            psnr = compute_psnr(loss_mse.item())
            msg = f"[{step:>6d}/{args.steps}] loss={loss.item():.6f} mse={loss_mse.item():.6f} psnr={psnr:.2f}dB"

            if args.use_seg:
                seg_acc = (seg_logits.argmax(dim=-1) == seg_labels).float().mean().item()
                msg += f" seg_acc={seg_acc:.4f}"
                log["seg_acc"].append(seg_acc)

            msg += f" lr={scheduler.get_last_lr()[0]:.2e}"
            print(msg)

            log["steps"].append(step)
            log["psnr"].append(psnr)
            log["loss"].append(loss.item())

            if psnr > best_psnr:
                best_psnr = psnr
                torch.save(model.state_dict(), out_dir / "best_model.pt")

        # ---- Checkpoint ----
        if step % args.save_every == 0:
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "psnr": compute_psnr(loss_mse.item()),
            }, out_dir / "checkpoint.pt")

    elapsed = time.time() - t0
    final_psnr = compute_psnr(loss_mse.item())
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Final PSNR: {final_psnr:.2f} dB, Best PSNR: {best_psnr:.2f} dB")

    # ---- Save final model + log ----
    torch.save(model.state_dict(), out_dir / "final_model.pt")
    log["elapsed_seconds"] = elapsed
    log["final_psnr"] = final_psnr
    log["best_psnr"] = best_psnr
    log["n_params"] = n_params
    with open(out_dir / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"Outputs saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train INR on CT data")

    # Data
    parser.add_argument("--mode", choices=["2d", "3d"], required=True)
    parser.add_argument("--image", type=str, required=True, help="Path to HDF5 file")
    parser.add_argument("--probs", type=str, default=None, help="Path to probability map HDF5")
    parser.add_argument("--use-seg", action="store_true", help="Enable segmentation head")

    # Architecture
    parser.add_argument("--arch", choices=["fourier", "siren"], default="fourier")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    # Fourier-specific
    parser.add_argument("--n-fourier", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=10.0)
    # SIREN-specific
    parser.add_argument("--omega-0", type=float, default=30.0)

    # Training
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=2**18,
                        help="Number of coordinate samples per step")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seg-weight", type=float, default=0.1,
                        help="Weight for segmentation loss")

    # Output
    parser.add_argument("--out-dir", type=str, default="unwrapping/inr/results")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
