"""
Training script for learned unwrapping (Step 1b).

Trains a deformation network f(x,y) → (u,v) that maps rolled film pixel
coordinates to flat strip coordinates.

Targets are standardized (zero mean, unit variance). The network output
is unbounded (no sigmoid). Normalization stats are saved for eval-time
denormalization.

Losses:
  1. Supervision: MSE to initial polar parameterization (standardized)
  2. Smoothness: penalizes large output differences for nearby coordinates

Usage:
    python -m unwrapping.inr.unwrap_train \
        --image path/to/image.h5 \
        --probs path/to/probs.h5 \
        --out-dir results/unwrap_2d
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.inr.unwrap_data import UnwrapDataset
from unwrapping.inr.unwrap_model import DeformationINR
from unwrapping.inr.data import load_2d_slice
from unwrapping.center_detection import find_spool_center


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    image, seg = load_2d_slice(args.image, args.probs)
    H, W = image.shape
    print(f"  Image: {H}x{W}")

    # Find spiral center
    print("Finding spiral center...")
    center = find_spool_center(seg)
    print(f"  Center: ({center[0]:.1f}, {center[1]:.1f})")

    # Prepare unwrap dataset
    print("Preparing unwrap dataset...")
    dataset = UnwrapDataset(image, seg, center, device=device)
    print(f"  Training on {dataset.n:,} film pixels, {dataset.n_layers} layers")

    # Build model
    model = DeformationINR(
        n_fourier=args.n_fourier,
        sigma=args.sigma,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01
    )

    # Training log
    log = {
        "config": vars(args),
        "n_film_pixels": dataset.n,
        "n_layers": dataset.n_layers,
        "norm_stats": {
            "u_mean": dataset.u_mean, "u_std": dataset.u_std,
            "v_mean": dataset.v_mean, "v_std": dataset.v_std,
            "u_min": dataset.u_min, "u_max": dataset.u_max,
            "v_min": dataset.v_min, "v_max": dataset.v_max,
        },
        "steps": [],
        "loss_total": [],
        "loss_supervision": [],
        "loss_smooth": [],
    }

    print(f"\nTraining for {args.steps} steps...")
    t0 = time.time()
    best_loss = float("inf")

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()

        # Sample batch
        batch = dataset.sample(args.batch_size)
        coords = batch["coords"]
        u_tgt = batch["u_target"]
        v_tgt = batch["v_target"]

        # Forward pass
        uv = model(coords)

        # === Losses ===

        # 1. Supervision: MSE to standardized targets
        target = torch.stack([u_tgt, v_tgt], dim=-1)
        l_sup = ((uv - target) ** 2).mean()

        # 2. Smoothness: compare output at nearby coordinates
        # Perturb each coordinate slightly, penalize large output changes
        eps = args.smooth_eps
        noise = torch.randn_like(coords) * eps
        coords_pert = coords + noise
        uv_pert = model(coords_pert)
        # Output difference should be proportional to input perturbation
        # (Lipschitz constraint: ||f(x+δ) - f(x)|| ≤ L||δ||)
        out_diff = (uv_pert - uv).norm(dim=-1)
        in_diff = noise.norm(dim=-1)
        # Penalize output change that's much larger than input change
        l_smooth = torch.relu(out_diff / (in_diff + 1e-8) - args.smooth_lip).mean()

        loss = l_sup + args.w_smooth * l_smooth

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            log["steps"].append(step)
            log["loss_total"].append(float(loss))
            log["loss_supervision"].append(float(l_sup))
            log["loss_smooth"].append(float(l_smooth))

            if float(loss) < best_loss:
                best_loss = float(loss)
                torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))

            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Step {step:5d}/{args.steps} | "
                  f"loss={loss:.6f} (sup={l_sup:.4f} smooth={l_smooth:.4f}) | "
                  f"lr={lr:.6f} | {elapsed:.0f}s")

        # Save checkpoint
        if step % args.save_every == 0 or step == args.steps:
            torch.save(model.state_dict(), os.path.join(args.out_dir, "last.pt"))

    elapsed = time.time() - t0
    log["elapsed_seconds"] = elapsed
    log["best_loss"] = best_loss

    # Save log
    with open(os.path.join(args.out_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s. Best loss: {best_loss:.6f}")
    print(f"Results saved to {args.out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train unwrapping deformation network")
    parser.add_argument("--image", required=True, help="Path to image HDF5")
    parser.add_argument("--probs", required=True, help="Path to segmentation probs HDF5")
    parser.add_argument("--out-dir", required=True, help="Output directory")

    # Architecture
    parser.add_argument("--n-fourier", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)

    # Training
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)

    # Smoothness
    parser.add_argument("--w-smooth", type=float, default=0.01,
                        help="Smoothness loss weight")
    parser.add_argument("--smooth-eps", type=float, default=0.01,
                        help="Perturbation size for smoothness (in normalized coords)")
    parser.add_argument("--smooth-lip", type=float, default=5.0,
                        help="Lipschitz threshold — penalize gradient ratios above this")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
