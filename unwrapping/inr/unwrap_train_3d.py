"""
Training script for 3D learned unwrapping (Step 1b — 3D).

Trains a deformation network f(x, y, z) -> (u, v) that maps rolled film voxel
coordinates to flat strip coordinates. The strip is generated as (u, z) where
u = learned position along the film and z = known slice coordinate.

Usage:
    python -m unwrapping.inr.unwrap_train_3d \
        --data-dir path/to/01_Mickey_3d \
        --out-dir results/unwrap_3d_ml
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.inr.unwrap_data_3d import UnwrapDataset3D
from unwrapping.inr.unwrap_model import DeformationINR


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
    print("Loading 3D volume data...")
    t_data = time.time()
    dataset = UnwrapDataset3D(
        args.data_dir, device=device, max_slices=args.max_slices
    )
    print(f"  Data loaded in {time.time() - t_data:.1f}s")
    print(f"  Training on {dataset.n:,} film voxels, {dataset.n_layers} layers")

    # Build model (3D input)
    model = DeformationINR(
        n_fourier=args.n_fourier,
        sigma=args.sigma,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        input_dim=3,
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
        "n_film_voxels": dataset.n,
        "n_layers": dataset.n_layers,
        "z_range": [dataset.z_min_global, dataset.z_max_global],
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

        # 2. Smoothness: penalize large output changes for nearby coords
        eps = args.smooth_eps
        noise = torch.randn_like(coords) * eps
        coords_pert = coords + noise
        uv_pert = model(coords_pert)
        out_diff = (uv_pert - uv).norm(dim=-1)
        in_diff = noise.norm(dim=-1)
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

    with open(os.path.join(args.out_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s. Best loss: {best_loss:.6f}")
    print(f"Results saved to {args.out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train 3D unwrapping deformation network")
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing volume_*.h5 and *_Probabilities.h5 files")
    parser.add_argument("--out-dir", required=True, help="Output directory")

    # Architecture
    parser.add_argument("--n-fourier", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=6)

    # Training
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--max-slices", type=int, default=None,
                        help="Limit number of slices for testing (default: all)")

    # Smoothness
    parser.add_argument("--w-smooth", type=float, default=0.001,
                        help="Smoothness loss weight")
    parser.add_argument("--smooth-eps", type=float, default=0.01,
                        help="Perturbation size for smoothness")
    parser.add_argument("--smooth-lip", type=float, default=5.0,
                        help="Lipschitz threshold")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
