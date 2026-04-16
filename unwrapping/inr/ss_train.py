"""
Self-supervised training for 3D unwrapping.

Learns f(x,y,z) → u without any pre-computed UV labels.
The mapping is discovered purely from geometric constraints:

  1. Eikonal:   |∇u|² should be constant on the film (uniform parameterization)
  2. Boundary:  u(inner) ≈ 0, u(outer) ≈ n_layers (anchoring)
  3. Ordering:  u increases with radius (inside-out unwinding)
  4. Coverage:  u spans [0, n_layers] (anti-collapse)

Usage:
    python -m unwrapping.inr.ss_train \
        --data-dir path/to/01_Mickey_3d \
        --out-dir results/ss_unwrap
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.inr.ss_data import SelfSupDataset3D
from unwrapping.inr.unwrap_model import DeformationINR


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data (no UV labels — just coords, radius, boundaries)
    print("Loading 3D volume data (self-supervised, no UV labels)...")
    t_data = time.time()
    dataset = SelfSupDataset3D(
        args.data_dir, device=device, max_slices=args.max_slices
    )
    data_time = time.time() - t_data
    print(f"  Data loaded in {data_time:.1f}s")
    print(f"  Training on {dataset.n:,} film voxels")
    print(f"  Estimated layers: {dataset.n_layers_est}")

    n_layers = dataset.n_layers_est

    # Build model (3D input, 1D output: just u)
    model = DeformationINR(
        n_fourier=args.n_fourier,
        sigma=args.sigma,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        input_dim=3,
        output_dim=1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01
    )

    log = {
        "config": vars(args),
        "n_film_voxels": dataset.n,
        "n_layers_est": n_layers,
        "z_range": [dataset.z_min_global, dataset.z_max_global],
        "data_load_seconds": data_time,
        "steps": [],
        "loss_total": [],
        "loss_eikonal": [],
        "loss_boundary": [],
        "loss_order": [],
        "loss_coverage": [],
        "loss_angular": [],
        "loss_z_smooth": [],
        "u_min": [],
        "u_max": [],
        "grad_mag_mean": [],
        "grad_mag_std": [],
        "frac_correct_order": [],
    }

    print(f"\nTraining for {args.steps} steps...")
    print(f"  Warm-up: {args.warmup_steps} steps (boundary + ordering only)")
    print(f"  Eikonal ramp: {args.ramp_steps} steps after warmup")
    t0 = time.time()
    best_loss = float("inf")

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()

        # --- Main batch with boundary oversampling ---
        batch = dataset.sample(args.batch_size)
        coords = batch["coords"]
        coords.requires_grad_(True)

        u = model(coords).squeeze(-1)  # (B,)

        # === Loss 1: Eikonal — constant gradient magnitude on film ===
        # Skip during warm-up, then ramp in linearly over ramp_steps
        if step > args.warmup_steps:
            grads = torch.autograd.grad(
                u.sum(), coords, create_graph=True
            )[0]  # (B, 3)
            grad_xy = grads[:, :2]  # only spatial gradients, not z
            grad_mag_sq = (grad_xy ** 2).sum(dim=-1)  # (B,)
            l_eikonal = grad_mag_sq.var()

            # Ramp from 0→1 over ramp_steps after warmup
            ramp_progress = min(1.0, (step - args.warmup_steps) / args.ramp_steps)
            eikonal_scale = ramp_progress

            # === Loss 6: Z-smoothness ===
            # u should be consistent across z-slices
            if args.w_z_smooth > 0:
                l_z_smooth = (grads[:, 2] ** 2).mean()
            else:
                l_z_smooth = torch.tensor(0.0, device=device)
        else:
            l_eikonal = torch.tensor(0.0, device=device)
            l_z_smooth = torch.tensor(0.0, device=device)
            grad_mag_sq = None
            eikonal_scale = 0.0

        # === Loss 5: Angular diversity (v3) ===
        # Sample pairs at similar radius (same winding). Their u should differ.
        # For u=f(r) [the v1/v2 failure]: same-radius ⟹ same u → loss ≈ 1.
        # For correct spiral: same-radius pairs differ → loss → 0.
        # Contrastive exp: always provides gradient (no dead zone like hinge).
        if args.w_angular > 0:
            coords_c, coords_d = dataset.sample_angular_pairs(args.n_pairs)
            if coords_c.shape[0] > 0:
                u_c = model(coords_c).squeeze(-1)
                u_d = model(coords_d).squeeze(-1)
                # exp(-alpha * |u_diff|): 1 when identical, →0 when different
                l_angular = torch.exp(
                    -args.angular_alpha * (u_c - u_d).abs()
                ).mean()
            else:
                l_angular = torch.tensor(0.0, device=device)
        else:
            l_angular = torch.tensor(0.0, device=device)

        # === Loss 2: Boundary anchoring ===
        inner_mask = batch["is_inner"].bool()
        outer_mask = batch["is_outer"].bool()

        l_inner = (u[inner_mask] ** 2).mean() if inner_mask.any() else torch.tensor(0.0, device=device)
        l_outer = ((u[outer_mask] - n_layers) ** 2).mean() if outer_mask.any() else torch.tensor(0.0, device=device)
        l_boundary = l_inner + l_outer

        # === Loss 3: Radial ordering ===
        coords_a, coords_b = dataset.sample_pairs(args.n_pairs)
        u_a = model(coords_a).squeeze(-1)
        u_b = model(coords_b).squeeze(-1)
        # u_a should be < u_b (inner < outer)
        l_order = F.relu(u_a - u_b + args.order_margin).mean()

        # Fraction of correctly ordered pairs (for monitoring)
        with torch.no_grad():
            frac_correct = (u_a < u_b).float().mean().item()

        # === Loss 4: Coverage / anti-collapse ===
        u_range = u.max() - u.min()
        l_coverage = F.relu(n_layers - u_range)

        # === Total loss ===
        loss = (args.w_eikonal * eikonal_scale * l_eikonal
                + args.w_boundary * l_boundary
                + args.w_order * l_order
                + args.w_coverage * l_coverage
                + args.w_angular * l_angular
                + args.w_z_smooth * eikonal_scale * l_z_smooth)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # --- Logging ---
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                u_min_val = u.min().item()
                u_max_val = u.max().item()
                if grad_mag_sq is not None:
                    gm = grad_mag_sq.sqrt()
                    grad_mean = gm.mean().item()
                    grad_std = gm.std().item()
                else:
                    grad_mean = 0.0
                    grad_std = 0.0

            log["steps"].append(step)
            log["loss_total"].append(float(loss))
            log["loss_eikonal"].append(float(l_eikonal))
            log["loss_boundary"].append(float(l_boundary))
            log["loss_order"].append(float(l_order))
            log["loss_coverage"].append(float(l_coverage))
            log["loss_angular"].append(float(l_angular))
            log["loss_z_smooth"].append(float(l_z_smooth))
            log["u_min"].append(u_min_val)
            log["u_max"].append(u_max_val)
            log["grad_mag_mean"].append(grad_mean)
            log["grad_mag_std"].append(grad_std)
            log["frac_correct_order"].append(frac_correct)

            if float(loss) < best_loss:
                best_loss = float(loss)
                torch.save(model.state_dict(), os.path.join(args.out_dir, "best.pt"))

            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            if step <= args.warmup_steps:
                phase = "warmup"
            elif eikonal_scale < 1.0:
                phase = f"ramp {eikonal_scale:.0%}"
            else:
                phase = "full"
            print(f"  Step {step:5d}/{args.steps} [{phase}] | "
                  f"loss={loss:.4f} "
                  f"(eik={float(l_eikonal):.4f}x{eikonal_scale:.2f} "
                  f"bnd={float(l_boundary):.4f} "
                  f"ord={float(l_order):.4f} cov={float(l_coverage):.4f} "
                  f"ang={float(l_angular):.4f} zsm={float(l_z_smooth):.4f}) | "
                  f"u=[{u_min_val:.2f},{u_max_val:.2f}] "
                  f"|∇u|={grad_mean:.4f}±{grad_std:.4f} "
                  f"order={frac_correct:.1%} | "
                  f"lr={lr:.6f} | {elapsed:.0f}s")

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
    parser = argparse.ArgumentParser(
        description="Self-supervised 3D unwrapping — no UV labels"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)

    # Architecture
    parser.add_argument("--n-fourier", type=int, default=512)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=6)

    # Training
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--n-pairs", type=int, default=16384,
                        help="Number of radius-ordered pairs per step")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--max-slices", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=2000,
                        help="Steps with boundary+ordering only (no eikonal)")
    parser.add_argument("--ramp-steps", type=int, default=3000,
                        help="Steps to linearly ramp eikonal from 0→full after warmup")

    # Loss weights
    parser.add_argument("--w-eikonal", type=float, default=1.0)
    parser.add_argument("--w-boundary", type=float, default=50.0)
    parser.add_argument("--w-order", type=float, default=1.0)
    parser.add_argument("--w-coverage", type=float, default=0.1)
    parser.add_argument("--w-angular", type=float, default=0.0,
                        help="Angular diversity weight (v3: break rotational symmetry)")
    parser.add_argument("--angular-alpha", type=float, default=5.0,
                        help="Sharpness of contrastive exp for angular loss")
    parser.add_argument("--w-z-smooth", type=float, default=0.0,
                        help="Z-smoothness weight (v3: u consistent across slices)")
    parser.add_argument("--order-margin", type=float, default=0.1)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
