"""
Train the residual inverse-mapping INR: f(u, z) = analytical_spiral(u) + INR(u, z).

Pure self-supervised: attachment loss (surface must live on the film mask) +
conformal loss (Jacobian is locally angle/length-preserving). No CDF targets.

See the plan in .claude/plans/joyful-growing-marshmallow.md for design rationale.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.inr.surface_data import SurfaceDataset
from unwrapping.inr.unwrap_model import DeformationINR


def build_model(n_fourier=256, sigma=10.0, hidden_dim=256, n_layers=4, device="cuda"):
    """Residual INR: zero-init the output head so step-0 prediction == analytical."""
    model = DeformationINR(
        n_fourier=n_fourier, sigma=sigma,
        hidden_dim=hidden_dim, n_layers=n_layers,
        input_dim=2, output_dim=2,
    ).to(device)
    nn.init.zeros_(model.head.weight)
    nn.init.zeros_(model.head.bias)
    return model


def attachment_loss(mask_val):
    """Surface must sit on film pixels → mask value should be 1 everywhere."""
    return ((1.0 - mask_val) ** 2).mean()


def conformal_loss(grad_x, grad_y):
    """E = G and F = 0 on the 2×2 Jacobian (angle- and scale-preserving)."""
    E = grad_x[:, 0] ** 2 + grad_y[:, 0] ** 2
    G = grad_x[:, 1] ** 2 + grad_y[:, 1] ** 2
    Fmet = grad_x[:, 0] * grad_x[:, 1] + grad_y[:, 0] * grad_y[:, 1]
    loss = ((E - G) ** 2).mean() + (4.0 * Fmet ** 2).mean()
    return loss, E.mean().item(), G.mean().item(), Fmet.abs().mean().item()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Loading dataset from {args.data_dir}")
    dataset = SurfaceDataset(
        args.data_dir, max_slices=args.max_slices, device=device,
        diag_dir=args.out_dir,
    )

    model = build_model(
        n_fourier=args.n_fourier, sigma=args.sigma,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers_mlp,
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: DeformationINR (2 → 2), {n_params:,} params, "
          f"head zero-init for residual start")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    log_keys = ["step", "loss", "l_attach", "l_conformal", "E", "G", "F",
                "res_px", "lr"]
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    log_file = open(log_path, "w")

    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        batch = dataset.sample(args.batch_size)
        uv = batch["uv_norm"].detach().clone().requires_grad_(True)
        u_raw = batch["u_raw"]
        z_idx = batch["z_idx"]

        xy_base = dataset.analytical_xy_norm(u_raw)                # (B, 2)
        xy_res = model(uv)                                         # (B, 2)
        pred_xy = xy_base + xy_res

        # Attachment — binary mask, bilinear-sampled at predicted (x, y, z)
        mask_val = dataset.sample_mask(pred_xy, z_idx)
        l_attach = attachment_loss(mask_val)

        # Conformal via autograd on pred_xy w.r.t. uv
        grad_x = torch.autograd.grad(
            pred_xy[:, 0].sum(), uv, create_graph=True, retain_graph=True,
        )[0]
        grad_y = torch.autograd.grad(
            pred_xy[:, 1].sum(), uv, create_graph=True,
        )[0]
        l_conformal, E_mean, G_mean, F_mean = conformal_loss(grad_x, grad_y)

        ramp = min(1.0, step / args.ramp_steps)
        loss = args.w_attach * l_attach + ramp * args.w_conformal * l_conformal

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0 or step == 1:
            # Residual magnitude in pixel units (denormalize by half image size)
            half_scale = 0.5 * (dataset.H + dataset.W) / 2.0
            res_px = (xy_res.detach().norm(dim=1).mean() * half_scale).item()
            entry = {
                "step": step,
                "loss": float(loss.item()),
                "l_attach": float(l_attach.item()),
                "l_conformal": float(l_conformal.item()),
                "E": E_mean, "G": G_mean, "F": F_mean,
                "res_px": res_px,
                "lr": optimizer.param_groups[0]["lr"],
            }
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()
            elapsed = time.time() - t0
            print(f"  step {step:>6d}  loss={loss.item():.4f}  "
                  f"attach={l_attach.item():.4f}  conf={l_conformal.item():.4f}  "
                  f"E={E_mean:.3f} G={G_mean:.3f} |F|={F_mean:.3f}  "
                  f"res_px={res_px:.2f}  ({elapsed:.0f}s)")

        if args.ckpt_every and step % args.ckpt_every == 0:
            torch.save(
                {"model": model.state_dict(), "step": step, "args": vars(args)},
                os.path.join(args.out_dir, f"ckpt_step{step}.pt"),
            )

    log_file.close()
    torch.save(
        {"model": model.state_dict(), "step": args.steps, "args": vars(args)},
        os.path.join(args.out_dir, "model_final.pt"),
    )

    # Save dataset geometry (needed by eval for analytical_xy_norm)
    geometry = {
        "cx": dataset.cx, "cy": dataset.cy,
        "n_layers": dataset.n_layers,
        "theta_seam": dataset.theta_seam,
        "boundaries": dataset.boundaries.cpu().tolist(),
        "H": dataset.H, "W": dataset.W, "Z": dataset.Z,
        "z_indices": dataset.z_indices.tolist(),
    }
    with open(os.path.join(args.out_dir, "geometry.json"), "w") as f:
        json.dump(geometry, f, indent=2)

    print(f"\nDone in {time.time() - t0:.0f}s.  Outputs: {args.out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Residual inverse-mapping INR for film unwrapping."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-slices", type=int, default=None)

    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--ramp-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--w-attach", type=float, default=1.0)
    parser.add_argument("--w-conformal", type=float, default=0.1)

    parser.add_argument("--n-fourier", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers-mlp", type=int, default=4)

    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=0,
                        help="Save checkpoint every N steps; 0 disables intermediate saves.")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
