"""
Train the residual inverse-mapping INR: f(u, z) = analytical_spiral(u) + INR(u, z).

Two phases (either may be empty):
  * supervised (steps <= --supervised-steps, needs --gt-npz): MSE of the
    predicted (x, y) against the ground-truth pixel position of (u, z)  [R1S];
  * self-supervised: attachment (squared distance to the emulsion, from a
    distance transform) + conformal loss on the residual's Jacobian (E = G,
    F = 0), ramped in over --ramp-steps, + optional residual L2  [R1n].

Model: --model-type inr (Fourier features + MLP) or grid (dense multi-resolution
2D feature grid + small MLP, encoder learning rate x --encoder-lr-mult).
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from virtual_unrolling.learned_unwrapping.surface_data import SurfaceDataset
from virtual_unrolling.learned_unwrapping.unwrap_model import DeformationINR
from virtual_unrolling.learned_unwrapping.grid_model import GridDeformation


def build_model(model_type="inr", n_fourier=256, sigma=10.0, hidden_dim=256,
                n_layers=4, grid_n_levels=8, grid_features=2,
                grid_base_u=64, grid_base_z=16,
                grid_finest_u=8192, grid_finest_z=256,
                grid_hidden=64, grid_mlp_layers=2, device="cuda"):
    """Residual model: zero-init so step-0 prediction == analytical_xy.

    model_type: "inr"  — Fourier features + MLP (DeformationINR)
                "grid" — dense multi-res 2D feature grid (GridDeformation)
    """
    if model_type == "inr":
        model = DeformationINR(
            n_fourier=n_fourier, sigma=sigma,
            hidden_dim=hidden_dim, n_layers=n_layers,
            input_dim=2, output_dim=2,
        ).to(device)
        nn.init.zeros_(model.head.weight)
        nn.init.zeros_(model.head.bias)
    elif model_type == "grid":
        model = GridDeformation(
            n_levels=grid_n_levels,
            n_features_per_level=grid_features,
            base_resolution_u=grid_base_u,
            base_resolution_z=grid_base_z,
            finest_resolution_u=grid_finest_u,
            finest_resolution_z=grid_finest_z,
            hidden_dim=grid_hidden,
            n_layers=grid_mlp_layers,
            output_dim=2,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model


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
        args.data_dir, device=device, diag_dir=args.out_dir,
        attachment=args.attachment, centerline_erode=args.centerline_erode,
    )

    if args.gt_npz:
        print(f"Loading synthetic ground truth: {args.gt_npz}")
        dataset.load_synthetic_gt(args.gt_npz)
    elif args.supervised_steps > 0:
        raise ValueError("--supervised-steps > 0 requires --gt-npz")

    model = build_model(
        model_type=args.model_type,
        n_fourier=args.n_fourier, sigma=args.sigma,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers_mlp,
        grid_n_levels=args.grid_n_levels,
        grid_features=args.grid_features,
        grid_base_u=args.grid_base_u, grid_base_z=args.grid_base_z,
        grid_finest_u=args.grid_finest_u, grid_finest_z=args.grid_finest_z,
        grid_hidden=args.grid_hidden, grid_mlp_layers=args.grid_mlp_layers,
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model_type} (2 → 2), {n_params:,} params, "
          f"zero-init for residual start")

    # Grid models use a split LR: the explicit feature parameters get a higher
    # LR (default 100× the MLP's) — standard for grid-based INRs.
    if args.model_type == "grid" and hasattr(model, "param_groups"):
        encoder_lr = args.lr * args.encoder_lr_mult
        optimizer = torch.optim.Adam(model.param_groups(grid_lr=encoder_lr, mlp_lr=args.lr))
        print(f"  Optimizer: encoder lr={encoder_lr:.2e}, MLP lr={args.lr:.2e} "
              f"({args.encoder_lr_mult}× split for --model-type {args.model_type})")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    log_file = open(log_path, "w")

    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        supervised_phase = step <= args.supervised_steps

        if supervised_phase:
            # MSE of pred_xy against GT pixel coordinates given GT u.
            batch = dataset.sample_supervised(args.batch_size)
            uv = batch["uv_norm"].detach().clone().requires_grad_(True)
            u_raw = batch["u_raw"]
            z_idx = batch["z_idx"]
            xy_target = batch["xy_target"]

            xy_base = dataset.analytical_xy_norm(u_raw)
            xy_res = model(uv)
            pred_xy = xy_base + xy_res

            l_mse = ((pred_xy - xy_target) ** 2).mean()

            # Attachment tracked for logging only during the supervised phase.
            with torch.no_grad():
                dist_val = dataset.sample_distance(pred_xy.detach(), z_idx)
                l_attach = (dist_val ** 2).mean()
            l_conformal_val = 0.0
            l_res_reg_val = float((xy_res.detach() ** 2).sum(dim=1).mean().item())
            E_mean = G_mean = F_mean = 0.0
            loss = l_mse
        else:
            batch = dataset.sample(args.batch_size)
            uv = batch["uv_norm"].detach().clone().requires_grad_(True)
            u_raw = batch["u_raw"]
            z_idx = batch["z_idx"]

            xy_base = dataset.analytical_xy_norm(u_raw)                # (B, 2)
            xy_res = model(uv)                                         # (B, 2)
            pred_xy = xy_base + xy_res

            # Attachment: squared distance to the target mask (smooth gradient).
            dist_val = dataset.sample_distance(pred_xy, z_idx)
            l_attach = (dist_val ** 2).mean()

            # Conformal on the RESIDUAL, not pred_xy. The analytical base already
            # carries the correct (non-conformal) scale: E_anal ~ arc-length², G_anal≈0.
            # Forcing E=G on pred_xy makes the INR cancel the analytical Jacobian,
            # which collapses ∇pred_xy → 0 (surface perfect-fits the emulsion but
            # revisits the same pixels for all u). Applying to xy_res keeps the
            # correction smooth without fighting the base geometry.
            grad_xr = torch.autograd.grad(
                xy_res[:, 0].sum(), uv, create_graph=True, retain_graph=True,
            )[0]
            grad_yr = torch.autograd.grad(
                xy_res[:, 1].sum(), uv, create_graph=True,
            )[0]
            l_conformal, E_mean, G_mean, F_mean = conformal_loss(grad_xr, grad_yr)

            # Residual magnitude anchor — stops xy_res from drifting when
            # attachment gradient is flat (especially with thin targets).
            l_res_reg = (xy_res ** 2).sum(dim=1).mean()

            # Ramp measured from the start of the self-supervised phase.
            ss_step = step - args.supervised_steps
            ramp = min(1.0, ss_step / max(1, args.ramp_steps))
            loss = (
                args.w_attach * l_attach
                + ramp * args.w_conformal * l_conformal
                + args.w_res_reg * l_res_reg
            )
            l_mse = torch.tensor(0.0, device=device)
            l_conformal_val = float(l_conformal.item())
            l_res_reg_val = float(l_res_reg.item())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0 or step == 1 or step == args.supervised_steps:
            # Residual magnitude in pixel units (denormalize by half image size)
            half_scale = 0.5 * (dataset.H + dataset.W) / 2.0
            res_px = (xy_res.detach().norm(dim=1).mean() * half_scale).item()
            entry = {
                "step": step,
                "phase": "supervised" if supervised_phase else "self_supervised",
                "loss": float(loss.item()),
                "l_mse": float(l_mse.item()),
                "l_attach": float(l_attach.item()),
                "l_conformal": l_conformal_val,
                "l_res_reg": l_res_reg_val,
                "E": E_mean, "G": G_mean, "F": F_mean,
                "res_px": res_px,
                "lr": optimizer.param_groups[0]["lr"],
            }
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()
            elapsed = time.time() - t0
            tag = "SUP" if supervised_phase else "SS "
            print(f"  [{tag}] step {step:>6d}  loss={loss.item():.5f}  "
                  f"mse={l_mse.item():.5f}  attach={l_attach.item():.4f}  "
                  f"conf={l_conformal_val:.4f}  reg={l_res_reg_val:.5f}  "
                  f"E={E_mean:.3f} G={G_mean:.3f} |F|={F_mean:.3f}  "
                  f"res_px={res_px:.2f}  ({elapsed:.0f}s)")

        if args.ckpt_every and step % args.ckpt_every == 0:
            torch.save(
                {"model": model.state_dict(), "step": step, "args": vars(args)},
                os.path.join(args.out_dir, f"ckpt_step{step}.pt"),
            )

    log_file.close()
    ckpt = {"model": model.state_dict(), "step": args.steps, "args": vars(args)}
    torch.save(ckpt, os.path.join(args.out_dir, "model_final.pt"))

    # Save dataset geometry (needed by eval for analytical_xy_norm)
    geometry = {
        "cx": dataset.cx, "cy": dataset.cy,
        "n_layers": dataset.n_layers,
        "theta_seam": dataset.theta_seam,
        "boundaries": dataset.boundaries.cpu().tolist(),
        "H": dataset.H, "W": dataset.W, "Z": dataset.Z,
        "z_indices": dataset.z_indices.tolist(),
        "ecc_amp": 0.0, "ecc_phase": 0.0, "theta_offset": [],
    }
    with open(os.path.join(args.out_dir, "geometry.json"), "w") as f:
        json.dump(geometry, f, indent=2)

    print(f"\nDone in {time.time() - t0:.0f}s.  Outputs: {args.out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train the residual inverse-mapping INR on a synthetic volume")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--gt-npz", type=str, default=None,
                        help="synthetic ground_truth.npz (required for --supervised-steps)")
    parser.add_argument("--supervised-steps", type=int, default=0,
                        help="first N steps: MSE vs the GT position (R1S); 0 = pure SS")
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--ramp-steps", type=int, default=3000,
                        help="conformal weight ramp, from the start of the SS phase")
    parser.add_argument("--batch-size", type=int, default=131072)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w-attach", type=float, default=1.0)
    parser.add_argument("--w-conformal", type=float, default=0.1)
    parser.add_argument("--w-res-reg", type=float, default=0.0)
    parser.add_argument("--attachment", choices=["film", "emulsion", "centerline"],
                        default="film", help="which mask drives the attachment loss")
    parser.add_argument("--centerline-erode", type=int, default=1)
    parser.add_argument("--model-type", choices=["inr", "grid"], default="inr")
    parser.add_argument("--n-fourier", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers-mlp", type=int, default=4)
    parser.add_argument("--encoder-lr-mult", type=float, default=100.0,
                        help="grid model: feature-grid LR = this x --lr")
    parser.add_argument("--grid-n-levels", type=int, default=8)
    parser.add_argument("--grid-features", type=int, default=2)
    parser.add_argument("--grid-base-u", type=int, default=64)
    parser.add_argument("--grid-base-z", type=int, default=16)
    parser.add_argument("--grid-finest-u", type=int, default=8192)
    parser.add_argument("--grid-finest-z", type=int, default=256)
    parser.add_argument("--grid-hidden", type=int, default=64)
    parser.add_argument("--grid-mlp-layers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=0,
                        help="save a checkpoint every N steps (0 = final only)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
