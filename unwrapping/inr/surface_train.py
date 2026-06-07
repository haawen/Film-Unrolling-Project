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
from unwrapping.inr.perwinding_model import PerWindingParametric
from unwrapping.inr.bspline_model import BSplineDeformation
from unwrapping.inr.grid_model import GridDeformation
from unwrapping.inr.hash_model import HashDeformation
from unwrapping.inr.wire_model import WIREDeformation
from unwrapping.inr.finer_model import FINERDeformation
from unwrapping.inr.strip_discriminator import load_discriminator
from unwrapping.inr.patch_feature_loss import (
    compute_patch_feature_zcoh,
    compute_patch_feature_infonce,
    compute_frame_periodicity_loss,
)


def build_model(model_type="inr", n_fourier=256, sigma=10.0, hidden_dim=256,
                n_layers=4, n_windings=1, n_harmonics=10,
                bspline_u_per_winding=4, bspline_n_z=3,
                grid_n_levels=8, grid_features=2,
                grid_base_u=64, grid_base_z=16,
                grid_finest_u=8192, grid_finest_z=256,
                grid_hidden=64, grid_mlp_layers=2,
                hash_n_levels=16, hash_features=2,
                hash_log2_size=19, hash_base_res=16, hash_finest_res=8192,
                hash_hidden=64, hash_mlp_layers=2,
                wire_omega=20.0, wire_sigma=10.0,
                finer_omega=30.0, finer_bias_k=5.0,
                device="cuda"):
    """Residual model: zero-init so step-0 prediction == analytical_xy.

    model_type: "inr"        — Fourier features + MLP (DeformationINR)
                "perwinding" — per-winding Fourier-in-θ (PerWindingParametric)
                "bspline"    — cubic B-spline tensor product (BSplineDeformation)
                "grid"       — G1: dense multi-res 2D feature grid (GridDeformation)
                "hash"       — H1: Instant-NGP hash encoding (HashDeformation)
                "wire"       — S2: WIRE Gabor wavelet MLP (WIREDeformation)
                "finer"      — S3: FINER variable-periodic MLP (FINERDeformation)
    """
    if model_type == "inr":
        model = DeformationINR(
            n_fourier=n_fourier, sigma=sigma,
            hidden_dim=hidden_dim, n_layers=n_layers,
            input_dim=2, output_dim=2,
        ).to(device)
        nn.init.zeros_(model.head.weight)
        nn.init.zeros_(model.head.bias)
    elif model_type == "perwinding":
        model = PerWindingParametric(
            n_layers=n_windings, n_harmonics=n_harmonics, output_dim=2,
        ).to(device)
    elif model_type == "bspline":
        model = BSplineDeformation(
            n_u_spans=bspline_u_per_winding * max(1, n_windings),
            n_z_spans=bspline_n_z,
            output_dim=2,
        ).to(device)
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
    elif model_type == "hash":
        model = HashDeformation(
            n_levels=hash_n_levels,
            n_features_per_level=hash_features,
            log2_hashmap_size=hash_log2_size,
            base_resolution=hash_base_res,
            finest_resolution=hash_finest_res,
            hidden_dim=hash_hidden,
            n_layers=hash_mlp_layers,
            output_dim=2,
        ).to(device)
    elif model_type == "wire":
        model = WIREDeformation(
            hidden_dim=hidden_dim, n_layers=n_layers,
            omega=wire_omega, sigma=wire_sigma,
            input_dim=2, output_dim=2,
        ).to(device)
    elif model_type == "finer":
        model = FINERDeformation(
            hidden_dim=hidden_dim, n_layers=n_layers,
            omega_0=finer_omega, bias_k=finer_bias_k,
            input_dim=2, output_dim=2,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
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
        attachment=args.attachment, centerline_erode=args.centerline_erode,
        winding_detector=args.winding_detector,
        data_driven_base=args.data_driven_base,
    )

    if args.gt_npz:
        print(f"Loading synthetic ground truth: {args.gt_npz}")
        dataset.load_synthetic_gt(args.gt_npz)
    elif args.supervised_steps > 0:
        raise ValueError("--supervised-steps > 0 requires --gt-npz")

    if args.w_winding_cc > 0:
        dataset.build_winding_label_volume(erode_iters=args.cc_erode_iters)

    model = build_model(
        model_type=args.model_type,
        n_fourier=args.n_fourier, sigma=args.sigma,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers_mlp,
        n_windings=dataset.n_layers, n_harmonics=args.n_harmonics,
        bspline_u_per_winding=args.bspline_u_per_winding,
        bspline_n_z=args.bspline_n_z,
        grid_n_levels=args.grid_n_levels,
        grid_features=args.grid_features,
        grid_base_u=args.grid_base_u, grid_base_z=args.grid_base_z,
        grid_finest_u=args.grid_finest_u, grid_finest_z=args.grid_finest_z,
        grid_hidden=args.grid_hidden, grid_mlp_layers=args.grid_mlp_layers,
        hash_n_levels=args.hash_n_levels, hash_features=args.hash_features,
        hash_log2_size=args.hash_log2_size,
        hash_base_res=args.hash_base_res, hash_finest_res=args.hash_finest_res,
        hash_hidden=args.hash_hidden, hash_mlp_layers=args.hash_mlp_layers,
        wire_omega=args.wire_omega, wire_sigma=args.wire_sigma,
        finer_omega=args.finer_omega, finer_bias_k=args.finer_bias_k,
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model_type} (2 → 2), {n_params:,} params, "
          f"zero-init for residual start")

    # A3 (joint autodecoder): a small INR g_φ(u, z) → s ∈ [0, 1] that
    # represents the unrolled strip content. Trained jointly with f_θ. The
    # render loss MSE(intensity_model(g_φ), CT_at_pred_xy) forces f_θ to
    # *explain* actual CT intensities, not just sit on the emulsion. Mask-
    # weighted so off-emulsion regions don't break the loss.
    strip_model = None
    if args.w_autodecoder > 0:
        strip_model = DeformationINR(
            n_fourier=args.strip_n_fourier, sigma=args.strip_sigma,
            hidden_dim=args.strip_hidden_dim, n_layers=args.strip_n_layers,
            input_dim=2, output_dim=1,
        ).to(device)
        # Zero-init head: sigmoid(0) = 0.5, strip starts as constant gray.
        nn.init.zeros_(strip_model.head.weight)
        nn.init.zeros_(strip_model.head.bias)
        n_strip_params = sum(p.numel() for p in strip_model.parameters())
        print(f"A3 strip INR (2 → 1), {n_strip_params:,} params, "
              f"sigma={args.strip_sigma}, hidden={args.strip_hidden_dim}, "
              f"L={args.strip_n_layers} → sigmoid → s ∈ [0, 1]")

    # Intensity model constants. Read from GT npz when available
    # (synthetic); for real data these would need to be estimated.
    if args.w_autodecoder > 0 and args.gt_npz:
        _gt = np.load(args.gt_npz)
        strip_base = float(_gt["film_base_intensity"]) if "film_base_intensity" in _gt.files else 0.12
        strip_max  = float(_gt["emulsion_max_intensity"]) if "emulsion_max_intensity" in _gt.files else 0.85
        print(f"  Intensity model: base={strip_base:.3f}, max={strip_max:.3f}")
    else:
        strip_base, strip_max = 0.12, 0.85

    # Track 2: warm-init residual weights from a prior checkpoint (curriculum).
    if args.init_from:
        print(f"Loading initial weights from: {args.init_from}")
        prior = torch.load(args.init_from, map_location=device)
        prior_state = prior["model"] if "model" in prior else prior
        missing, unexpected = model.load_state_dict(prior_state, strict=False)
        if missing:
            print(f"  init-from: missing keys: {missing}")
        if unexpected:
            print(f"  init-from: unexpected keys: {unexpected}")
        print(f"  init-from: loaded (optimizer state ignored).")

    # Step A — anchor pose-tether: snapshot the model's residual at a fixed
    # set of (u, z) anchor points, freeze it, then penalize drift away from
    # that snapshot during training. Acts as a soft pseudo-GT replacing the
    # MSE anchor that made synthetic R1PF1ws work; allows patch-feature loss
    # to refine without finding the trivial-collapse mode.
    anchor_uv = None
    anchor_u_raw_t = None
    anchor_res_init = None
    if args.w_pose_tether > 0:
        n_anchor = args.pose_tether_anchors
        torch.manual_seed(0)  # deterministic anchor sampling
        anchor_u_raw_t = torch.rand(n_anchor, device=device) * dataset.n_layers
        anchor_z_idx = torch.randint(0, dataset.Z, (n_anchor,), device=device)
        anchor_u_norm = anchor_u_raw_t / dataset.n_layers * 2.0 - 1.0
        if dataset.Z > 1:
            anchor_z_norm = anchor_z_idx.float() / (dataset.Z - 1) * 2.0 - 1.0
        else:
            anchor_z_norm = torch.zeros(n_anchor, device=device)
        anchor_uv = torch.stack([anchor_u_norm, anchor_z_norm], dim=1)
        with torch.no_grad():
            anchor_res_init = model(anchor_uv).detach().clone()
        print(f"Pose-tether enabled: {n_anchor} anchor (u, z) points, "
              f"initial residual L2 mean = "
              f"{anchor_res_init.norm(dim=1).mean().item():.4f} (normalized)")

    # Analytical-base learnable parameters — composable: any subset of
    #   --learn-eccentricity        (R2b: global amp·cos(θ−φ))
    #   --per-winding-eccentricity  (R3 lever 2: per-winding amp_k·cos(θ−φ_k))
    #   --per-winding-theta-phase   (A4: per-winding θ_offset[k])
    # share a single dedicated optimizer (ecc_lr_mult × main LR, no scheduler).
    base_params = []
    if args.learn_eccentricity:
        base_params += dataset.enable_learnable_eccentricity()
        print("Learnable global eccentricity enabled (2 scalars).")
    if args.per_winding_eccentricity:
        base_params += dataset.enable_per_winding_eccentricity()
        print(f"Per-winding eccentricity enabled "
              f"({dataset.n_layers} windings × 2 scalars).")
    if args.per_winding_theta_phase:
        base_params += dataset.enable_per_winding_theta_phase()
        print(f"A4: per-winding angular phase enabled "
              f"({dataset.n_layers} scalars).")

    if base_params:
        ecc_lr = args.lr * args.ecc_lr_mult
        ecc_optimizer = torch.optim.Adam(base_params, lr=ecc_lr)
        print(f"  Base-params optimizer: lr={ecc_lr:.2e} "
              f"({args.ecc_lr_mult}× INR LR, no cosine).")
    else:
        ecc_optimizer = None
    # Main optimizer: f_θ params + strip g_φ params (when A3 enabled). Joint
    # update with the same LR; cosine schedule applies to both.
    # Grid/hash models use a split LR: the explicit feature parameters get a
    # higher LR (default 100× the MLP's) — this is standard for grid-based
    # INRs and matters for convergence.
    if args.model_type in ("grid", "hash") and hasattr(model, "param_groups"):
        encoder_lr = args.lr * args.encoder_lr_mult
        param_groups = model.param_groups(grid_lr=encoder_lr, mlp_lr=args.lr) \
            if args.model_type == "grid" \
            else model.param_groups(hash_lr=encoder_lr, mlp_lr=args.lr)
        if strip_model is not None:
            param_groups.append({"params": list(strip_model.parameters()),
                                 "lr": args.lr})
        optimizer = torch.optim.Adam(param_groups)
        print(f"  Optimizer: encoder lr={encoder_lr:.2e}, MLP lr={args.lr:.2e} "
              f"({args.encoder_lr_mult}× split for --model-type {args.model_type})")
    elif strip_model is not None:
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(strip_model.parameters()),
            lr=args.lr,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    # Load strip discriminator (frozen) if specified — used as Path D loss.
    discriminator = None
    disc_patch_w = None
    if args.discriminator_ckpt:
        print(f"Loading strip discriminator: {args.discriminator_ckpt}")
        discriminator, disc_ckpt = load_discriminator(
            args.discriminator_ckpt, device=device,
        )
        disc_patch_w = disc_ckpt["args"]["patch_w"]
        print(f"  Discriminator loaded, patch_w={disc_patch_w}, frozen.")

    log_keys = ["step", "loss", "l_attach", "l_conformal", "E", "G", "F",
                "res_px", "lr"]
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    log_file = open(log_path, "w")

    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        supervised_phase = step <= args.supervised_steps

        if supervised_phase:
            # Warm-start: MSE of pred_xy against GT pixel coordinates given GT u.
            batch = dataset.sample_supervised(args.batch_size)
            uv = batch["uv_norm"].detach().clone().requires_grad_(True)
            u_raw = batch["u_raw"]
            z_idx = batch["z_idx"]
            xy_target = batch["xy_target"]

            # Z3 diagnostic: how robust is the supervised path to LABEL NOISE?
            # If MAPS (skeleton-derived pseudo-u) gives noisy labels, this tells
            # us the noise budget. xy_target is in normalized [-1, 1]; px
            # noise = norm_noise * (W-1)/2. Default 0 = clean GT.
            if args.label_noise_px > 0:
                noise_norm_x = args.label_noise_px / max(1.0, (dataset.W - 1) / 2.0)
                noise_norm_y = args.label_noise_px / max(1.0, (dataset.H - 1) / 2.0)
                noise = torch.randn_like(xy_target)
                noise[:, 0] *= noise_norm_x
                noise[:, 1] *= noise_norm_y
                xy_target = xy_target + noise

            xy_base = dataset.analytical_xy_norm(u_raw)
            xy_res = model(uv)
            pred_xy = xy_base + xy_res

            l_mse = ((pred_xy - xy_target) ** 2).mean()

            # Strip-intensity MSE (auxiliary loss-side lever, R3).
            # coord-MSE alone has many local minima with similar values but
            # very different SSIM. Adding MSE(CT[pred_xy], CT[xy_target])
            # weights coord errors by local CT gradient — small in smooth
            # emulsion regions, large near frame boundaries — which directly
            # tracks SSIM. dataset.sample_image is bilinear and differentiable.
            l_strip_mse_val = 0.0
            if args.w_strip_mse > 0:
                intens_pred = dataset.sample_image(pred_xy, z_idx)
                with torch.no_grad():
                    intens_target = dataset.sample_image(xy_target, z_idx)
                l_strip_mse = ((intens_pred - intens_target) ** 2).mean()
                l_strip_mse_val = float(l_strip_mse.item())
            else:
                l_strip_mse = torch.tensor(0.0, device=device)

            # Attachment + conformal tracked for logging only during warm-start.
            with torch.no_grad():
                if args.use_distance_attach:
                    dist_val = dataset.sample_distance(pred_xy.detach(), z_idx)
                    l_attach = (dist_val ** 2).mean()
                else:
                    mask_val = dataset.sample_mask(pred_xy.detach(), z_idx)
                    l_attach = ((1.0 - mask_val) ** 2).mean()
            # skip conformal autograd during supervised phase (cheaper)
            l_conformal_val = 0.0
            l_res_reg_val = float((xy_res.detach() ** 2).sum(dim=1).mean().item())
            l_zcoh_val = 0.0
            l_band_val = 0.0
            l_radial_cap_val = 0.0
            l_wcc_val = 0.0
            l_speed_val = 0.0
            l_intensity_val = 0.0
            l_int_zcoh_val = 0.0
            l_strip_q_val = 0.0
            l_disc_val = 0.0
            l_patch_feat_val = 0.0
            l_frame_period_val = 0.0
            l_maps_val = 0.0
            l_render_val = 0.0
            l_infonce_val = 0.0
            l_tether_val = 0.0
            E_mean = G_mean = F_mean = 0.0
            loss = l_mse + args.w_strip_mse * l_strip_mse
        else:
            batch = dataset.sample(args.batch_size)
            uv = batch["uv_norm"].detach().clone().requires_grad_(True)
            u_raw = batch["u_raw"]
            z_idx = batch["z_idx"]

            xy_base = dataset.analytical_xy_norm(u_raw)                # (B, 2)
            xy_res = model(uv)                                         # (B, 2)
            pred_xy = xy_base + xy_res

            # Attachment: distance² (smooth gradient off-target) or (1-mask)² fallback.
            if args.use_distance_attach:
                dist_val = dataset.sample_distance(pred_xy, z_idx)
                l_attach = (dist_val ** 2).mean()
            else:
                mask_val = dataset.sample_mask(pred_xy, z_idx)
                l_attach = attachment_loss(mask_val)

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

            # R3 lever "winding-band consistency": enforce that pred_xy lands
            # in the radial band of the expected winding (=floor(u_raw)).
            # Differentiable hinge: penalize r being outside [b_k, b_{k+1}].
            # Breaks the radial degeneracy that lets the surface hop windings
            # while still satisfying attachment on a thin emulsion mask.
            l_band_val = 0.0
            if args.w_winding_band > 0:
                # Denormalize pred_xy to pixel coords, compute radial distance.
                pred_x_px = (pred_xy[:, 0] + 1.0) * 0.5 * (dataset.W - 1)
                pred_y_px = (pred_xy[:, 1] + 1.0) * 0.5 * (dataset.H - 1)
                r_pred = torch.sqrt(
                    (pred_x_px - dataset.cx) ** 2 + (pred_y_px - dataset.cy) ** 2
                )
                k = u_raw.long().clamp(0, dataset.n_layers - 1)
                r_lo = dataset.boundaries[k]
                r_hi = dataset.boundaries[k + 1]
                # Soft hinge: zero inside the band, quadratic outside.
                over = torch.relu(r_pred - r_hi)
                under = torch.relu(r_lo - r_pred)
                l_band = (over ** 2 + under ** 2).mean()
                l_band_val = float(l_band.item())

            # B3 (radial residual cap): penalize the radial component of
            # (pred_xy − analytical_xy) when it exceeds half the local layer
            # spacing. Encodes "do not hop into a neighbour winding" relative
            # to where the analytical base actually sits (emulsion-offset
            # applied), unlike --w-winding-band which uses film-centerline
            # boundaries. Quadratic outside the band → zero gradient when
            # inside, so it does not fight small SS-driven corrections.
            l_radial_cap_val = 0.0
            if args.w_radial_cap > 0:
                base_x_px = (xy_base[:, 0] + 1.0) * 0.5 * (dataset.W - 1)
                base_y_px = (xy_base[:, 1] + 1.0) * 0.5 * (dataset.H - 1)
                r_base = torch.sqrt(
                    (base_x_px - dataset.cx) ** 2
                    + (base_y_px - dataset.cy) ** 2
                )
                pred_x_px = (pred_xy[:, 0] + 1.0) * 0.5 * (dataset.W - 1)
                pred_y_px = (pred_xy[:, 1] + 1.0) * 0.5 * (dataset.H - 1)
                r_pred = torch.sqrt(
                    (pred_x_px - dataset.cx) ** 2
                    + (pred_y_px - dataset.cy) ** 2
                )
                cap_px = 0.5 * dataset.layer_spacing
                over = torch.relu((r_pred - r_base).abs() - cap_px)
                l_radial_cap = (over ** 2).mean()
                l_radial_cap_val = float(l_radial_cap.item())

            # R3 lever "winding-label CC": sample the precomputed per-pixel
            # winding-label volume (CC on eroded film, ordered by radius,
            # propagated by EDT) at pred_xy and penalize the squared diff
            # from the expected winding (=floor(u_raw)). Bilinear sampling
            # gives a smooth ramp across air gaps, so the gradient pushes
            # pred toward the correct winding when it lands on the wrong one.
            l_wcc_val = 0.0
            if args.w_winding_cc > 0:
                wlbl = dataset.sample_winding_label(pred_xy, z_idx)
                expected_w = u_raw.clamp(0, dataset.n_layers - 1).floor()
                l_wcc = ((wlbl - expected_w) ** 2).mean()
                l_wcc_val = float(l_wcc.item())

            # R3 lever "arc-length speed": ∂pred_xy/∂u should advance at a
            # roughly constant pixel-per-u rate (the analytical's expected
            # arc/u). Penalize variance to discourage "dwelling" / angle-jumps
            # without pinning a specific value (which would over-constrain).
            l_speed_val = 0.0
            if args.w_arc_speed > 0:
                # ∂pred_xy/∂u (column 0 of uv) — already needed graph for conformal
                # so reuse: speed² is just E from the first fundamental form on pred_xy.
                grad_xp = torch.autograd.grad(
                    pred_xy[:, 0].sum(), uv, create_graph=True, retain_graph=True,
                )[0]
                grad_yp = torch.autograd.grad(
                    pred_xy[:, 1].sum(), uv, create_graph=True, retain_graph=True,
                )[0]
                speed_sq = grad_xp[:, 0] ** 2 + grad_yp[:, 0] ** 2
                l_speed = ((speed_sq - speed_sq.mean().detach()) ** 2).mean()
                l_speed_val = float(l_speed.item())

            # Intensity-profile attachment (Path B): mask attachment is flat
            # inside the emulsion band so the residual is free to drift any-
            # where on emulsion. Sampling raw image intensity at pred_xy ±
            # δ·n (where n is the in-plane surface normal) gives a true
            # gradient — the on-surface point should be brighter than its
            # off-surface neighbors because emulsion is bright vs film_base.
            # This is the Vesuvius-style fix for the 0.55 ceiling.
            l_intensity_val = 0.0
            if args.w_intensity > 0:
                # ∂pred_xy/∂u via autograd (column 0 of uv).
                grad_xp_int = torch.autograd.grad(
                    pred_xy[:, 0].sum(), uv,
                    create_graph=True, retain_graph=True,
                )[0]
                grad_yp_int = torch.autograd.grad(
                    pred_xy[:, 1].sum(), uv,
                    create_graph=True, retain_graph=True,
                )[0]
                tx, ty = grad_xp_int[:, 0], grad_yp_int[:, 0]
                norm_mag = torch.sqrt(tx ** 2 + ty ** 2 + 1e-8)
                # 90° rotation of tangent → in-plane normal (radial-ish).
                nx = -ty / norm_mag
                ny = tx / norm_mag
                # Convert pixel offset to normalized [-1,1] coords.
                dx_norm = args.intensity_delta_px * 2.0 / (dataset.W - 1)
                dy_norm = args.intensity_delta_px * 2.0 / (dataset.H - 1)
                offset = torch.stack([dx_norm * nx, dy_norm * ny], dim=1)
                pred_pos = pred_xy + offset
                pred_neg = pred_xy - offset
                I_center = dataset.sample_image(pred_xy, z_idx)
                I_pos = dataset.sample_image(pred_pos, z_idx)
                I_neg = dataset.sample_image(pred_neg, z_idx)
                margin = args.intensity_margin
                l_intensity = (
                    torch.relu(I_pos - I_center + margin) ** 2
                    + torch.relu(I_neg - I_center + margin) ** 2
                ).mean()
                l_intensity_val = float(l_intensity.item())

            # z-coherence: synthetic geometry is z-invariant by construction
            # (the per-pixel jitter is computed once and applied to every
            # slice). Penalizing |pred_xy(u,z) - pred_xy(u,z+1)|² breaks the
            # underdetermined-attachment degeneracy on imperfect_4k by
            # forcing the surface to vote across z. Weight 0 = disabled.
            l_zcoh_val = 0.0
            l_int_zcoh_val = 0.0
            if (args.w_z_coherence > 0 or args.w_intensity_zcoh > 0) and dataset.Z >= 2:
                zb = dataset.sample_z_pairs(args.z_coh_batch_size)
                if zb is not None:
                    base_z = dataset.analytical_xy_norm(zb["u_raw"])
                    res_a = model(zb["uv_a"])
                    res_b = model(zb["uv_b"])
                    pred_a = base_z + res_a
                    pred_b = base_z + res_b
                    if args.w_z_coherence > 0:
                        l_zcoh = ((pred_a - pred_b) ** 2).sum(dim=1).mean()
                        l_zcoh_val = float(l_zcoh.item())
                    # Intensity z-coherence (Path B v2): same u across z should
                    # sample the same image content. Synthetic geometry is
                    # z-invariant so intensities at the correct angular
                    # mapping should match. Catches the imperfect_4k failure
                    # mode (correct radial position, wrong angular).
                    if args.w_intensity_zcoh > 0:
                        # z indices for sampling — sample_image needs int z.
                        # zb stores them in uv_*[:, 1] as normalized; recover.
                        z_a = (((zb["uv_a"][:, 1] + 1.0) * 0.5) * (dataset.Z - 1)
                               ).round().long().clamp(0, dataset.Z - 1)
                        z_b = (((zb["uv_b"][:, 1] + 1.0) * 0.5) * (dataset.Z - 1)
                               ).round().long().clamp(0, dataset.Z - 1)
                        I_a = dataset.sample_image(pred_a, z_a)
                        I_b = dataset.sample_image(pred_b, z_b)
                        if args.mask_weight_int_zcoh:
                            # Z2: mask-weight to prevent the closed R1B4/B5 mode
                            # (off-emulsion uniform-region collapse). With the
                            # mask gate, the off-emulsion regions contribute 0
                            # to z-coh so attachment dominates; z-coh only
                            # constrains the mapping where it's actually on
                            # the emulsion.
                            M_a = dataset.sample_mask(pred_a, z_a).detach()
                            M_b = dataset.sample_mask(pred_b, z_b).detach()
                            w_pair = (M_a * M_b).clamp(0.0, 1.0)
                            w_sum = w_pair.sum().clamp(min=1.0)
                            l_int_zcoh = (((I_a - I_b) ** 2) * w_pair).sum() / w_sum
                        else:
                            l_int_zcoh = ((I_a - I_b) ** 2).mean()
                        l_int_zcoh_val = float(l_int_zcoh.item())

            # Strip-quality loss (Path C): differentiable proxies that mirror
            # the unsupervised strip_score metrics. Render a partial strip
            # from the model's current pred_xy, compute variance + horizontal
            # high-frequency energy on it, and maximize both.
            #
            # Why these signals:
            # - max variance: collapsed/wrong unwrap = featureless region with
            #   little intensity variation. Correct unwrap = bright/dark frames
            #   = high variance. Resists the trivial collapse modes that broke
            #   intensity z-coherence (uniform-region drift gives 0 variance).
            # - max HF energy: correct unwrap has SHARP frame edges along u.
            #   Wrong/blurred unwrap (averaging emulsion across windings) has
            #   low HF content. FFT power in top frequency band is maximized
            #   exactly when sharp transitions are preserved.
            l_strip_q_val = 0.0
            if args.w_strip_quality > 0:
                # Render a partial strip on a dense u grid at a single z.
                n_cols = args.strip_q_cols
                z_target = min(dataset.Z // 2, dataset.Z - 1)
                u_grid = torch.linspace(
                    0.0, dataset.n_layers - 1e-6, n_cols, device=device
                )
                u_norm_strip = u_grid / dataset.n_layers * 2.0 - 1.0
                z_norm_strip = (
                    (z_target / max(1, dataset.Z - 1)) * 2.0 - 1.0
                ) * torch.ones(n_cols, device=device)
                uv_strip = torch.stack([u_norm_strip, z_norm_strip], dim=1)
                z_idx_strip = torch.full(
                    (n_cols,), z_target, device=device, dtype=torch.long
                )
                base_strip = dataset.analytical_xy_norm(u_grid)
                res_strip = model(uv_strip)
                pred_strip = base_strip + res_strip
                I_strip = dataset.sample_image(pred_strip, z_idx_strip)
                # 1) variance term: -var(strip) (maximize)
                l_var = -I_strip.var()
                # 2) HF energy: top 25% of FFT spectrum power, normalized.
                I_centered = I_strip - I_strip.mean()
                spec = torch.fft.rfft(I_centered)
                power = spec.real ** 2 + spec.imag ** 2
                k_cut = int(power.shape[0] * 0.75)
                hf_frac = power[k_cut:].sum() / (power.sum() + 1e-9)
                l_hf = -hf_frac
                l_strip_q = l_var + args.strip_q_hf_weight * l_hf
                l_strip_q_val = float(l_strip_q.item())

            # Discriminator loss (Path D): render strip patches at random
            # u-positions × all z-rows, feed through frozen StripDiscriminator,
            # maximize "real movie content" probability. The CNN was trained
            # on synthetic clean GT strips vs corruptions (shuffles, blurs,
            # jitter offsets), so it encodes high-dimensional learned features
            # that aggregate-property losses can't capture. Resists the
            # "satisfy aggregate metric trivially" failure mode by requiring
            # patch-level structure that matches real movie content.
            l_disc_val = 0.0
            if discriminator is not None and args.w_discriminator > 0:
                n_patches = args.disc_n_patches
                pw = disc_patch_w
                # Random u_start in [0, n_layers - patch_extent_in_u]
                # patch_w (strip pixels) → u extent: depends on strip arc length
                # which we approximate as constant rate n_layers / total_arc.
                # In u-space: patch covers patch_w / strip_pixels_per_winding.
                # Use a reasonable default of n_layers/8000 windings per pixel
                # (clean_4k: 5128 px/winding ⇒ 64px ≈ 0.0125 windings).
                pix_per_w = max(50.0, dataset.total_arc / dataset.n_layers
                                / 1.0)  # arc length per radian × 2π / px ≈ ...
                # Simpler: fixed u extent per patch covering ~half a winding
                u_extent = pw / 5128.0  # tuned for ~5000 strip-px per winding
                u_starts = (torch.rand(n_patches, device=device)
                            * (dataset.n_layers - u_extent))
                u_offsets = torch.linspace(0, u_extent, pw, device=device)
                # (n_patches, pw)
                u_grid_2d = u_starts.unsqueeze(1) + u_offsets.unsqueeze(0)
                # broadcast across z: (n_patches, n_z, pw)
                n_z = dataset.Z
                u_flat = u_grid_2d.unsqueeze(1).expand(-1, n_z, -1).reshape(-1)
                z_all = torch.arange(n_z, device=device, dtype=torch.long)
                z_flat = z_all.view(1, n_z, 1).expand(
                    n_patches, -1, pw
                ).reshape(-1)
                u_norm_p = u_flat / dataset.n_layers * 2.0 - 1.0
                z_norm_p = z_flat.float() / max(1, dataset.Z - 1) * 2.0 - 1.0
                uv_p = torch.stack([u_norm_p, z_norm_p], dim=1)
                base_p = dataset.analytical_xy_norm(u_flat)
                res_p = model(uv_p)
                pred_p = base_p + res_p
                I_p = dataset.sample_image(pred_p, z_flat)
                # (n_patches, 1, n_z, pw)
                I_p_2d = I_p.view(n_patches, 1, n_z, pw)
                disc_score = discriminator.score(I_p_2d)
                l_disc = -disc_score.mean()
                l_disc_val = float(l_disc.item())

            # Ramp measured from the start of the self-supervised phase.
            ss_step = step - args.supervised_steps
            ramp = min(1.0, ss_step / max(1, args.ramp_steps))
            loss = (
                args.w_attach * l_attach
                + ramp * args.w_conformal * l_conformal
                + args.w_res_reg * l_res_reg
            )
            if args.w_z_coherence > 0 and dataset.Z >= 2:
                loss = loss + args.w_z_coherence * l_zcoh
            if args.w_winding_band > 0:
                loss = loss + args.w_winding_band * l_band
            if args.w_radial_cap > 0:
                loss = loss + args.w_radial_cap * l_radial_cap
            if args.w_winding_cc > 0:
                loss = loss + args.w_winding_cc * l_wcc
            if args.w_arc_speed > 0:
                loss = loss + args.w_arc_speed * l_speed
            if args.w_intensity > 0:
                loss = loss + args.w_intensity * l_intensity
            if args.w_intensity_zcoh > 0 and dataset.Z >= 2:
                loss = loss + args.w_intensity_zcoh * l_int_zcoh
            if args.w_strip_quality > 0:
                loss = loss + args.w_strip_quality * l_strip_q
            if discriminator is not None and args.w_discriminator > 0:
                loss = loss + args.w_discriminator * l_disc
            # Step A — pose-tether: residual at anchor points should stay
            # close to its initial value. Replaces the MSE-supervised anchor
            # that made R1PF1ws work on synthetic; required when no GT exists
            # to prevent the patch-feature trivial-collapse mode on real.
            l_tether_val = 0.0
            if args.w_pose_tether > 0 and anchor_uv is not None:
                # Sample subset of anchor points for this step.
                n_sub = min(args.pose_tether_batch, anchor_uv.shape[0])
                idx_sub = torch.randint(
                    0, anchor_uv.shape[0], (n_sub,), device=device
                )
                sub_uv = anchor_uv[idx_sub]
                sub_init = anchor_res_init[idx_sub]
                sub_res_now = model(sub_uv)
                l_tether = ((sub_res_now - sub_init) ** 2).sum(dim=1).mean()
                l_tether_val = float(l_tether.item())
                loss = loss + args.w_pose_tether * l_tether

            # Track 4: DCT patch-feature z-coherence (fixed feature extractor;
            # not gameable like the trainable discriminator).
            l_patch_feat_val = 0.0
            if args.w_patch_feature_zcoh > 0 and dataset.Z >= 2:
                l_patch_feat = compute_patch_feature_zcoh(
                    dataset, model,
                    n_patches=args.patch_feature_n_patches,
                    patch_w=args.patch_feature_w,
                    n_dct_keep=args.patch_feature_n_dct,
                    target_norm=args.patch_feature_target_norm,
                )
                loss = loss + args.w_patch_feature_zcoh * l_patch_feat
                l_patch_feat_val = float(l_patch_feat.item())

            # Step B: InfoNCE contrastive patch loss. Forbids trivial collapse
            # by construction — if all (u, z) map to one patch, batch features
            # converge and InfoNCE explodes to log(n_anchors).
            l_infonce_val = 0.0
            if args.w_patch_feature_infonce > 0 and dataset.Z >= 2:
                l_infonce = compute_patch_feature_infonce(
                    dataset, model,
                    n_anchors=args.infonce_n_anchors,
                    patch_w=args.patch_feature_w,
                    n_dct_keep=args.patch_feature_n_dct,
                    temperature=args.infonce_temperature,
                )
                loss = loss + args.w_patch_feature_infonce * l_infonce
                l_infonce_val = float(l_infonce.item())

            # Z1 (MAPS): soft pseudo-supervision from raycast per-angle
            # centerlines. The centerlines give per-(winding, angle) → (x, y)
            # mapping; we compute the matching arc-length u and supervise
            # pred_xy(u, z) ≈ centerline_xy. Synth geometry is z-invariant so
            # the same (u, xy) labels apply at every z. Low weight: soft
            # tie-breaker among the many on-emulsion mappings that satisfy
            # attachment+conformal.
            l_maps_val = 0.0
            if args.w_maps > 0:
                maps_batch = dataset.sample_maps_labels(args.maps_batch_size)
                if maps_batch is not None:
                    uv_m = maps_batch["uv_norm"]
                    u_m = maps_batch["u_raw"]
                    z_m = maps_batch["z_idx"]
                    xy_t = maps_batch["xy_target"]
                    base_m = dataset.analytical_xy_norm(u_m)
                    res_m = model(uv_m)
                    pred_m = base_m + res_m
                    l_maps = ((pred_m - xy_t) ** 2).mean()
                    loss = loss + args.w_maps * l_maps
                    l_maps_val = float(l_maps.item())

            # A3: joint autodecoder render loss. Strip g_φ(u,z) → s,
            # intensity_pred = base + s·(max - base), compared to actual
            # CT at pred_xy. Mask-weighted: off-emulsion contributes 0 so
            # attachment continues to drive the geometry. Ramped from 0 to
            # full weight over args.autodecoder_ramp_steps starting at
            # args.autodecoder_start_step, so f_θ first reaches the manifold
            # before the render loss adds pressure.
            l_render_val = 0.0
            if strip_model is not None:
                ramp = max(0.0, min(
                    1.0,
                    (step - args.autodecoder_start_step) / max(
                        1, args.autodecoder_ramp_steps
                    ),
                ))
                if ramp > 0:
                    # Sample strip content at the same (u, z) anchors used
                    # by attachment. uv is already in normalized coords.
                    s_pred = torch.sigmoid(strip_model(uv).squeeze(-1))  # (B,)
                    intensity_pred = strip_base + s_pred * (strip_max - strip_base)
                    intensity_gt = dataset.sample_image(pred_xy, z_idx)
                    # Mask-weight to gate off-emulsion samples (detached so
                    # we don't try to push pred_xy onto emulsion via the
                    # render loss — that's attachment's job).
                    M = dataset.sample_mask(pred_xy, z_idx).detach()
                    M_sum = M.sum().clamp(min=1.0)
                    l_render = (((intensity_pred - intensity_gt) ** 2) * M).sum() / M_sum
                    # Optional strip-smoothness TV (penalize ∂s/∂u, ∂s/∂z).
                    if args.w_strip_tv > 0:
                        s_grad = torch.autograd.grad(
                            s_pred.sum(), uv,
                            create_graph=True, retain_graph=True,
                        )[0]
                        l_strip_tv = (s_grad ** 2).mean()
                        l_render = l_render + args.w_strip_tv * l_strip_tv
                    loss = loss + args.w_autodecoder * ramp * l_render
                    l_render_val = float(l_render.item())

            # A1: frame-period periodicity on the implied strip. Pairs
            # patches at (u, z) and (u + frame_period_u, z); same shape as
            # the DCT z-coh loss but with a PHYSICAL period along u rather
            # than data-driven z-adjacency.
            l_frame_period_val = 0.0
            if args.w_frame_periodicity > 0:
                l_frame_period = compute_frame_periodicity_loss(
                    dataset, model,
                    n_patches=args.patch_feature_n_patches,
                    patch_w=args.patch_feature_w,
                    n_dct_keep=args.patch_feature_n_dct,
                    frame_period_u=args.frame_period_u,
                    target_norm=args.patch_feature_target_norm,
                )
                loss = loss + args.w_frame_periodicity * l_frame_period
                l_frame_period_val = float(l_frame_period.item())
            if args.learn_eccentricity:
                # Weak L2 prior on amplitude (keeps fit from absorbing jitter).
                loss = loss + args.w_ecc_prior * dataset.ecc_amp ** 2
            if args.per_winding_eccentricity:
                # L2 prior on per-winding amplitudes; small to allow real jitter.
                loss = loss + args.w_ecc_prior * (dataset.pw_amp ** 2).mean()
            if args.per_winding_theta_phase:
                # A4: L2 prior on θ_offset[k] to keep phase shifts modest.
                loss = loss + args.w_theta_phase_prior * (
                    dataset.theta_offset ** 2
                ).mean()
            l_mse = torch.tensor(0.0, device=device)
            l_conformal_val = float(l_conformal.item())
            l_res_reg_val = float(l_res_reg.item())
            # l_zcoh_val already set above (0.0 if disabled)

        optimizer.zero_grad()
        if ecc_optimizer is not None:
            ecc_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # A5: two-stage refine→freeze→residual.
        #   stage 1 (step < two_stage_base_steps): base params only, INR frozen.
        #   stage 2 (step ≥ two_stage_base_steps): INR only, base frozen.
        # When two_stage_base_steps == 0 (default) both step every iter (joint).
        in_stage1 = (args.two_stage_base_steps > 0
                     and step < args.two_stage_base_steps)
        in_stage2 = (args.two_stage_base_steps > 0
                     and step >= args.two_stage_base_steps)
        if not in_stage1:
            optimizer.step()
        if ecc_optimizer is not None and not in_stage2:
            ecc_optimizer.step()
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
                "l_zcoh": l_zcoh_val if not supervised_phase else 0.0,
                "l_band": l_band_val,
                "l_wcc": l_wcc_val,
                "l_speed": l_speed_val,
                "l_intensity": l_intensity_val,
                "l_int_zcoh": l_int_zcoh_val,
                "l_strip_q": l_strip_q_val,
                "l_disc": l_disc_val,
                "l_patch_feat": l_patch_feat_val if not supervised_phase else 0.0,
                "l_infonce": l_infonce_val if not supervised_phase else 0.0,
                "l_tether": l_tether_val if not supervised_phase else 0.0,
                "E": E_mean, "G": G_mean, "F": F_mean,
                "res_px": res_px,
                "lr": optimizer.param_groups[0]["lr"],
            }
            if args.learn_eccentricity:
                entry["ecc_amp"] = float(dataset.ecc_amp.item())
                entry["ecc_phase"] = float(dataset.ecc_phase.item())
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()
            elapsed = time.time() - t0
            tag = "SUP" if supervised_phase else "SS "
            print(f"  [{tag}] step {step:>6d}  loss={loss.item():.5f}  "
                  f"mse={l_mse.item():.5f}  attach={l_attach.item():.4f}  "
                  f"conf={l_conformal_val:.4f}  reg={l_res_reg_val:.5f}  "
                  f"zcoh={l_zcoh_val:.5f}  band={l_band_val:.4f}  "
                  f"wcc={l_wcc_val:.4f}  speed={l_speed_val:.4f}  "
                  f"intens={l_intensity_val:.4f}  "
                  f"izcoh={l_int_zcoh_val:.4f}  "
                  f"stripQ={l_strip_q_val:+.4f}  "
                  f"disc={l_disc_val:+.4f}  "
                  f"E={E_mean:.3f} G={G_mean:.3f} |F|={F_mean:.3f}  "
                  f"res_px={res_px:.2f}  ({elapsed:.0f}s)")

        if args.ckpt_every and step % args.ckpt_every == 0:
            torch.save(
                {"model": model.state_dict(), "step": step, "args": vars(args)},
                os.path.join(args.out_dir, f"ckpt_step{step}.pt"),
            )

    log_file.close()
    ckpt = {"model": model.state_dict(), "step": args.steps, "args": vars(args)}
    if strip_model is not None:
        ckpt["strip_model"] = strip_model.state_dict()
        ckpt["strip_base"] = strip_base
        ckpt["strip_max"] = strip_max
    if args.learn_eccentricity:
        ckpt["ecc_amp"] = float(dataset.ecc_amp.item())
        ckpt["ecc_phase"] = float(dataset.ecc_phase.item())
        print(f"  Learned eccentricity: amp={ckpt['ecc_amp']:+.3f}px, "
              f"phase={ckpt['ecc_phase']:+.3f} rad")
    if args.per_winding_eccentricity:
        ckpt["pw_amp"] = dataset.pw_amp.detach().cpu().tolist()
        ckpt["pw_phase"] = dataset.pw_phase.detach().cpu().tolist()
        print(f"  Per-winding amp range: "
              f"[{dataset.pw_amp.min().item():+.2f}, "
              f"{dataset.pw_amp.max().item():+.2f}]px, "
              f"|amp|_mean={dataset.pw_amp.abs().mean().item():.2f}px")
    if args.per_winding_theta_phase:
        ckpt["theta_offset"] = dataset.theta_offset.detach().cpu().tolist()
        to = dataset.theta_offset.detach()
        print(f"  A4 θ_offset range: "
              f"[{to.min().item():+.4f}, {to.max().item():+.4f}] rad, "
              f"|θ|_mean={to.abs().mean().item():.4f} rad")
    torch.save(ckpt, os.path.join(args.out_dir, "model_final.pt"))

    # Save dataset geometry (needed by eval for analytical_xy_norm)
    geometry = {
        "cx": dataset.cx, "cy": dataset.cy,
        "n_layers": dataset.n_layers,
        "theta_seam": dataset.theta_seam,
        "boundaries": dataset.boundaries.cpu().tolist(),
        "H": dataset.H, "W": dataset.W, "Z": dataset.Z,
        "z_indices": dataset.z_indices.tolist(),
        "ecc_amp": float(dataset.ecc_amp.item()) if args.learn_eccentricity else 0.0,
        "ecc_phase": float(dataset.ecc_phase.item()) if args.learn_eccentricity else 0.0,
        "theta_offset": (
            dataset.theta_offset.detach().cpu().tolist()
            if args.per_winding_theta_phase else []
        ),
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
    parser.add_argument("--w-strip-mse", type=float, default=0.0,
                        help="R3 (loss-side): MSE between CT-sampled intensity "
                             "at pred_xy vs at xy_target during R1S. Targets "
                             "SSIM directly by weighting coord errors by local "
                             "CT gradient. 0 = off (legacy R1S coord-MSE only). "
                             "Try 1.0 / 5.0 / 10.0 to start; CT-intensity range "
                             "is ≈[0.12, 0.85] so a per-sample intens-MSE term "
                             "is O(1e-2) — needs a non-trivial weight to matter.")
    parser.add_argument("--w-res-reg", type=float, default=0.0,
                        help="L2 penalty on xy_res magnitude (anchors residual "
                             "near zero; prevents drift when attachment gradient "
                             "is flat off-target).")
    parser.add_argument("--w-z-coherence", type=float, default=0.0,
                        help="Weight on |pred_xy(u,z)-pred_xy(u,z+1)|² penalty. "
                             "Synthetic geometry is z-invariant so the surface "
                             "should be too — breaks the underdetermined-attachment "
                             "degeneracy on imperfect_4k. 0 disables.")
    parser.add_argument("--z-coh-batch-size", type=int, default=32768,
                        help="Batch size for z-coherence sampling (separate from "
                             "main attachment batch to keep memory bounded).")
    parser.add_argument("--learn-eccentricity", action="store_true",
                        help="Make r_analytical += amp·cos(theta − phase) a "
                             "learnable 2-scalar correction. Directly absorbs "
                             "the ecc term the synthetic generator adds.")
    parser.add_argument("--per-winding-eccentricity", action="store_true",
                        help="Per-winding (amp_k, phase_k): r_k += "
                             "amp_k·cos(theta − phase_k). 2·n_layers params, "
                             "trained by the dedicated base-params optimizer. "
                             "Composes with --learn-eccentricity and "
                             "--per-winding-theta-phase.")
    parser.add_argument("--per-winding-theta-phase", action="store_true",
                        help="A4: per-winding angular phase θ_offset[k] added "
                             "to θ in the analytical base. n_layers scalars, "
                             "linearly interpolated between adjacent windings. "
                             "Trained by the base-params optimizer. Targets "
                             "global+per-winding angular misalignment "
                             "(median_shift > 0 on SS imperfect) that the INR "
                             "residual learns slowly via σ=20 Fourier features. "
                             "L2-regularize via --w-theta-phase-prior.")
    parser.add_argument("--w-theta-phase-prior", type=float, default=1e-4,
                        help="L2 prior on θ_offset to discourage runaway "
                             "phase shifts; small to allow real per-winding "
                             "phase errors (expected |θ_offset| < 0.05 rad).")
    parser.add_argument("--two-stage-base-steps", type=int, default=0,
                        help="A5: coordinate descent. For the first N steps "
                             "step ONLY the analytical-base optimizer (ecc / "
                             "per-winding ecc / θ_offset), keeping the INR "
                             "head frozen at zero-init. After step N, freeze "
                             "base params and step ONLY the INR optimizer. "
                             "0 (default) = joint optimization, current "
                             "behavior. Requires at least one base-param "
                             "flag (else stage 1 is a no-op).")
    parser.add_argument("--w-winding-band", type=float, default=0.0,
                        help="Soft-hinge penalty: pred_xy radial position "
                             "should fall in the boundary-band of the expected "
                             "winding (=floor(u_raw)). Breaks the radial "
                             "degeneracy that lets attachment hop windings.")
    parser.add_argument("--w-radial-cap", type=float, default=0.0,
                        help="B3: soft cap on |r_pred − r_analytical| at half "
                             "the local layer_spacing. Quadratic outside the "
                             "cap, zero inside — does not fight small SS "
                             "corrections. Different from --w-winding-band: "
                             "this is relative to the (emulsion-offset) "
                             "analytical base, not to film centerlines.")
    parser.add_argument("--w-winding-cc", type=float, default=0.0,
                        help="CC-based winding-label penalty: builds a "
                             "per-pixel winding-number volume from connected "
                             "components on the eroded film mask, then "
                             "penalizes |sampled_label - floor(u_raw)|². "
                             "Topology-aware variant of --w-winding-band.")
    parser.add_argument("--cc-erode-iters", type=int, default=2,
                        help="Erosion iterations before CC labeling — "
                             "needs to break pinch points without destroying "
                             "thin emulsion arcs.")
    parser.add_argument("--w-arc-speed", type=float, default=0.0,
                        help="Variance penalty on |∂pred_xy/∂u|² — encourages "
                             "uniform pixel-per-u progression along the spiral. "
                             "0 disables.")
    parser.add_argument("--w-ecc-prior", type=float, default=1e-5,
                        help="L2 prior weight on eccentricity amplitude.")
    parser.add_argument("--ecc-lr-mult", type=float, default=100.0,
                        help="Multiplier on base LR for ecc params "
                             "(via a dedicated, unclipped optimizer).")
    parser.add_argument("--use-distance-attach", action="store_true",
                        help="Replace (1-mask)² attachment with a smooth "
                             "distance-transform loss (d/dist_scale)², where "
                             "d is the Euclidean distance to the target mask.")
    parser.add_argument("--w-intensity", type=float, default=0.0,
                        help="Intensity-profile attachment weight (Path B). "
                             "Penalizes off-surface neighbors brighter than "
                             "on-surface pixel. Provides non-zero gradient "
                             "inside emulsion band where mask attachment is "
                             "flat. 0 disables.")
    parser.add_argument("--intensity-delta-px", type=float, default=5.0,
                        help="Normal offset δ in pixels for intensity loss.")
    parser.add_argument("--intensity-margin", type=float, default=0.05,
                        help="Margin in normalized intensity for the relu.")
    parser.add_argument("--w-autodecoder", type=float, default=0.0,
                        help="A3: joint autodecoder weight. Co-learns a strip "
                             "INR g_φ(u,z)→s and uses MSE(base+s·(max-base), "
                             "CT_at_pred_xy) as a render loss, mask-gated to "
                             "on-emulsion samples. Forces pred_xy to *explain* "
                             "actual CT intensities rather than just sitting "
                             "on the emulsion. Recommended: 1.0 with ramp.")
    parser.add_argument("--autodecoder-start-step", type=int, default=500,
                        help="Step at which the render-loss ramp begins. "
                             "Before this step, render loss is 0 so f_θ can "
                             "first reach the on-emulsion manifold without "
                             "fighting an immature strip.")
    parser.add_argument("--autodecoder-ramp-steps", type=int, default=1000,
                        help="Linear ramp length (in steps) from 0 to full "
                             "w-autodecoder weight after the start step.")
    parser.add_argument("--w-strip-tv", type=float, default=0.0,
                        help="Optional TV regularizer on s(u, z) — penalizes "
                             "|∂s/∂u|² + |∂s/∂z|² to keep the learned strip "
                             "smooth. 0 = off.")
    parser.add_argument("--strip-n-fourier", type=int, default=128,
                        help="Strip INR Fourier feature count (smaller than "
                             "f_θ; strip is single-channel + smooth).")
    parser.add_argument("--strip-sigma", type=float, default=20.0,
                        help="Strip INR Fourier σ — matches f_θ default 20 "
                             "so the strip can represent ~28-winding content.")
    parser.add_argument("--strip-hidden-dim", type=int, default=128,
                        help="Strip INR hidden dim (small: limits capacity "
                             "so g_φ can't memorize arbitrary CT for any f_θ).")
    parser.add_argument("--strip-n-layers", type=int, default=2,
                        help="Strip INR MLP depth.")
    parser.add_argument("--w-maps", type=float, default=0.0,
                        help="Z1: weight for medial-axis pseudo-supervision "
                             "(MAPS). Soft MSE against per-(winding, angle) "
                             "centerlines from the raycast detector. Gives "
                             "the model an angular signal at every (u, z) "
                             "that pure SS attachment+conformal lacks. "
                             "Recommended: 0.01-1.0 (start at 0.1).")
    parser.add_argument("--maps-batch-size", type=int, default=8192,
                        help="Batch size for MAPS labels (smaller than main "
                             "batch since labels are denser per loss unit).")
    parser.add_argument("--label-noise-px", type=float, default=0.0,
                        help="Z3: σ (in pixels) of Gaussian noise added to "
                             "GT xy_target during supervised training. "
                             "Diagnostic: measures the noise budget the "
                             "supervised path can tolerate, which bounds how "
                             "noisy a pseudo-supervision signal (e.g. MAPS) "
                             "can be before it stops helping. 0 = clean GT "
                             "(default = identical to current R1S).")
    parser.add_argument("--mask-weight-int-zcoh", action="store_true",
                        help="Z2: gate intensity z-coh by sample_mask product "
                             "so off-emulsion pairs contribute 0. Without this "
                             "gate the loss is closed (R1B4/B5: trivial "
                             "collapse to uniform film-base). With it the loss "
                             "only constrains the mapping where it's actually "
                             "on the emulsion, attachment dominates the off-"
                             "manifold drift mode. Used with --w-intensity-zcoh.")
    parser.add_argument("--w-intensity-zcoh", type=float, default=0.0,
                        help="Intensity z-coherence weight (Path B v2). "
                             "Penalizes |I(pred(u,z_a)) - I(pred(u,z_b))|² "
                             "across adjacent z. Targets imperfect_4k failure "
                             "mode where surface lands on emulsion but at "
                             "wrong angular position. 0 disables.")
    parser.add_argument("--w-strip-quality", type=float, default=0.0,
                        help="Strip-quality loss (Path C). Renders a partial "
                             "strip on a dense u-grid at one z-slice and "
                             "maximizes (-variance) + hf_weight·(-hf_energy). "
                             "Differentiable proxies of strip_score metrics. "
                             "0 disables.")
    parser.add_argument("--strip-q-cols", type=int, default=4096,
                        help="Number of u-columns sampled for strip-quality.")
    parser.add_argument("--strip-q-hf-weight", type=float, default=1.0,
                        help="Relative weight of HF-energy vs variance.")
    parser.add_argument("--discriminator-ckpt", type=str, default=None,
                        help="Path to strip_discriminator.pt — when set, "
                             "renders strip patches at pred_xy and uses the "
                             "frozen CNN as a learned similarity loss "
                             "(Path D: Vesuvius-style ink-detector).")
    parser.add_argument("--w-discriminator", type=float, default=0.0,
                        help="Weight of the discriminator loss.")
    parser.add_argument("--w-pose-tether", type=float, default=0.0,
                        help="Step A: anchor pose-tether weight. At step 0, "
                             "snapshot the model residual at a fixed set of "
                             "(u, z) anchor points; penalize drift from that "
                             "snapshot during training. Replaces the MSE "
                             "anchor that made synthetic R1PF1ws work on "
                             "real data where no GT exists. 0 disables.")
    parser.add_argument("--pose-tether-anchors", type=int, default=65536,
                        help="Number of fixed anchor (u, z) points whose "
                             "initial residual is snapshotted.")
    parser.add_argument("--pose-tether-batch", type=int, default=16384,
                        help="Number of anchor points re-evaluated per step.")
    parser.add_argument("--w-patch-feature-zcoh", type=float, default=0.0,
                        help="Track 4: DCT patch-feature z-coherence weight. "
                             "Renders strip patches at pred_xy(u, z_a) and "
                             "pred_xy(u, z_b=z_a+1), takes per-patch DCT "
                             "(drop DC, keep K coeffs), penalizes 1 - "
                             "cos_sim(feat_a, feat_b) + magnitude floor. "
                             "Fixed feature extractor (not gameable). 0 disables.")
    parser.add_argument("--patch-feature-n-patches", type=int, default=64,
                        help="Number of (u_start, z_a) patch pairs per step.")
    parser.add_argument("--patch-feature-w", type=int, default=64,
                        help="Patch width in strip-pixels (= number of u "
                             "samples per patch).")
    parser.add_argument("--patch-feature-n-dct", type=int, default=16,
                        help="Number of mid-frequency DCT coefficients kept "
                             "(after dropping DC).")
    parser.add_argument("--w-patch-feature-infonce", type=float, default=0.0,
                        help="Step B: InfoNCE contrastive patch-feature "
                             "weight. Trivial collapse forbidden by "
                             "construction (if all (u,z) → one patch, "
                             "positives = negatives, InfoNCE → log(N)). "
                             "0 disables.")
    parser.add_argument("--infonce-n-anchors", type=int, default=64,
                        help="Batch size for InfoNCE patch-feature loss.")
    parser.add_argument("--infonce-temperature", type=float, default=0.1,
                        help="Softmax temperature for InfoNCE. Lower = "
                             "harder contrast.")
    parser.add_argument("--w-frame-periodicity", type=float, default=0.0,
                        help="A1: frame-period periodicity loss on the "
                             "implied strip. Pairs patches at u and "
                             "u + frame_period_u along the same z; cosine-"
                             "distance + magnitude floor (same shape as DCT "
                             "z-coh but with a physical u-period rather than "
                             "data-driven z-adjacency).")
    parser.add_argument("--frame-period-u", type=float, default=0.0665,
                        help="Frame pitch in u-units (winding fractions). "
                             "For HQ video presets (n_z=256, frame_w=341 px, "
                             "total_arc=143558, n_layers=28): "
                             "frame_period_u = 341/143558*28 ≈ 0.0665. "
                             "Compute from the generator, do not fit.")
    parser.add_argument("--patch-feature-target-norm", type=float, default=0.05,
                        help="L2-norm floor on feature vectors. Patches with "
                             "weaker content than this are penalized to "
                             "prevent collapse to uniform regions.")
    parser.add_argument("--disc-n-patches", type=int, default=32,
                        help="Number of patches per training step for "
                             "discriminator loss. Patch dims are (n_z, "
                             "patch_w) where patch_w comes from the ckpt.")

    parser.add_argument("--n-fourier", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers-mlp", type=int, default=4)
    parser.add_argument("--bspline-u-per-winding", type=int, default=4,
                        help="Cubic B-spline u-spans per winding (Track 3). "
                             "Default 4 = enough for 3-component sinusoidal "
                             "jitter with per-winding phase. Total u-spans = "
                             "this × n_windings; control points = +3.")
    parser.add_argument("--bspline-n-z", type=int, default=3,
                        help="Cubic B-spline z-spans. Default 3 = mostly "
                             "z-invariant deformation (synthetic geometry "
                             "is z-constant).")
    parser.add_argument("--model-type",
                        choices=["inr", "perwinding", "bspline",
                                 "grid", "hash", "wire", "finer"],
                        default="inr",
                        help="Residual model architecture. inr (default): "
                             "Fourier features + MLP. perwinding/bspline: "
                             "structured parametrics. grid: G1 dense multi-"
                             "resolution 2D feature grid. hash: H1 Instant-"
                             "NGP hash encoding (pure PyTorch). wire: S2 "
                             "WIRE Gabor wavelet MLP. finer: S3 FINER "
                             "variable-periodic MLP.")
    parser.add_argument("--n-harmonics", type=int, default=10,
                        help="Harmonics per winding for --model-type perwinding.")

    # Encoder/MLP LR split, used by grid/hash models.
    parser.add_argument("--encoder-lr-mult", type=float, default=100.0,
                        help="Multiplier on --lr for the explicit feature "
                             "encoder (grid planes or hash tables). Standard "
                             "100× for grid-based INRs; the MLP head keeps "
                             "--lr. Ignored for non-grid/-hash models.")

    # G1 dense feature grid (--model-type grid).
    parser.add_argument("--grid-n-levels", type=int, default=8)
    parser.add_argument("--grid-features", type=int, default=2,
                        help="Features per level (G1).")
    parser.add_argument("--grid-base-u", type=int, default=64)
    parser.add_argument("--grid-base-z", type=int, default=16)
    parser.add_argument("--grid-finest-u", type=int, default=8192,
                        help="Finest u-resolution (G1). Set high — u-axis "
                             "is ≈143k px effective on HQ presets.")
    parser.add_argument("--grid-finest-z", type=int, default=256)
    parser.add_argument("--grid-hidden", type=int, default=64)
    parser.add_argument("--grid-mlp-layers", type=int, default=2)

    # H1 hash encoding (--model-type hash).
    parser.add_argument("--hash-n-levels", type=int, default=16)
    parser.add_argument("--hash-features", type=int, default=2)
    parser.add_argument("--hash-log2-size", type=int, default=19,
                        help="log2(T): table size = 2**this per level.")
    parser.add_argument("--hash-base-res", type=int, default=16)
    parser.add_argument("--hash-finest-res", type=int, default=8192,
                        help="Finest grid resolution (H1).")
    parser.add_argument("--hash-hidden", type=int, default=64)
    parser.add_argument("--hash-mlp-layers", type=int, default=2)

    # S2 WIRE (--model-type wire).
    parser.add_argument("--wire-omega", type=float, default=20.0,
                        help="Gabor sine frequency for WIRE.")
    parser.add_argument("--wire-sigma", type=float, default=10.0,
                        help="Gabor Gaussian width for WIRE.")

    # S3 FINER (--model-type finer).
    parser.add_argument("--finer-omega", type=float, default=30.0,
                        help="SIREN-style omega_0 for FINER.")
    parser.add_argument("--finer-bias-k", type=float, default=5.0,
                        help="FINER bias-init range U(-k, k). k>1 unlocks "
                             "higher frequencies per the CVPR 2024 paper.")

    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=0,
                        help="Save checkpoint every N steps; 0 disables intermediate saves.")

    # Round 1: attachment target × supervision variants.
    parser.add_argument("--attachment", choices=["film", "emulsion", "centerline"],
                        default="film",
                        help="Which binary mask drives the attachment loss.")
    parser.add_argument("--centerline-erode", type=int, default=1,
                        help="Erosion iterations when --attachment=centerline.")
    parser.add_argument("--winding-detector",
                        choices=["raycast", "histogram"], default="raycast",
                        help="Winding-boundary detector. raycast (default) "
                             "uses multi-angle emulsion-run counting, robust "
                             "to pinch points and touching windings. "
                             "histogram is the legacy circumference-normalized "
                             "radial-valley detector.")
    parser.add_argument("--supervised-steps", type=int, default=0,
                        help="If > 0, first N steps train with MSE vs GT "
                             "u_map (requires --gt-npz), then switch to "
                             "self-supervised attachment+conformal.")
    parser.add_argument("--data-driven-base", action="store_true",
                        help="Track 1: use per-angle centerline radii from "
                             "the raycast detector as the analytical base, "
                             "instead of concentric circles. Each winding "
                             "becomes its own non-circular curve, absorbing "
                             "eccentricity and a fraction of jitter into "
                             "the base. Requires --winding-detector raycast.")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Track 2: path to a model_final.pt checkpoint "
                             "to initialize the residual model weights from "
                             "before training starts. Optimizer state is "
                             "ignored (fresh Adam). Useful for curriculum "
                             "(clean → imperfect).")
    parser.add_argument("--gt-npz", type=str, default=None,
                        help="Path to synthetic ground_truth.npz. "
                             "Required when --supervised-steps > 0.")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
