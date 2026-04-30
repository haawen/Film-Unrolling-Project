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
from unwrapping.inr.strip_discriminator import load_discriminator


def build_model(model_type="inr", n_fourier=256, sigma=10.0, hidden_dim=256,
                n_layers=4, n_windings=1, n_harmonics=10, device="cuda"):
    """Residual model: zero-init so step-0 prediction == analytical_xy.

    model_type: "inr"        — Fourier features + MLP (DeformationINR)
                "perwinding" — per-winding Fourier-in-θ (PerWindingParametric)
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
        # Zero-init handled in __init__.
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
        device=device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model_type} (2 → 2), {n_params:,} params, "
          f"zero-init for residual start")

    # R2b: learnable per-theta radial correction (amp·cos(theta - phase)).
    # Directly targets the ecc term the synthetic generator adds.
    if args.learn_eccentricity:
        ecc_params = dataset.enable_learnable_eccentricity()
        ecc_lr = args.lr * args.ecc_lr_mult
        ecc_optimizer = torch.optim.Adam(ecc_params, lr=ecc_lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f"Learnable eccentricity enabled. ecc_lr={ecc_lr:.2e} "
              f"({args.ecc_lr_mult}× INR LR, separate optimizer, unclipped).")
    elif args.per_winding_eccentricity:
        # R3 lever 2: per-winding (amp_k, phase_k) targets the synthetic jitter
        # that a single global (amp, phase) cannot fit. Same separate-optimizer
        # trick as global ecc.
        pw_params = dataset.enable_per_winding_eccentricity()
        ecc_lr = args.lr * args.ecc_lr_mult
        ecc_optimizer = torch.optim.Adam(pw_params, lr=ecc_lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print(f"Per-winding eccentricity enabled "
              f"({dataset.n_layers} windings × 2 scalars). "
              f"ecc_lr={ecc_lr:.2e} ({args.ecc_lr_mult}× INR LR).")
    else:
        ecc_optimizer = None
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

            xy_base = dataset.analytical_xy_norm(u_raw)
            xy_res = model(uv)
            pred_xy = xy_base + xy_res

            l_mse = ((pred_xy - xy_target) ** 2).mean()
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
            l_wcc_val = 0.0
            l_speed_val = 0.0
            l_intensity_val = 0.0
            l_int_zcoh_val = 0.0
            l_strip_q_val = 0.0
            l_disc_val = 0.0
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
            if args.learn_eccentricity:
                # Weak L2 prior on amplitude (keeps fit from absorbing jitter).
                loss = loss + args.w_ecc_prior * dataset.ecc_amp ** 2
            if args.per_winding_eccentricity:
                # L2 prior on per-winding amplitudes; small to allow real jitter.
                loss = loss + args.w_ecc_prior * (dataset.pw_amp ** 2).mean()
            l_mse = torch.tensor(0.0, device=device)
            l_conformal_val = float(l_conformal.item())
            l_res_reg_val = float(l_res_reg.item())
            # l_zcoh_val already set above (0.0 if disabled)

        optimizer.zero_grad()
        if ecc_optimizer is not None:
            ecc_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if ecc_optimizer is not None:
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
                             "trained by the same dedicated optimizer as "
                             "--learn-eccentricity. Mutually exclusive.")
    parser.add_argument("--w-winding-band", type=float, default=0.0,
                        help="Soft-hinge penalty: pred_xy radial position "
                             "should fall in the boundary-band of the expected "
                             "winding (=floor(u_raw)). Breaks the radial "
                             "degeneracy that lets attachment hop windings.")
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
    parser.add_argument("--disc-n-patches", type=int, default=32,
                        help="Number of patches per training step for "
                             "discriminator loss. Patch dims are (n_z, "
                             "patch_w) where patch_w comes from the ckpt.")

    parser.add_argument("--n-fourier", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers-mlp", type=int, default=4)
    parser.add_argument("--model-type", choices=["inr", "perwinding"],
                        default="inr",
                        help="Residual model architecture. inr (default): "
                             "Fourier features + MLP with shared weights "
                             "across windings. perwinding: independent "
                             "Fourier-in-θ series per winding (no cross-"
                             "winding weight sharing).")
    parser.add_argument("--n-harmonics", type=int, default=10,
                        help="Harmonics per winding for --model-type perwinding.")

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
    parser.add_argument("--gt-npz", type=str, default=None,
                        help="Path to synthetic ground_truth.npz. "
                             "Required when --supervised-steps > 0.")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
