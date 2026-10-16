"""
Render the unwrapped film strip from a trained residual inverse-mapping INR,
pred_xy(u, z) = analytical_xy(u) + INR(u, z), and score it against the synthetic
ground truth (--gt-npz) and the oracle strip (--oracle-npz).

Also saves diagnostics:
  - strip_full.png        : full unwrapped strip
  - detail_layers_*.png   : 5-layer crops
  - surface_overlay.png   : predicted (x,y) path overlaid on a CT slice
  - speed_map.png         : ||∂f/∂u|| over the (u, z) grid
  - conformal_map.png     : |E - G| + 2|F| over the (u, z) grid
  - residual_map.png      : ||INR(u,z)|| over the (u, z) grid
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from virtual_unrolling.learned_unwrapping.surface_data import SurfaceDataset
from virtual_unrolling.learned_unwrapping.unwrap_model import DeformationINR
from virtual_unrolling.learned_unwrapping.grid_model import GridDeformation


def load_model(ckpt_path, device, n_windings_fallback=None):
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt["args"]
    mt = a.get("model_type", "inr")
    if mt == "inr":
        model = DeformationINR(
            n_fourier=a.get("n_fourier", 256),
            sigma=a.get("sigma", 10.0),
            hidden_dim=a.get("hidden_dim", 256),
            n_layers=a.get("n_layers_mlp", 4),
            input_dim=2, output_dim=2,
        ).to(device)
    elif mt == "grid":
        model = GridDeformation(
            n_levels=a.get("grid_n_levels", 8),
            n_features_per_level=a.get("grid_features", 2),
            base_resolution_u=a.get("grid_base_u", 64),
            base_resolution_z=a.get("grid_base_z", 16),
            finest_resolution_u=a.get("grid_finest_u", 8192),
            finest_resolution_z=a.get("grid_finest_z", 256),
            hidden_dim=a.get("grid_hidden", 64),
            n_layers=a.get("grid_mlp_layers", 2),
            output_dim=2,
        ).to(device)
        # Back-compat: old checkpoints stored grids as (1, C, Z, U) for the
        # original F.grid_sample path. The current model expects (C, Z, U).
        sd = ckpt["model"]
        for k, v in list(sd.items()):
            if k.startswith("grids.") and v.dim() == 4 and v.shape[0] == 1:
                sd[k] = v.squeeze(0)
    else:
        raise ValueError(f"Unknown model_type in ckpt: {mt}")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def build_uv_grid(dataset, n_cols, n_rows, device):
    """(n_rows * n_cols, 2) grid covering (u, z) in normalized coords [-1, 1].

    Also returns u_raw (float, [0, n_layers)) and z_idx (int, [0, Z)) for
    matched analytical spiral & mask lookup.
    """
    u_raw = torch.linspace(0.0, dataset.n_layers - 1e-4, n_cols, device=device)
    z_idx = torch.linspace(0, dataset.Z - 1, n_rows, device=device).round().long()

    # Outer product via meshgrid
    uu, zz = torch.meshgrid(u_raw, z_idx.float(), indexing="xy")   # (n_rows, n_cols)
    u_flat = uu.flatten()
    z_flat = zz.flatten().long()

    u_norm = u_flat / dataset.n_layers * 2.0 - 1.0
    if dataset.Z > 1:
        z_norm = z_flat.float() / (dataset.Z - 1) * 2.0 - 1.0
    else:
        z_norm = torch.zeros_like(u_norm)
    uv_norm = torch.stack([u_norm, z_norm], dim=1)
    return uv_norm, u_flat, z_flat


def render_strip(dataset, model, n_cols, n_rows, device, chunk=2 ** 18):
    """Sample the (u, z) grid, predict (x, y), bilinear CT lookup -> strip.

    Returns dict with the strip and the Jacobian / residual diagnostic maps.
    """
    uv_norm, u_flat, z_flat = build_uv_grid(dataset, n_cols, n_rows, device)
    total = uv_norm.shape[0]
    strip_flat = torch.zeros(total, device=device)
    res_flat = torch.zeros(total, device=device)
    E_flat = torch.zeros(total, device=device)
    G_flat = torch.zeros(total, device=device)
    F_flat = torch.zeros(total, device=device)

    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        uv = uv_norm[start:end]
        u_raw = u_flat[start:end]
        z_idx = z_flat[start:end]

        xy_base = dataset.analytical_xy_norm(u_raw)
        uv_req = uv.detach().clone().requires_grad_(True)
        xy_res = model(uv_req)
        pred_xy = xy_base + xy_res

        gx = torch.autograd.grad(pred_xy[:, 0].sum(), uv_req,
                                 create_graph=False, retain_graph=True)[0]
        gy = torch.autograd.grad(pred_xy[:, 1].sum(), uv_req,
                                 create_graph=False)[0]
        E_flat[start:end] = gx[:, 0] ** 2 + gy[:, 0] ** 2
        G_flat[start:end] = gx[:, 1] ** 2 + gy[:, 1] ** 2
        F_flat[start:end] = gx[:, 0] * gx[:, 1] + gy[:, 0] * gy[:, 1]
        res_flat[start:end] = xy_res.detach().norm(dim=1)

        with torch.no_grad():
            strip_flat[start:end] = dataset.sample_image(pred_xy.detach(), z_idx)

    strip = strip_flat.view(n_rows, n_cols).cpu().numpy()
    return {"strip": strip,
            "E_map": E_flat.view(n_rows, n_cols).cpu().numpy(),
            "G_map": G_flat.view(n_rows, n_cols).cpu().numpy(),
            "F_map": F_flat.view(n_rows, n_cols).cpu().numpy(),
            "res_map": res_flat.view(n_rows, n_cols).cpu().numpy()}


def save_strip_figures(strip, out_dir, n_layers, pixels_per_winding):
    n_rows, n_cols = strip.shape
    fig, ax = plt.subplots(1, 1, figsize=(24, max(6, n_rows / 30)))
    ax.imshow(strip, cmap="gray", aspect="auto", interpolation="nearest")
    ax.set_title(f"Unwrapped strip ({n_layers} windings, {n_rows} slices)")
    ax.set_xlabel("u")
    ax.set_ylabel("z")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "strip_full.png"), dpi=200)
    plt.close()

    # 2-winding bands, rows upsampled 8× so short strips (e.g. 32 z-slices)
    # are visually readable.
    row_up = max(1, 256 // n_rows)
    for start_l in range(0, n_layers, 2):
        end_l = min(start_l + 2, n_layers)
        c0 = int(start_l * pixels_per_winding)
        c1 = int(end_l * pixels_per_winding)
        crop = strip[:, c0:c1]
        crop_up = np.repeat(crop, row_up, axis=0)
        h, w = crop_up.shape
        fig, ax = plt.subplots(1, 1, figsize=(min(24, w / 80), max(3, h / 80)))
        ax.imshow(crop_up, cmap="gray", aspect="auto", interpolation="nearest",
                  vmin=0, vmax=1)
        ax.set_title(f"Layers {start_l}–{end_l}  (rows ×{row_up})")
        ax.set_xlabel("u (strip columns)")
        ax.set_ylabel("z")
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"detail_layers_{start_l}_{end_l}.png"), dpi=150
        )
        plt.close()


def save_diagnostic_maps(maps, out_dir):
    for key, label in [("res_map", "||INR residual|| (normalized)"),
                       ("E_map", "E = ||∂f/∂u||²"),
                       ("G_map", "G = ||∂f/∂z||²"),
                       ("F_map", "|F| = |<∂f/∂u, ∂f/∂z>|")]:
        arr = maps[key]
        if key == "F_map":
            arr = np.abs(arr)
        fig, ax = plt.subplots(1, 1, figsize=(16, 4))
        im = ax.imshow(arr, aspect="auto", cmap="viridis")
        ax.set_title(label)
        ax.set_xlabel("u")
        ax.set_ylabel("z")
        plt.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{key}.png"), dpi=200)
        plt.close()

    # Conformal residual = |E - G| + 2|F|
    conf = np.abs(maps["E_map"] - maps["G_map"]) + 2.0 * np.abs(maps["F_map"])
    fig, ax = plt.subplots(1, 1, figsize=(16, 4))
    im = ax.imshow(conf, aspect="auto", cmap="magma")
    ax.set_title("Conformal residual |E − G| + 2|F|")
    ax.set_xlabel("u")
    ax.set_ylabel("z")
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "conformal_map.png"), dpi=200)
    plt.close()


def _estimate_frame_w(n_rows, W):
    """One movie frame's width in strip columns.

    Matches the synthetic generator's layout (`generate.py::_video_strip`),
    which resizes each video frame to height `h` × width `h * 4 // 3` (4:3
    aspect) and tiles frames end-to-end. Using 16:9 here would put the crop
    at a fractional frame offset, splitting one panel across two frames.
    """
    return max(50, (n_rows * 4) // 3)


def save_zoomed_strip_windows(aligned, gt_crop, out_dir, n_windows=4,
                               n_frames_per_window=2):
    """Zoomed GT vs predicted comparisons at N equally-spaced strip positions.

    Each window spans `n_frames_per_window` consecutive movie frames at natural
    (square-pixel) aspect ratio — no row upsampling. Saves zoom_window_00.png …
    """
    n_rows, W = aligned.shape
    frame_w = _estimate_frame_w(n_rows, W)
    window_cols = frame_w * n_frames_per_window

    # Snap window centres to frame boundaries: choose c0 = k * frame_w nearest
    # to the desired centre. Without this, panels straddle two frames.
    centers = np.linspace(W // 8, W * 7 // 8, n_windows, dtype=int)
    for i, center in enumerate(centers):
        c0_unsnapped = int(max(0, center - window_cols // 2))
        c0 = (c0_unsnapped // frame_w) * frame_w
        c1 = int(min(W, c0 + window_cols))
        w = c1 - c0

        gt_w   = np.clip(gt_crop[:, c0:c1], 0.0, 1.0)
        pr_w   = np.clip(aligned[:, c0:c1], 0.0, 1.0)
        diff_w = np.abs(pr_w - gt_w)

        dpi = 100
        pw = w / dpi          # panel width in inches
        ph = n_rows / dpi     # panel height in inches
        fig, axes = plt.subplots(3, 1,
                                  figsize=(pw, ph * 3 + 1.2),
                                  constrained_layout=True)
        for ax, img, title, cmap, vmax in zip(
            axes,
            [gt_w, pr_w, diff_w],
            [f"GT  (cols {c0}–{c1})", "Predicted (aligned)", "|diff|  (max=0.3)"],
            ["gray", "gray", "magma"],
            [1.0, 1.0, 0.3],
        ):
            ax.imshow(img, cmap=cmap, aspect="equal",
                      vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.savefig(os.path.join(out_dir, f"zoom_window_{i:02d}.png"), dpi=dpi)
        plt.close()


def save_single_frame_compare(aligned, gt_crop, out_dir, n_frames=1):
    """One movie frame shown at correct aspect: GT | Predicted | |diff|.

    Finds the richest-content frame (highest GT std) and shows it with square
    pixels. Saves frame_compare.png. For a 256-row strip this is ≈256×455 px
    per panel — clearly shows movie content with no upsampling or distortion.
    """
    n_rows, W = aligned.shape
    frame_w = _estimate_frame_w(n_rows, W)
    crop_w = frame_w * n_frames
    crop_w = min(crop_w, W)

    # Find column window with highest GT content (std). Stride = frame_w so
    # the chosen c0 starts on a movie-frame boundary instead of splitting one.
    best_c0, best_std = 0, -1.0
    for c0 in range(0, W - crop_w, frame_w):
        v = float(gt_crop[:, c0:c0 + crop_w].std())
        if v > best_std:
            best_std, best_c0 = v, c0
    c0, c1 = best_c0, best_c0 + crop_w

    gt_f   = np.clip(gt_crop[:, c0:c1], 0.0, 1.0)
    pr_f   = np.clip(aligned[:, c0:c1], 0.0, 1.0)
    diff_f = np.abs(pr_f - gt_f)

    dpi = 100
    pw = (c1 - c0) / dpi
    ph = n_rows / dpi
    fig, axes = plt.subplots(1, 3,
                              figsize=(pw * 3 + 0.3, ph + 0.6),
                              constrained_layout=True)
    axes[0].imshow(gt_f,   cmap="gray",  aspect="equal",
                   vmin=0, vmax=1, interpolation="nearest")
    axes[0].set_title("GT", fontsize=9)
    axes[1].imshow(pr_f,   cmap="gray",  aspect="equal",
                   vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title("Predicted", fontsize=9)
    im = axes[2].imshow(diff_f, cmap="magma", aspect="equal",
                        vmin=0, vmax=0.3, interpolation="nearest")
    axes[2].set_title("|diff|", fontsize=9)
    plt.colorbar(im, ax=axes[2], shrink=0.8, pad=0.02)
    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
    plt.suptitle(
        f"Best frame  cols {c0}–{c1}  ({n_frames} frame{'s' if n_frames > 1 else ''},"
        f" {frame_w}px/frame)",
        fontsize=8,
    )
    plt.savefig(os.path.join(out_dir, "frame_compare.png"), dpi=dpi)
    plt.close()


def save_surface_overlay(dataset, model, out_dir, n_samples=4000, z_slice=None):
    """Overlay predicted (x,y) path for a dense u sweep on a CT cross-section."""
    device = dataset.device
    if z_slice is None:
        z_slice = dataset.Z // 2

    u_raw = torch.linspace(0.0, dataset.n_layers - 1e-4, n_samples, device=device)
    u_norm = u_raw / dataset.n_layers * 2.0 - 1.0
    if dataset.Z > 1:
        z_norm = torch.full_like(u_norm, z_slice / (dataset.Z - 1) * 2.0 - 1.0)
    else:
        z_norm = torch.zeros_like(u_norm)
    uv = torch.stack([u_norm, z_norm], dim=1)

    xy_base = dataset.analytical_xy_norm(u_raw)
    if model is None:
        pred_xy = xy_base
    else:
        with torch.no_grad():
            pred_xy = xy_base + model(uv)

    xs = ((pred_xy[:, 0] + 1) * 0.5 * (dataset.W - 1)).cpu().numpy()
    ys = ((pred_xy[:, 1] + 1) * 0.5 * (dataset.H - 1)).cpu().numpy()
    u_vals = u_raw.cpu().numpy()

    img = dataset.image_volume[0, 0, z_slice].cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img, cmap="gray")
    sc = ax.scatter(xs, ys, c=u_vals, s=2, cmap="turbo")
    plt.colorbar(sc, ax=ax, label="u", shrink=0.8)
    ax.set_title(f"Predicted surface path @ z={z_slice}")
    ax.set_xlim(0, dataset.W)
    ax.set_ylim(dataset.H, 0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "surface_overlay.png"), dpi=200)
    plt.close()


def _resample_row_nearest(row, new_len):
    """Nearest-neighbor resample a 1-D row to `new_len`."""
    n_old = row.shape[0]
    idx = np.clip(
        np.round(np.linspace(0, n_old - 1, new_len)).astype(np.int64),
        0, n_old - 1,
    )
    return row[idx]


def _align_row_xcorr(pred, target, max_shift):
    """Find integer circular shift that maximises normalised cross-correlation
    of pred against target (both 1-D, same length). Returns
    (aligned_pred, best_shift, best_score).

    FFT-based: O(n log n) vs the O(max_shift × n) python loop the original
    implementation used — at n≈143k and max_shift≈36k that loop is hours of
    pure-numpy multiplies per row × 256 rows. The FFT path runs in seconds
    for the whole strip.
    """
    n = len(target)
    p = (pred - pred.mean()) / (pred.std() + 1e-8)
    t = (target - target.mean()) / (target.std() + 1e-8)
    # Circular cross-correlation via FFT. ccorr[k] = sum_i p[i]·t[(i−k) mod n],
    # normalised by n so it's directly comparable to the original (a*b).mean().
    P = np.fft.rfft(p)
    T = np.fft.rfft(t)
    ccorr = np.fft.irfft(P * np.conj(T), n=n) / n
    # Restrict to |shift| ≤ max_shift on a circular axis.
    m = min(max_shift, n // 2)
    valid = np.concatenate([ccorr[:m + 1], ccorr[n - m:]])
    valid_shifts = np.concatenate([
        np.arange(0, m + 1, dtype=np.int64),
        np.arange(-m, 0, dtype=np.int64),
    ])
    best_idx = int(np.argmax(valid))
    best_shift = int(valid_shifts[best_idx])
    best_score = float(valid[best_idx])
    aligned = np.roll(pred, -best_shift)
    return aligned, best_shift, best_score


def _percentile_normalize(x, lo_p=1.0, hi_p=99.0):
    lo, hi = np.percentile(x, [lo_p, hi_p])
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _compare_strips_ssim(a, b, normalize_per_strip=False):
    """Aligned-row SSIM + RMSE between two same-shape strips.

    If normalize_per_strip is True, each strip is independently rescaled to
    [0, 1] via 1st/99th percentile before comparison — robust to wildly
    different absolute brightness (e.g. raw CT intensity vs content space).
    """
    n_rows = min(a.shape[0], b.shape[0])
    W = min(a.shape[1], b.shape[1])
    a, b = a[:n_rows, :W], b[:n_rows, :W]
    if normalize_per_strip:
        a = _percentile_normalize(a)
        b = _percentile_normalize(b)
    max_shift = W // 4
    aligned = np.stack([
        _align_row_xcorr(a[z], b[z], max_shift)[0] for z in range(n_rows)
    ], axis=0)
    rmse = float(np.sqrt(((aligned - b) ** 2).mean()))
    ssim_val = None
    try:
        from skimage.metrics import structural_similarity as _ssim
        ssim_val = float(_ssim(aligned, b, data_range=1.0,
                               gaussian_weights=True, use_sample_covariance=False,
                               sigma=1.5))
    except Exception:
        pass
    return ssim_val, rmse


def evaluate_against_gt(strip, gt_npz_path, dataset_n_layers, out_dir,
                        oracle_npz_path=None):
    """Compute strip RMSE vs GT strip + winding-count accuracy.

    strip          : (n_rows, n_cols) rendered INR strip, intensity ∈ [0, 1]
    gt_npz_path    : path to synthetic ground_truth.npz
    dataset_n_layers : the INR's detected winding count

    Returns a dict of metrics and writes eval_metrics.json + gt_compare.png.
    """
    gt = np.load(gt_npz_path)
    gt_strip = gt["strip"]                      # (n_z, strip_width)
    gt_n_windings = int(gt["n_windings"])

    n_rows_gt, W_gt = gt_strip.shape
    n_rows, W_pred = strip.shape

    # Normalise strip content to ~[0, 1] for a fair comparison.
    # INR strip is raw CT absorption; GT strip is the source pattern before
    # being projected through the intensity model. We invert on the CT side:
    #   pred_content = (pred_intens - base) / (emul_max - base)
    base = float(gt["film_base_intensity"])
    emul_max = float(gt["emulsion_max_intensity"])
    pred_content = np.clip((strip - base) / (emul_max - base), 0.0, 1.0)

    # Resample predicted strip cols to match GT cols; crop rows to common min.
    n_rows_common = min(n_rows, n_rows_gt)
    pred_resized = np.stack([
        _resample_row_nearest(pred_content[z], W_gt)
        for z in range(n_rows_common)
    ], axis=0)
    gt_crop = gt_strip[:n_rows_common]

    # Per-row cross-correlation to absorb seam-angle offset (circular shift).
    max_shift = W_gt // 4
    aligned_rows = []
    shifts = []
    for z in range(n_rows_common):
        aligned, shift, _ = _align_row_xcorr(
            pred_resized[z], gt_crop[z], max_shift
        )
        aligned_rows.append(aligned)
        shifts.append(shift)
    aligned = np.stack(aligned_rows, axis=0)

    strip_rmse = float(np.sqrt(((aligned - gt_crop) ** 2).mean()))
    strip_mae = float(np.abs(aligned - gt_crop).mean())
    # PSNR assuming peak=1.0 (strip values are in [0, 1] after the intensity
    # model inversion). Higher = better; ∞ would be perfect reconstruction.
    strip_psnr = (
        float(20.0 * np.log10(1.0 / strip_rmse)) if strip_rmse > 0 else float("inf")
    )

    # Per-row correlation (circular-shift-corrected).
    rowcorrs = []
    for z in range(n_rows_common):
        a = aligned[z] - aligned[z].mean()
        t = gt_crop[z] - gt_crop[z].mean()
        denom = (a.std() * t.std() + 1e-8) * len(a)
        rowcorrs.append(float((a * t).sum() / denom))
    row_corr_mean = float(np.mean(rowcorrs))

    # SSIM (structural similarity, [-1, 1], higher is better). Captures edge
    # contrast / local structure that row_corr does not — better tracks the
    # "looks like a low-res version of the GT" visual deficit. skimage is
    # available on Merlin7 (conda env "nnunet"); fall back to None locally if
    # absent.
    ssim_val = None
    try:
        from skimage.metrics import structural_similarity as _ssim
        # SSIM on the aligned pred vs gt_crop, treating intensity range as 1.0.
        ssim_val = float(_ssim(aligned, gt_crop, data_range=1.0,
                               gaussian_weights=True, use_sample_covariance=False,
                               sigma=1.5))
    except Exception as exc:                                    # noqa: BLE001
        print(f"  SSIM unavailable: {exc}")

    metrics = {
        "gt_n_windings": gt_n_windings,
        "detected_n_layers": int(dataset_n_layers),
        "winding_detection_correct": int(dataset_n_layers) == gt_n_windings,
        "strip_rmse": strip_rmse,
        "strip_mae": strip_mae,
        "strip_psnr_db": strip_psnr,
        "row_corr_mean": row_corr_mean,
        "median_shift_px": int(np.median(shifts)),
        "ssim": ssim_val,
        "strip_shape_pred": list(pred_resized.shape),
        "strip_shape_gt": list(gt_crop.shape),
    }

    # Oracle diagnostic: how much of the gap to GT is mapping error
    # (recoverable) vs CT-rendering floor (irreducible)? Oracle = CT sampled
    # at the GT u_map. Comparing pred ↔ oracle isolates mapping error;
    # comparing oracle ↔ GT bounds what any geometrically-perfect mapping
    # could achieve.
    if oracle_npz_path is not None and os.path.exists(oracle_npz_path):
        ora = np.load(oracle_npz_path)
        # The oracle file convention may vary; accept several common keys.
        if "strip" in ora.files:
            oracle_strip = ora["strip"]
        elif "oracle_strip" in ora.files:
            oracle_strip = ora["oracle_strip"]
        else:
            print(f"  oracle npz {oracle_npz_path} has no 'strip' or 'oracle_strip' key; skipping")
            oracle_strip = None
        if oracle_strip is not None:
            # Oracle, pred, and GT live in different absolute scales (raw CT,
            # CT-intensity, content). For diagnostic comparison, normalize each
            # to [0, 1] via 1/99 percentile so SSIM measures structural
            # similarity not absolute brightness.
            pred_v_oracle_ssim, pred_v_oracle_rmse = _compare_strips_ssim(
                pred_resized, oracle_strip, normalize_per_strip=True
            )
            oracle_v_gt_ssim, oracle_v_gt_rmse = _compare_strips_ssim(
                oracle_strip, gt_crop, normalize_per_strip=True
            )
            metrics["oracle_ssim_vs_gt"] = oracle_v_gt_ssim
            metrics["oracle_rmse_vs_gt"] = oracle_v_gt_rmse
            metrics["pred_ssim_vs_oracle"] = pred_v_oracle_ssim
            metrics["pred_rmse_vs_oracle"] = pred_v_oracle_rmse

    with open(os.path.join(out_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Synthetic GT metrics:")
    print(json.dumps(metrics, indent=2))

    # Comparison figure: three rows — GT, pred (aligned), abs diff.
    fig, axes = plt.subplots(3, 1, figsize=(24, 9))
    axes[0].imshow(gt_crop, cmap="gray", aspect="auto", vmin=0, vmax=1,
                   interpolation="nearest")
    axes[0].set_title(f"GT strip ({gt_n_windings} windings)")
    axes[1].imshow(aligned, cmap="gray", aspect="auto", vmin=0, vmax=1,
                   interpolation="nearest")
    axes[1].set_title(
        f"Predicted strip (aligned; {dataset_n_layers} detected windings)  "
        f"rmse={strip_rmse:.3f}  psnr={strip_psnr:.2f} dB  row-corr={row_corr_mean:.3f}"
    )
    axes[2].imshow(np.abs(aligned - gt_crop), cmap="magma", aspect="auto",
                   vmin=0, vmax=0.5, interpolation="nearest")
    axes[2].set_title("|aligned − GT|")
    for a in axes:
        a.set_xlabel("u")
        a.set_ylabel("z")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "gt_compare.png"), dpi=150)
    plt.close()

    # Additional zoomed visualizations
    save_zoomed_strip_windows(aligned, gt_crop, out_dir)
    save_single_frame_compare(aligned, gt_crop, out_dir)

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ckpt", required=True, help="trained checkpoint (.pt)")
    parser.add_argument("--pixels-per-winding", type=int, default=1440)
    parser.add_argument("--attachment", choices=["film", "emulsion", "centerline"],
                        default="film",
                        help="match the attachment mode used at training time")
    parser.add_argument("--centerline-erode", type=int, default=1)
    parser.add_argument("--gt-npz", type=str, default=None,
                        help="synthetic ground truth: strip metrics + winding count")
    parser.add_argument("--oracle-npz", type=str, default=None,
                        help="oracle_strip.npz (CT sampled at the GT mapping): adds "
                             "pred<->oracle and oracle<->GT SSIM/RMSE")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = SurfaceDataset(
        args.data_dir, device=device, diag_dir=args.out_dir,
        attachment=args.attachment, centerline_erode=args.centerline_erode,
    )

    # If a GT file is provided, override geometry so eval matches the
    # analytical base the INR was trained under.
    if args.gt_npz:
        dataset.load_synthetic_gt(args.gt_npz, geometry_only=True)

    model, ckpt = load_model(args.ckpt, device)
    print(f"Loaded checkpoint: {args.ckpt}")

    n_cols = dataset.n_layers * args.pixels_per_winding
    n_rows = dataset.Z
    print(f"Rendering {n_rows} × {n_cols} strip...")

    result = render_strip(dataset, model, n_cols, n_rows, device)

    save_strip_figures(result["strip"], args.out_dir,
                       dataset.n_layers, args.pixels_per_winding)
    save_surface_overlay(dataset, model, args.out_dir)
    save_diagnostic_maps(result, args.out_dir)

    np.savez_compressed(
        os.path.join(args.out_dir, "strip.npz"),
        strip=result["strip"].astype(np.float32),
        n_layers=dataset.n_layers,
        pixels_per_winding=args.pixels_per_winding,
    )

    if args.gt_npz:
        evaluate_against_gt(
            result["strip"], args.gt_npz, dataset.n_layers, args.out_dir,
            oracle_npz_path=args.oracle_npz,
        )

    print(f"Done.  Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
