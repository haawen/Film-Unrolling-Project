"""
Render unwrapped film strip from the residual inverse-mapping INR.

Two modes:
  --analytical-only : render strip using only the analytical spiral
                      (no training required, no checkpoint needed).
                      Used as a visual sanity check before training.
  (default)         : load a trained model and render strip as
                      analytical_xy(u) + INR(u, z).

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

from unwrapping.inr.surface_data import SurfaceDataset
from unwrapping.inr.unwrap_model import DeformationINR
from unwrapping.inr.perwinding_model import PerWindingParametric


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
    elif mt == "perwinding":
        # Recover n_windings from checkpoint (param shape) so eval works even if
        # detection re-runs differently.
        coeffs_shape = ckpt["model"]["coeffs"].shape
        n_w = int(coeffs_shape[0])
        n_h = int(coeffs_shape[2])
        model = PerWindingParametric(
            n_layers=n_w, n_harmonics=n_h, output_dim=2,
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type in ckpt: {mt}")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def apply_learned_eccentricity(dataset, ckpt):
    """If the checkpoint trained with --learn-eccentricity, copy the learned
    amp/phase onto the dataset so analytical_xy_norm applies the correction."""
    if "ecc_amp" in ckpt and "ecc_phase" in ckpt:
        dataset.ecc_amp = torch.tensor(
            ckpt["ecc_amp"], device=dataset.device, requires_grad=False
        )
        dataset.ecc_phase = torch.tensor(
            ckpt["ecc_phase"], device=dataset.device, requires_grad=False
        )
        dataset._ecc_learnable = True
        print(f"Applied learned eccentricity: amp={ckpt['ecc_amp']:+.3f}px, "
              f"phase={ckpt['ecc_phase']:+.3f} rad")
    if "pw_amp" in ckpt and "pw_phase" in ckpt:
        pw_amp = torch.tensor(
            ckpt["pw_amp"], device=dataset.device, requires_grad=False
        )
        pw_phase = torch.tensor(
            ckpt["pw_phase"], device=dataset.device, requires_grad=False
        )
        # Eval may have re-detected n_layers; clip to the shorter length.
        n = min(pw_amp.numel(), dataset.n_layers)
        dataset.pw_amp = torch.zeros(
            (dataset.n_layers,), device=dataset.device, requires_grad=False
        )
        dataset.pw_phase = torch.zeros(
            (dataset.n_layers,), device=dataset.device, requires_grad=False
        )
        dataset.pw_amp[:n] = pw_amp[:n]
        dataset.pw_phase[:n] = pw_phase[:n]
        dataset._pw_ecc_learnable = True
        print(f"Applied per-winding ecc: |amp|_mean="
              f"{dataset.pw_amp.abs().mean().item():.2f}px, "
              f"max|amp|={dataset.pw_amp.abs().max().item():.2f}px")


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


def render_strip(dataset, model, n_cols, n_rows, device, chunk=2 ** 18,
                 also_maps=True):
    """Sample grid, predict (x,y), bilinear-lookup intensity → strip.

    Returns dict with strip, optional Jacobian / residual diagnostic maps.
    """
    uv_norm, u_flat, z_flat = build_uv_grid(dataset, n_cols, n_rows, device)
    total = uv_norm.shape[0]
    strip_flat = torch.zeros(total, device=device)
    res_flat = torch.zeros(total, device=device) if also_maps else None
    E_flat = torch.zeros(total, device=device) if also_maps else None
    G_flat = torch.zeros(total, device=device) if also_maps else None
    F_flat = torch.zeros(total, device=device) if also_maps else None

    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        uv = uv_norm[start:end]
        u_raw = u_flat[start:end]
        z_idx = z_flat[start:end]

        xy_base = dataset.analytical_xy_norm(u_raw)
        if model is None:
            xy_res = torch.zeros_like(xy_base)
            pred_xy = xy_base
            if also_maps:
                # Analytical Jacobian computed numerically via small perturbation
                # — not needed here; just record zero for residual / pass-through for E,G.
                pass
        else:
            if also_maps:
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
            else:
                with torch.no_grad():
                    xy_res = model(uv)
                    pred_xy = xy_base + xy_res

        with torch.no_grad():
            intens = dataset.sample_image(pred_xy.detach(), z_idx)
            strip_flat[start:end] = intens

    strip = strip_flat.view(n_rows, n_cols).cpu().numpy()
    out = {"strip": strip}
    if also_maps and model is not None:
        out["E_map"] = E_flat.view(n_rows, n_cols).cpu().numpy()
        out["G_map"] = G_flat.view(n_rows, n_cols).cpu().numpy()
        out["F_map"] = F_flat.view(n_rows, n_cols).cpu().numpy()
        out["res_map"] = res_flat.view(n_rows, n_cols).cpu().numpy()
    return out


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

    for start_l in range(0, n_layers, 5):
        end_l = min(start_l + 5, n_layers)
        c0 = int(start_l * pixels_per_winding)
        c1 = int(end_l * pixels_per_winding)
        crop = strip[:, c0:c1]
        fig, ax = plt.subplots(1, 1, figsize=(16, max(4, n_rows / 50)))
        ax.imshow(crop, cmap="gray", aspect="auto", interpolation="nearest")
        ax.set_title(f"Layers {start_l}-{end_l}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"detail_layers_{start_l}_{end_l}.png"), dpi=200
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


def save_surface_overlay(dataset, model, out_dir, n_samples=4000, z_slice=None):
    """Overlay predicted (x,y) path for a dense u sweep on a CT cross-section."""
    device = dataset.device
    if z_slice is None:
        z_slice = dataset.Z // 2

    u_raw = torch.linspace(0.0, dataset.n_layers - 1e-4, n_samples, device=device)
    z_idx = torch.full((n_samples,), z_slice, dtype=torch.long, device=device)
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
    """Find integer shift that maximises normalised cross-correlation of pred
    against target (both 1-D).  Returns (aligned_pred, best_shift).
    """
    p = (pred - pred.mean()) / (pred.std() + 1e-8)
    t = (target - target.mean()) / (target.std() + 1e-8)
    n = len(t)
    best_shift, best_score = 0, -np.inf
    for s in range(-max_shift, max_shift + 1):
        if s >= 0:
            a = p[s:s + n - abs(s)]
            b = t[:n - abs(s)]
        else:
            a = p[:n - abs(s)]
            b = t[abs(s):]
        if len(a) < n // 2:
            continue
        score = float((a * b).mean())
        if score > best_score:
            best_score, best_shift = score, s
    if best_shift >= 0:
        aligned = np.concatenate([pred[best_shift:], pred[:best_shift]])
    else:
        aligned = np.concatenate([pred[best_shift:], pred[:best_shift]])
    return aligned, best_shift, best_score


def evaluate_against_gt(strip, gt_npz_path, dataset_n_layers, out_dir):
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

    # Per-row correlation (circular-shift-corrected).
    rowcorrs = []
    for z in range(n_rows_common):
        a = aligned[z] - aligned[z].mean()
        t = gt_crop[z] - gt_crop[z].mean()
        denom = (a.std() * t.std() + 1e-8) * len(a)
        rowcorrs.append(float((a * t).sum() / denom))
    row_corr_mean = float(np.mean(rowcorrs))

    metrics = {
        "gt_n_windings": gt_n_windings,
        "detected_n_layers": int(dataset_n_layers),
        "winding_detection_correct": int(dataset_n_layers) == gt_n_windings,
        "strip_rmse": strip_rmse,
        "strip_mae": strip_mae,
        "row_corr_mean": row_corr_mean,
        "median_shift_px": int(np.median(shifts)),
        "strip_shape_pred": list(pred_resized.shape),
        "strip_shape_gt": list(gt_crop.shape),
    }

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
        f"rmse={strip_rmse:.3f}  row-corr={row_corr_mean:.3f}"
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

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ckpt", default=None,
                        help="Path to trained checkpoint (.pt). "
                             "Required unless --analytical-only is set.")
    parser.add_argument("--analytical-only", action="store_true",
                        help="Render strip using only the analytical spiral.")
    parser.add_argument("--max-slices", type=int, default=None)
    parser.add_argument("--pixels-per-winding", type=int, default=1440)
    parser.add_argument("--attachment", choices=["film", "emulsion", "centerline"],
                        default="film",
                        help="Match the attachment mode used at training time "
                             "(affects which mask is displayed/diagnosed).")
    parser.add_argument("--centerline-erode", type=int, default=1)
    parser.add_argument("--winding-detector",
                        choices=["raycast", "histogram"], default="raycast",
                        help="Match the detector used at training.")
    parser.add_argument("--gt-npz", type=str, default=None,
                        help="If provided, compute synthetic GT strip-RMSE "
                             "and winding-count metrics.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = SurfaceDataset(
        args.data_dir, max_slices=args.max_slices, device=device,
        diag_dir=args.out_dir,
        attachment=args.attachment, centerline_erode=args.centerline_erode,
        winding_detector=args.winding_detector,
    )

    # If a GT file is provided, override geometry so eval matches the
    # analytical base the INR was trained under.
    if args.gt_npz:
        dataset.load_synthetic_gt(args.gt_npz, geometry_only=True)

    if args.analytical_only:
        model = None
        print("Rendering analytical-only strip (no INR residual).")
    else:
        if args.ckpt is None:
            parser.error("--ckpt is required unless --analytical-only is set.")
        model, ckpt = load_model(args.ckpt, device)
        apply_learned_eccentricity(dataset, ckpt)
        print(f"Loaded checkpoint: {args.ckpt}")

    n_cols = dataset.n_layers * args.pixels_per_winding
    n_rows = dataset.Z
    print(f"Rendering {n_rows} × {n_cols} strip...")

    result = render_strip(
        dataset, model, n_cols, n_rows, device,
        also_maps=(model is not None),
    )

    save_strip_figures(result["strip"], args.out_dir,
                       dataset.n_layers, args.pixels_per_winding)
    save_surface_overlay(dataset, model, args.out_dir)

    if model is not None:
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
        )

    print(f"Done.  Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
