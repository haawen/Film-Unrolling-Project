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


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt["args"]
    model = DeformationINR(
        n_fourier=a.get("n_fourier", 256),
        sigma=a.get("sigma", 10.0),
        hidden_dim=a.get("hidden_dim", 256),
        n_layers=a.get("n_layers_mlp", 4),
        input_dim=2, output_dim=2,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = SurfaceDataset(
        args.data_dir, max_slices=args.max_slices, device=device,
        diag_dir=args.out_dir,
    )

    if args.analytical_only:
        model = None
        print("Rendering analytical-only strip (no INR residual).")
    else:
        if args.ckpt is None:
            parser.error("--ckpt is required unless --analytical-only is set.")
        model = load_model(args.ckpt, device)
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
    print(f"Done.  Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
