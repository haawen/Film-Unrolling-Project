"""
Evaluation for self-supervised 3D unwrapping.

Loads trained model, queries all film voxels for u, generates (u, z) strip,
and produces diagnostic visualizations (u-map on cross-section, u vs radius).

Usage:
    python -m unwrapping.inr.ss_eval \
        --data-dir path/to/01_Mickey_3d \
        --model-dir results/ss_unwrap \
        --out-dir results/ss_unwrap/eval
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.inr.ss_data import SelfSupDataset3D
from unwrapping.inr.unwrap_model import DeformationINR


def predict_all(model, dataset, batch_size=65536):
    """Query model at all film voxel coordinates. Returns u as numpy array."""
    model.eval()
    all_u = []

    with torch.no_grad():
        for i in range(0, dataset.n, batch_size):
            end = min(i + batch_size, dataset.n)
            coords = dataset.coords[i:end].to(dataset.device, non_blocking=True)
            u = model(coords).squeeze(-1)
            all_u.append(u.cpu())

    return torch.cat(all_u, dim=0).numpy()


def generate_strip(u_raw, z_global, intensities, n_layers,
                   pixels_per_winding=1440):
    """Generate (u, z) strip from predicted u and known z.

    u is normalized from its learned range to [0, n_layers * pixels_per_winding].
    """
    strip_W = n_layers * pixels_per_winding
    z_min = int(z_global.min())
    z_max = int(z_global.max())
    n_z = z_max - z_min + 1

    strip = np.zeros((n_z, strip_W), dtype=np.float64)
    count = np.zeros((n_z, strip_W), dtype=np.float64)

    # Normalize u to [0, 1]
    u_min, u_max = u_raw.min(), u_raw.max()
    u_norm = (u_raw - u_min) / (u_max - u_min + 1e-8)

    col = (u_norm * (strip_W - 1)).astype(np.int64)
    col = np.clip(col, 0, strip_W - 1)
    row = (z_global - z_min).astype(np.int64)

    np.add.at(strip, (row, col), intensities.astype(np.float64))
    np.add.at(count, (row, col), 1.0)

    mask = count > 0
    strip[mask] /= count[mask]

    return strip, count, z_min, z_max


def save_strip_visualizations(strip, count, n_layers, z_min, z_max,
                              z_slices_unique, out_dir):
    """Save strip overview and detail crops."""
    n_z = strip.shape[0]
    pixels_per_winding = strip.shape[1] // n_layers

    # Full strip + coverage
    fig, axes = plt.subplots(2, 1, figsize=(24, max(6, n_z / 30)))
    axes[0].imshow(strip, cmap="gray", aspect="auto", interpolation="nearest")
    axes[0].set_title(f"Self-Supervised Unrolled Film Strip ({n_layers} layers, "
                      f"z=[{z_min}-{z_max}])")
    axes[0].set_ylabel(f"z (slice index - {z_min})")
    axes[0].set_xlabel("u (along film)")

    coverage = (count > 0).astype(float)
    filled_rows = len(z_slices_unique)
    total_cov = (coverage.sum() / (filled_rows * strip.shape[1]) * 100
                 if filled_rows > 0 else 0)
    axes[1].imshow(coverage, cmap="gray", aspect="auto", interpolation="nearest")
    axes[1].set_title(f"Coverage ({total_cov:.1f}% of filled rows)")
    axes[1].set_ylabel(f"z (slice index - {z_min})")
    axes[1].set_xlabel("u (along film)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "strip_3d.png"), dpi=200)
    plt.close()

    # Detail crops per layer group
    for start_layer in range(0, n_layers, 5):
        end_layer = min(start_layer + 5, n_layers)
        col_start = start_layer * pixels_per_winding
        col_end = end_layer * pixels_per_winding
        crop = strip[:, col_start:col_end]

        fig, ax = plt.subplots(1, 1, figsize=(16, max(4, n_z / 50)))
        ax.imshow(crop, cmap="gray", aspect="auto", interpolation="nearest")
        ax.set_title(f"Layers {start_layer}-{end_layer}")
        ax.set_xlabel("u (along film)")
        ax.set_ylabel(f"z (slice index - {z_min})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir,
                    f"detail_layers_{start_layer}_{end_layer}.png"), dpi=200)
        plt.close()

    # Single winding zoom
    mid = n_layers // 2
    crop = strip[:, mid * pixels_per_winding:(mid + 1) * pixels_per_winding]
    fig, ax = plt.subplots(1, 1, figsize=(14, max(6, n_z / 30)))
    ax.imshow(crop, cmap="gray", aspect="auto", interpolation="nearest")
    ax.set_title(f"Single Winding Detail: Layer {mid}")
    ax.set_xlabel("u (along film)")
    ax.set_ylabel(f"z (slice index - {z_min})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "detail_single_winding.png"), dpi=200)
    plt.close()


def save_diagnostics(u_raw, dataset, out_dir, slice_idx=0):
    """Save diagnostic plots: u-map on cross-section, u vs radius."""
    z_global_np = dataset.z_global.numpy()
    z_slices = sorted(set(z_global_np.tolist()))

    if slice_idx >= len(z_slices):
        slice_idx = 0
    target_z = z_slices[slice_idx]
    mask_z = z_global_np == target_z

    if mask_z.sum() == 0:
        print("  WARNING: no pixels for diagnostic slice, skipping")
        return

    # Extract slice data
    coords_slice = dataset.coords[mask_z].numpy()
    u_slice = u_raw[mask_z]
    r_slice = dataset.radius[mask_z].numpy()

    H, W = dataset.image_shape
    xs = ((coords_slice[:, 0] + 1) / 2 * (W - 1)).astype(int)
    ys = ((coords_slice[:, 1] + 1) / 2 * (H - 1)).astype(int)

    # 1. u-map on cross-section
    u_map = np.full((H, W), np.nan)
    u_map[ys, xs] = u_slice

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    im = ax.imshow(u_map, cmap="turbo", interpolation="nearest")
    ax.set_title(f"Learned u field — z={int(target_z)} "
                 f"(should show spiral color gradient)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="u (learned)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "umap_crosssection.png"), dpi=200)
    plt.close()

    # 2. u vs radius scatter
    # Subsample for readability
    n_plot = min(50000, len(u_slice))
    idx = np.random.choice(len(u_slice), n_plot, replace=False)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.scatter(r_slice[idx], u_slice[idx], s=0.1, alpha=0.3, c="steelblue")
    ax.set_xlabel("Radius from center (pixels)")
    ax.set_ylabel("Learned u")
    ax.set_title(f"u vs radius — z={int(target_z)} "
                 f"(should show monotonic staircase)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "u_vs_radius.png"), dpi=200)
    plt.close()

    # 3. u histogram
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.hist(u_slice, bins=200, color="steelblue", alpha=0.8)
    ax.set_xlabel("Learned u")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of u (should be roughly uniform if eikonal works)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "u_histogram.png"), dpi=200)
    plt.close()


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load training config
    log_path = os.path.join(args.model_dir, "train_log.json")
    with open(log_path) as f:
        log = json.load(f)
    config = log["config"]

    # Load data
    print("Loading 3D volume data (self-supervised)...")
    dataset = SelfSupDataset3D(
        args.data_dir, device=device, max_slices=args.max_slices
    )

    # Build and load model
    model = DeformationINR(
        n_fourier=config["n_fourier"],
        sigma=config["sigma"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        input_dim=3,
        output_dim=1,
    ).to(device)

    ckpt_path = os.path.join(args.model_dir, f"{args.checkpoint}.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                     weights_only=True))
    print(f"  Loaded checkpoint: {ckpt_path}")

    # Predict u for all film voxels
    print("Predicting u for all film voxels...")
    u_raw = predict_all(model, dataset, batch_size=args.batch_size)
    print(f"  u range: [{u_raw.min():.2f}, {u_raw.max():.2f}]")

    z_global = dataset.z_global.numpy()
    intensities = dataset.intensities.numpy()

    # Generate strip
    print("Generating (u, z) strip...")
    strip, count, z_min, z_max = generate_strip(
        u_raw, z_global, intensities, dataset.n_layers_est,
        pixels_per_winding=args.pixels_per_winding,
    )
    print(f"  Strip shape: {strip.shape}")

    # Save strip
    print("Saving results...")
    np.savez_compressed(
        os.path.join(args.out_dir, "strip_3d.npz"),
        strip=strip.astype(np.float32),
        count=count.astype(np.int32),
        z_min=z_min, z_max=z_max,
        u_raw_min=float(u_raw.min()),
        u_raw_max=float(u_raw.max()),
    )

    z_slices_unique = sorted(set(z_global.tolist()))
    save_strip_visualizations(strip, count, dataset.n_layers_est, z_min, z_max,
                              z_slices_unique, args.out_dir)

    # Diagnostics
    print("Generating diagnostic plots...")
    save_diagnostics(u_raw, dataset, args.out_dir)

    # Metrics
    filled_rows = len(z_slices_unique)
    filled_coverage = sum(
        (count[int(z) - z_min, :] > 0).mean() for z in z_slices_unique
    ) / filled_rows * 100 if filled_rows > 0 else 0

    stats = {
        "strip_shape": list(strip.shape),
        "n_layers_est": dataset.n_layers_est,
        "n_slices": filled_rows,
        "z_range": [z_min, z_max],
        "coverage_filled_rows_pct": float(filled_coverage),
        "pixels_per_winding": args.pixels_per_winding,
        "u_range_learned": [float(u_raw.min()), float(u_raw.max())],
        "approach": "self-supervised",
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Coverage: {filled_coverage:.1f}%")
    print(f"  u range: [{u_raw.min():.2f}, {u_raw.max():.2f}]")
    print(f"  Results: {args.out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate self-supervised unwrapping model"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--pixels-per-winding", type=int, default=1440)
    parser.add_argument("--max-slices", type=int, default=None)

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
