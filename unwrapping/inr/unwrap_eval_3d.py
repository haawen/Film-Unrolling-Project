"""
Evaluation and strip generation for 3D learned unwrapping (Step 1b — 3D).

Loads trained 3D deformation model, queries all film voxels, generates the
(u, z) strip where u = predicted position along film and z = slice index.

Usage:
    python -m unwrapping.inr.unwrap_eval_3d \
        --data-dir path/to/01_Mickey_3d \
        --model-dir results/unwrap_3d_ml \
        --out-dir results/unwrap_3d_ml/eval
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

from unwrapping.inr.unwrap_data_3d import UnwrapDataset3D
from unwrapping.inr.unwrap_model import DeformationINR


def predict_all(model, dataset, batch_size=65536):
    """Query model at all film voxel coordinates."""
    model.eval()
    all_uv = []

    with torch.no_grad():
        for i in range(0, dataset.n, batch_size):
            end = min(i + batch_size, dataset.n)
            coords = dataset.coords[i:end].to(dataset.device, non_blocking=True)
            uv = model(coords)
            all_uv.append(uv.cpu())

    return torch.cat(all_uv, dim=0).numpy()


def denormalize_uv(uv_std, norm_stats):
    """Convert standardized (u, v) back to raw coordinates."""
    uv_raw = np.zeros_like(uv_std)
    uv_raw[:, 0] = uv_std[:, 0] * norm_stats["u_std"] + norm_stats["u_mean"]
    uv_raw[:, 1] = uv_std[:, 1] * norm_stats["v_std"] + norm_stats["v_mean"]
    return uv_raw


def generate_strip_3d(u_raw, z_global, intensities, n_layers,
                      pixels_per_winding=1440):
    """Generate (u, z) strip from predicted u and known z.

    Args:
        u_raw: (N,) predicted u coordinate (in [0, n_layers])
        z_global: (N,) global z-index for each voxel
        intensities: (N,) voxel intensities
        n_layers: number of spiral layers
        pixels_per_winding: u-resolution per 360 degrees

    Returns:
        strip, count, z_min, z_max
    """
    strip_W = n_layers * pixels_per_winding
    z_min = int(z_global.min())
    z_max = int(z_global.max())
    n_z = z_max - z_min + 1

    strip = np.zeros((n_z, strip_W), dtype=np.float64)
    count = np.zeros((n_z, strip_W), dtype=np.float64)

    col = ((u_raw / n_layers) * strip_W).astype(np.int64)
    col = np.clip(col, 0, strip_W - 1)
    row = (z_global - z_min).astype(np.int64)

    np.add.at(strip, (row, col), intensities.astype(np.float64))
    np.add.at(count, (row, col), 1.0)

    mask = count > 0
    strip[mask] /= count[mask]

    return strip, count, z_min, z_max


def save_visualizations(strip, count, n_layers, z_min, z_max,
                        z_slices_unique, out_dir):
    """Save strip visualizations."""
    n_z = strip.shape[0]
    pixels_per_winding = strip.shape[1] // n_layers

    # 1. Full strip + coverage
    fig, axes = plt.subplots(2, 1, figsize=(24, max(6, n_z / 30)))

    axes[0].imshow(strip, cmap="gray", aspect="auto", interpolation="nearest")
    axes[0].set_title(f"ML Unrolled Film Strip ({n_layers} layers, "
                      f"z=[{z_min}-{z_max}])")
    axes[0].set_ylabel(f"z (slice index - {z_min})")
    axes[0].set_xlabel("u (along film)")

    coverage = (count > 0).astype(float)
    filled_rows = len(z_slices_unique)
    total_cov = coverage.sum() / (filled_rows * strip.shape[1]) * 100 if filled_rows > 0 else 0
    axes[1].imshow(coverage, cmap="gray", aspect="auto", interpolation="nearest")
    axes[1].set_title(f"Coverage ({total_cov:.1f}% of filled rows)")
    axes[1].set_ylabel(f"z (slice index - {z_min})")
    axes[1].set_xlabel("u (along film)")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "strip_3d.png"), dpi=200)
    plt.close()

    # 2. Detail crops per layer group
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
        plt.savefig(os.path.join(out_dir, f"detail_layers_{start_layer}_{end_layer}.png"),
                    dpi=200)
        plt.close()

    # 3. Single winding zoom
    mid_layer = n_layers // 2
    col_start = mid_layer * pixels_per_winding
    col_end = (mid_layer + 1) * pixels_per_winding
    crop = strip[:, col_start:col_end]

    fig, ax = plt.subplots(1, 1, figsize=(14, max(6, n_z / 30)))
    ax.imshow(crop, cmap="gray", aspect="auto", interpolation="nearest")
    ax.set_title(f"Single Winding Detail: Layer {mid_layer}")
    ax.set_xlabel("u (along film)")
    ax.set_ylabel(f"z (slice index - {z_min})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "detail_single_winding.png"), dpi=200)
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
    norm_stats = log["norm_stats"]

    # Load data
    print("Loading 3D volume data...")
    dataset = UnwrapDataset3D(args.data_dir, device=device, max_slices=args.max_slices)

    # Build and load model
    model = DeformationINR(
        n_fourier=config["n_fourier"],
        sigma=config["sigma"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
        input_dim=3,
    ).to(device)

    ckpt_path = os.path.join(args.model_dir, f"{args.checkpoint}.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f"  Loaded checkpoint: {ckpt_path}")

    # Predict (u, v) for all film voxels
    print("Predicting (u, v) for all film voxels...")
    uv_std = predict_all(model, dataset, batch_size=args.batch_size)

    # Denormalize
    uv_raw = denormalize_uv(uv_std, norm_stats)
    u_raw = np.clip(uv_raw[:, 0], 0, dataset.n_layers)
    print(f"  u range: [{u_raw.min():.2f}, {u_raw.max():.2f}]")

    z_global = dataset.z_global.numpy()
    intensities = dataset.intensities.numpy()

    # Generate strip
    print("Generating (u, z) strip...")
    strip, count, z_min, z_max = generate_strip_3d(
        u_raw, z_global, intensities, dataset.n_layers,
        pixels_per_winding=args.pixels_per_winding,
    )
    print(f"  Strip shape: {strip.shape}")

    # Save
    print("Saving results...")
    np.savez_compressed(
        os.path.join(args.out_dir, "strip_3d.npz"),
        strip=strip.astype(np.float32),
        count=count.astype(np.int32),
        z_min=z_min, z_max=z_max,
    )

    z_slices_unique = sorted(set(z_global.tolist()))
    save_visualizations(strip, count, dataset.n_layers, z_min, z_max,
                        z_slices_unique, args.out_dir)

    # Metrics
    filled_rows = len(z_slices_unique)
    filled_coverage = sum(
        (count[int(z) - z_min, :] > 0).mean() for z in z_slices_unique
    ) / filled_rows * 100

    stats = {
        "strip_shape": list(strip.shape),
        "n_layers": dataset.n_layers,
        "n_slices": filled_rows,
        "z_range": [z_min, z_max],
        "coverage_filled_rows_pct": float(filled_coverage),
        "pixels_per_winding": args.pixels_per_winding,
        "u_range_predicted": [float(u_raw.min()), float(u_raw.max())],
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Coverage (filled rows): {filled_coverage:.1f}%")
    print(f"  Results: {args.out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate 3D unwrapping model")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--pixels-per-winding", type=int, default=1440)
    parser.add_argument("--max-slices", type=int, default=None,
                        help="Limit number of slices to evaluate (default: all)")

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
