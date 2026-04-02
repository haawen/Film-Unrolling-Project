"""
Evaluation and strip generation for learned unwrapping (Step 1b).

Loads trained deformation model, queries all film pixels, denormalizes the
predicted (u, v) using saved stats, and generates the unrolled strip image.

Usage:
    python -m unwrapping.inr.unwrap_eval \
        --image path/to/image.h5 \
        --probs path/to/probs.h5 \
        --model-dir results/unwrap_2d \
        --out-dir results/unwrap_2d/eval
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

from unwrapping.inr.unwrap_data import UnwrapDataset
from unwrapping.inr.unwrap_model import DeformationINR
from unwrapping.inr.data import load_2d_slice
from unwrapping.center_detection import find_spool_center


def predict_all(model, dataset, batch_size=65536):
    """Query model at all film pixel coordinates.

    Returns:
        uv: (N, 2) predicted (u, v) — standardized (zero mean, unit variance)
    """
    model.eval()
    all_uv = []

    with torch.no_grad():
        for i in range(0, dataset.n, batch_size):
            coords = dataset.coords[i:i + batch_size]
            uv = model(coords)
            all_uv.append(uv.cpu())

    return torch.cat(all_uv, dim=0).numpy()


def denormalize_uv(uv_std, norm_stats):
    """Convert standardized (u, v) back to raw coordinates.

    Args:
        uv_std: (N, 2) standardized predictions
        norm_stats: dict with u_mean, u_std, v_mean, v_std

    Returns:
        uv_raw: (N, 2) in original coordinate space
    """
    uv_raw = np.zeros_like(uv_std)
    uv_raw[:, 0] = uv_std[:, 0] * norm_stats["u_std"] + norm_stats["u_mean"]
    uv_raw[:, 1] = uv_std[:, 1] * norm_stats["v_std"] + norm_stats["v_mean"]
    return uv_raw


def generate_strip(uv_raw, intensities, seg_labels, n_layers, strip_height=20):
    """Generate the unrolled strip image by scattering film pixel intensities.

    Args:
        uv_raw: (N, 2) raw (u, v) — u in [0, n_layers], v in [0, 1]
        intensities: (N,) pixel intensities
        seg_labels: (N,) segmentation labels
        n_layers: number of spiral layers
        strip_height: pixel height of each layer in the strip

    Returns:
        strip_img, strip_seg, strip_count
    """
    u = uv_raw[:, 0]  # [0, n_layers]
    v = uv_raw[:, 1]  # [0, 1]

    pixels_per_winding = 1440
    strip_W = n_layers * pixels_per_winding
    strip_H = strip_height

    # Map to pixel positions
    col = ((u / n_layers) * strip_W).astype(np.int64)
    row = (v * strip_H).astype(np.int64)

    col = np.clip(col, 0, strip_W - 1)
    row = np.clip(row, 0, strip_H - 1)

    # Scatter with averaging
    strip_img = np.zeros((strip_H, strip_W), dtype=np.float64)
    strip_seg = np.zeros((strip_H, strip_W), dtype=np.float64)
    strip_count = np.zeros((strip_H, strip_W), dtype=np.float64)

    np.add.at(strip_img, (row, col), intensities)
    np.add.at(strip_seg, (row, col), seg_labels)
    np.add.at(strip_count, (row, col), 1.0)

    mask = strip_count > 0
    strip_img[mask] /= strip_count[mask]
    strip_seg[mask] /= strip_count[mask]
    strip_seg = np.round(strip_seg).astype(np.uint8)

    return strip_img, strip_seg, strip_count


def save_visualizations(uv_raw, intensities, seg_labels, image, seg, dataset,
                        strip_img, strip_seg, strip_count, out_dir):
    """Save diagnostic visualizations."""

    # 1. Strip overview
    fig, axes = plt.subplots(3, 1, figsize=(20, 6))

    axes[0].imshow(strip_img, cmap="gray", aspect="auto")
    axes[0].set_title("Unrolled Strip (Intensity)")
    axes[0].set_ylabel("v (thickness)")

    axes[1].imshow(strip_seg, cmap="tab10", vmin=0, vmax=2, aspect="auto")
    axes[1].set_title("Unrolled Strip (Segmentation)")
    axes[1].set_ylabel("v (thickness)")

    coverage = (strip_count > 0).astype(float)
    axes[2].imshow(coverage, cmap="gray", aspect="auto")
    axes[2].set_title(f"Coverage (filled: {100 * coverage.mean():.1f}%)")
    axes[2].set_ylabel("v (thickness)")
    axes[2].set_xlabel("u (along strip)")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "strip.png"), dpi=150)
    plt.close()

    # 2. UV mapping on the original image
    H, W = dataset.image_shape
    u_map = np.full((H, W), np.nan)
    v_map = np.full((H, W), np.nan)

    film_yx = dataset.film_yx.cpu().numpy()
    u_map[film_yx[:, 0], film_yx[:, 1]] = uv_raw[:, 0]
    v_map[film_yx[:, 0], film_yx[:, 1]] = uv_raw[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original Image")

    im1 = axes[1].imshow(u_map, cmap="hsv")
    axes[1].set_title(f"Predicted u (along strip) [{uv_raw[:,0].min():.1f} – {uv_raw[:,0].max():.1f}]")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(v_map, cmap="viridis")
    axes[2].set_title(f"Predicted v (across strip) [{uv_raw[:,1].min():.2f} – {uv_raw[:,1].max():.2f}]")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "uv_mapping.png"), dpi=150)
    plt.close()

    # 3. Detail crop
    n_layers = dataset.n_layers
    pixels_per_winding = strip_img.shape[1] // n_layers
    start_layer = min(5, n_layers - 1)
    end_layer = min(start_layer + 5, n_layers)
    col_start = start_layer * pixels_per_winding
    col_end = end_layer * pixels_per_winding

    fig, ax = plt.subplots(1, 1, figsize=(16, 3))
    crop = strip_img[:, col_start:col_end]
    ax.imshow(crop, cmap="gray", aspect="auto")
    ax.set_title(f"Strip Detail: Layers {start_layer}-{end_layer}")
    ax.set_xlabel("u (along strip)")
    ax.set_ylabel("v (thickness)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "strip_detail.png"), dpi=150)
    plt.close()

    # 4. Coverage statistics
    stats = {
        "strip_shape": list(strip_img.shape),
        "coverage_pct": float(100 * (strip_count > 0).mean()),
        "mean_count_where_filled": float(strip_count[strip_count > 0].mean()),
        "max_count": int(strip_count.max()),
        "n_layers": n_layers,
        "u_range": [float(uv_raw[:, 0].min()), float(uv_raw[:, 0].max())],
        "v_range": [float(uv_raw[:, 1].min()), float(uv_raw[:, 1].max())],
    }

    for k in range(n_layers):
        col_s = k * pixels_per_winding
        col_e = (k + 1) * pixels_per_winding
        layer_cov = (strip_count[:, col_s:col_e] > 0).mean()
        stats[f"layer_{k}_coverage_pct"] = float(100 * layer_cov)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  Strip shape: {strip_img.shape}")
    print(f"  Coverage: {stats['coverage_pct']:.1f}%")
    print(f"  Mean overlap where filled: {stats['mean_count_where_filled']:.1f}")


def evaluate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load config and normalization stats from training
    log_path = os.path.join(args.model_dir, "train_log.json")
    with open(log_path) as f:
        log = json.load(f)
    config = log["config"]
    norm_stats = log["norm_stats"]

    # Load data
    print("Loading data...")
    image, seg = load_2d_slice(args.image, args.probs)

    print("Finding spiral center...")
    center = find_spool_center(seg)

    print("Preparing unwrap dataset...")
    dataset = UnwrapDataset(image, seg, center, device=device)

    # Build and load model
    model = DeformationINR(
        n_fourier=config["n_fourier"],
        sigma=config["sigma"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
    ).to(device)

    ckpt = args.checkpoint
    ckpt_path = os.path.join(args.model_dir, f"{ckpt}.pt")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    print(f"  Loaded checkpoint: {ckpt_path}")

    # Predict standardized (u, v) for all film pixels
    print("Predicting (u, v) for all film pixels...")
    uv_std = predict_all(model, dataset, batch_size=args.batch_size)

    # Denormalize back to raw coordinates
    uv_raw = denormalize_uv(uv_std, norm_stats)
    print(f"  u range: [{uv_raw[:, 0].min():.2f}, {uv_raw[:, 0].max():.2f}] "
          f"(target: [{norm_stats['u_min']:.2f}, {norm_stats['u_max']:.2f}])")
    print(f"  v range: [{uv_raw[:, 1].min():.4f}, {uv_raw[:, 1].max():.4f}] "
          f"(target: [{norm_stats['v_min']:.4f}, {norm_stats['v_max']:.4f}])")

    # Generate strip
    print("Generating strip...")
    intensities = dataset.intensities.cpu().numpy()
    seg_labels = dataset.seg_labels.cpu().numpy()

    # Clip to valid range before strip generation
    uv_clipped = uv_raw.copy()
    uv_clipped[:, 0] = np.clip(uv_raw[:, 0], 0, dataset.n_layers)
    uv_clipped[:, 1] = np.clip(uv_raw[:, 1], 0, 1)

    strip_img, strip_seg, strip_count = generate_strip(
        uv_clipped, intensities, seg_labels, dataset.n_layers,
        strip_height=args.strip_height,
    )

    # Save visualizations and metrics
    print("Saving visualizations...")
    save_visualizations(
        uv_raw, intensities, seg_labels, image, seg, dataset,
        strip_img, strip_seg, strip_count, args.out_dir,
    )

    # Save raw strip
    np.savez_compressed(
        os.path.join(args.out_dir, "strip.npz"),
        intensity=strip_img.astype(np.float32),
        seg=strip_seg,
        count=strip_count.astype(np.int32),
    )
    print("  Saved strip.npz")


def main():
    parser = argparse.ArgumentParser(description="Evaluate unwrapping model")
    parser.add_argument("--image", required=True)
    parser.add_argument("--probs", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--checkpoint", default="best", choices=["best", "last"])
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--strip-height", type=int, default=20)

    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
