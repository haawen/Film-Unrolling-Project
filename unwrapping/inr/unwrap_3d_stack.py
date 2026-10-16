"""
3D unwrapping by stacking per-slice polar parameterization.

For each z-slice across multiple 3D volume chunks:
  1. Compute segmentation → find spiral center → detect layers
  2. Assign u (along-strip) coordinate to each film pixel
  3. Scatter intensity to (u, z) position in output strip

The result is a 2D image where:
  - Horizontal axis (u): position along the unrolled film
  - Vertical axis (z): position across the film width (z-slices)

This should reveal actual film content (frames/images).

Usage:
    python -m unwrapping.inr.unwrap_3d_stack \
        --data-dir path/to/01_Mickey_3d \
        --out-dir results/unwrap_3d_full
"""

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.inr.unwrap_data import (
    compute_polar, assign_layers_polar, compute_initial_uv_fast
)
from unwrapping.center_detection import find_spool_center


def discover_volumes(data_dir):
    """Find all volume + probability file pairs, sorted by z-range.

    Returns:
        list of (vol_path, probs_path, z_start, z_end)
    """
    vol_files = sorted(glob.glob(os.path.join(data_dir, "volume_*[0-9].h5")))
    pairs = []
    for vf in vol_files:
        if "Probabilities" in vf:
            continue
        m = re.search(r"volume_(\d+)-(\d+)\.h5$", vf)
        if not m:
            continue
        z_start, z_end = int(m.group(1)), int(m.group(2))
        probs_path = vf.replace(".h5", "_Probabilities.h5")
        if os.path.exists(probs_path):
            pairs.append((vf, probs_path, z_start, z_end))
        else:
            print(f"  WARNING: no probabilities for {vf}, skipping")

    pairs.sort(key=lambda x: x[2])
    return pairs


def load_volume_chunk(volume_path, probs_path):
    """Load one 3D volume chunk and its segmentation."""
    import h5py

    with h5py.File(volume_path, "r") as f:
        volume = f["volume"][:].astype(np.float32)  # (D, H, W)

    vmin, vmax = volume.min(), volume.max()
    volume = (volume - vmin) / (vmax - vmin + 1e-8)

    with h5py.File(probs_path, "r") as f:
        probs = f["exported_data"][:].astype(np.float32)  # (D, H, W, 3)
    seg = np.argmax(probs, axis=-1).astype(np.int64)  # (D, H, W)

    return volume, seg


def unwrap_slice(image_2d, seg_2d, center, ref_layers=None):
    """Unwrap a single z-slice using polar parameterization."""
    film_yx, polar_r, polar_theta = compute_polar(seg_2d, center)

    if ref_layers is not None:
        n_layers, layer_radii, layer_starts, layer_ends, angles = ref_layers
    else:
        n_layers, layer_radii, layer_starts, layer_ends, angles = \
            assign_layers_polar(seg_2d, center)

    layer_info = (n_layers, layer_radii, layer_starts, layer_ends, angles)

    u, v, valid = compute_initial_uv_fast(
        film_yx, polar_r, polar_theta,
        n_layers, layer_radii, layer_starts, layer_ends, angles
    )

    film_yx_valid = film_yx[valid]
    u_valid = u[valid]
    intensities = image_2d[film_yx_valid[:, 0], film_yx_valid[:, 1]]

    return u_valid, intensities, n_layers, layer_info


def generate_strip_3d(all_u, all_intensities, all_z_global, n_layers,
                      pixels_per_winding=1440):
    """Generate (u, z) strip using global z-indices (preserving gaps).

    The strip height equals (max_z - min_z + 1) so gaps appear as empty rows.
    """
    strip_W = n_layers * pixels_per_winding
    z_min = min(all_z_global)
    z_max = max(all_z_global)
    n_z = z_max - z_min + 1

    strip = np.zeros((n_z, strip_W), dtype=np.float64)
    count = np.zeros((n_z, strip_W), dtype=np.float64)

    for u, intensity, z_global in zip(all_u, all_intensities, all_z_global):
        col = ((u / n_layers) * strip_W).astype(np.int64)
        col = np.clip(col, 0, strip_W - 1)
        row_idx = z_global - z_min
        row = np.full_like(col, row_idx)

        np.add.at(strip, (row, col), intensity)
        np.add.at(count, (row, col), 1.0)

    mask = count > 0
    strip[mask] /= count[mask]

    return strip, count, z_min, z_max


def save_visualizations(strip, count, n_layers, z_min, z_max,
                        all_z_global, out_dir):
    """Save strip visualizations."""
    n_z = strip.shape[0]

    # 1. Full strip
    fig, axes = plt.subplots(2, 1, figsize=(24, max(6, n_z / 30)))

    axes[0].imshow(strip, cmap="gray", aspect="auto", interpolation="nearest")
    axes[0].set_title(f"Unrolled Film Strip ({n_layers} layers × {n_z} z-range, "
                      f"z=[{z_min}-{z_max}])")
    axes[0].set_ylabel(f"z (slice index - {z_min})")
    axes[0].set_xlabel("u (along film)")

    coverage = (count > 0).astype(float)
    filled_rows = len(set(all_z_global))
    total_coverage = coverage.sum() / (filled_rows * strip.shape[1]) * 100 if filled_rows > 0 else 0
    axes[1].imshow(coverage, cmap="gray", aspect="auto", interpolation="nearest")
    axes[1].set_title(f"Coverage ({total_coverage:.1f}% of filled rows)")
    axes[1].set_ylabel(f"z (slice index - {z_min})")
    axes[1].set_xlabel("u (along film)")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "strip_3d.png"), dpi=200)
    plt.close()

    # 2. Detail crops per layer group
    pixels_per_winding = strip.shape[1] // n_layers
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

    # 3. Detail crop: single winding, zoomed in (best chance to see content)
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


def save_chunk_checkpoint(out_dir, chunk_idx, z_start, z_end, n_layers,
                          chunk_u, chunk_intensities, chunk_z_global):
    """Save per-chunk checkpoint to disk."""
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"chunk_{chunk_idx:03d}_z{z_start}-{z_end}.npz")
    np.savez_compressed(
        ckpt_path,
        n_layers=n_layers,
        z_global=np.array(chunk_z_global),
        **{f"u_{i}": u for i, u in enumerate(chunk_u)},
        **{f"int_{i}": inten for i, inten in enumerate(chunk_intensities)},
        n_slices=len(chunk_u),
    )
    print(f"  Checkpoint saved: {ckpt_path}")
    return ckpt_path


def load_chunk_checkpoint(ckpt_path):
    """Load a per-chunk checkpoint."""
    data = np.load(ckpt_path)
    n_layers = int(data["n_layers"])
    z_global = list(data["z_global"])
    n_slices = int(data["n_slices"])
    chunk_u = [data[f"u_{i}"] for i in range(n_slices)]
    chunk_intensities = [data[f"int_{i}"] for i in range(n_slices)]
    return n_layers, chunk_u, chunk_intensities, z_global


def find_completed_chunks(out_dir):
    """Find which chunks already have checkpoints."""
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        return {}
    completed = {}
    for f in sorted(os.listdir(ckpt_dir)):
        if f.startswith("chunk_") and f.endswith(".npz"):
            m = re.match(r"chunk_(\d+)_z(\d+)-(\d+)\.npz", f)
            if m:
                idx = int(m.group(1))
                completed[idx] = os.path.join(ckpt_dir, f)
    return completed


def main():
    parser = argparse.ArgumentParser(description="3D unwrapping via per-slice stacking")
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing volume_*.h5 and *_Probabilities.h5 files")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--pixels-per-winding", type=int, default=1440,
                        help="u-resolution per full 360° winding")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Discover all volume chunks
    print("Discovering volume files...")
    pairs = discover_volumes(args.data_dir)
    if not pairs:
        print("ERROR: No volume files found!")
        sys.exit(1)

    total_slices = sum(z_end - z_start + 1 for _, _, z_start, z_end in pairs)
    print(f"  Found {len(pairs)} volume chunks, {total_slices} total slices")
    for vf, _, z_start, z_end in pairs:
        print(f"    z=[{z_start}-{z_end}] ({z_end - z_start + 1} slices) — {os.path.basename(vf)}")

    # Check for existing checkpoints (resume support)
    completed = find_completed_chunks(args.out_dir)
    if completed:
        print(f"\n  Found {len(completed)} completed chunk checkpoints — resuming")

    # Process all chunks
    t0 = time.time()
    all_u = []
    all_intensities = []
    all_z_global = []
    n_layers = None
    slices_processed = 0

    for chunk_idx, (vol_path, probs_path, z_start, z_end) in enumerate(pairs):
        print(f"\n═══ Chunk {chunk_idx + 1}/{len(pairs)}: z=[{z_start}-{z_end}] ═══")

        # Resume: load from checkpoint if available
        if chunk_idx in completed:
            print(f"  Loading from checkpoint...")
            t_load = time.time()
            nl, chunk_u, chunk_int, chunk_z = load_chunk_checkpoint(completed[chunk_idx])
            if n_layers is None:
                n_layers = nl
            all_u.extend(chunk_u)
            all_intensities.extend(chunk_int)
            all_z_global.extend(chunk_z)
            slices_processed += len(chunk_u)
            print(f"  Restored {len(chunk_u)} slices in {time.time() - t_load:.1f}s")
            continue

        t_chunk = time.time()
        volume, seg = load_volume_chunk(vol_path, probs_path)
        D = volume.shape[0]
        print(f"  Loaded {D} slices in {time.time() - t_chunk:.1f}s")

        chunk_u = []
        chunk_intensities = []
        chunk_z_global = []

        for local_z in range(D):
            t1 = time.time()
            global_z = z_start + local_z
            image_2d = volume[local_z]
            seg_2d = seg[local_z]

            center = find_spool_center(seg_2d)

            u, intensity, nl, layer_info = unwrap_slice(image_2d, seg_2d, center)
            if n_layers is None:
                n_layers = nl

            chunk_u.append(u)
            chunk_intensities.append(intensity)
            chunk_z_global.append(global_z)

            elapsed = time.time() - t1
            film_pixels = (seg_2d > 0).sum()
            print(f"  z={global_z:4d}: {len(u):,} valid / {film_pixels:,} film "
                  f"({100 * len(u) / max(film_pixels, 1):.1f}%) | {elapsed:.1f}s")

        # Free memory after each chunk
        del volume, seg

        # Save checkpoint
        save_chunk_checkpoint(args.out_dir, chunk_idx, z_start, z_end,
                              n_layers, chunk_u, chunk_intensities, chunk_z_global)

        all_u.extend(chunk_u)
        all_intensities.extend(chunk_intensities)
        all_z_global.extend(chunk_z_global)
        slices_processed += len(chunk_u)

    # Generate strip
    print(f"\n═══ Generating strip ({n_layers} layers × {slices_processed} slices) ═══")
    strip, count, z_min, z_max = generate_strip_3d(
        all_u, all_intensities, all_z_global, n_layers,
        pixels_per_winding=args.pixels_per_winding,
    )
    print(f"  Strip shape: {strip.shape} (z range {z_min}-{z_max}, "
          f"{z_max - z_min + 1} rows, {slices_processed} filled)")

    # Save
    print("Saving results...")
    np.savez_compressed(
        os.path.join(args.out_dir, "strip_3d.npz"),
        strip=strip.astype(np.float32),
        count=count.astype(np.int32),
        z_min=z_min,
        z_max=z_max,
        z_slices=np.array(all_z_global),
    )

    save_visualizations(strip, count, n_layers, z_min, z_max,
                        all_z_global, args.out_dir)

    # Metrics
    filled_rows = len(set(all_z_global))
    filled_coverage = sum(
        (count[z - z_min, :] > 0).mean() for z in all_z_global
    ) / filled_rows * 100

    stats = {
        "strip_shape": list(strip.shape),
        "n_layers": n_layers,
        "n_volume_chunks": len(pairs),
        "n_slices_processed": slices_processed,
        "z_range": [z_min, z_max],
        "z_range_total": z_max - z_min + 1,
        "z_gaps": z_max - z_min + 1 - slices_processed,
        "coverage_filled_rows_pct": float(filled_coverage),
        "pixels_per_winding": args.pixels_per_winding,
    }

    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(stats, f, indent=2)

    total_time = time.time() - t0
    print(f"\nDone in {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"  Strip shape: {strip.shape}")
    print(f"  Coverage (filled rows): {filled_coverage:.1f}%")
    print(f"  Z-slices: {slices_processed} filled / {z_max - z_min + 1} total range")
    print(f"  Results: {args.out_dir}")


if __name__ == "__main__":
    main()
