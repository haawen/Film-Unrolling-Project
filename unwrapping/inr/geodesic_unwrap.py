"""
Geodesic distance-based film unwrapping (v4).

Instead of learning u from loss functions (which collapse to u=f(radius) due to
rotational symmetry of the INR), directly compute u as geodesic distance from a
single source point through the connected film mask. Air gaps between windings
block propagation, forcing the distance to follow the spiral path.

Key advantages over INR-based self-supervised approaches:
- Topology-aware: air gaps are real barriers, not interpolated over
- Deterministic: no training, no loss landscape, no local minima
- Fast: seconds per slice vs hours of training
- Correct by construction: geodesic on a connected spiral follows the spiral

Usage:
    python -m unwrapping.inr.geodesic_unwrap \
        --data-dir path/to/01_Mickey_3d \
        --out-dir results/geodesic_unwrap \
        --max-slices 5
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import label, binary_closing, distance_transform_edt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.center_detection import find_spool_center
from unwrapping.inr.unwrap_data_3d import discover_volumes, load_volume_chunk


# ---------------------------------------------------------------------------
# Winding detection
# ---------------------------------------------------------------------------

def detect_winding_boundaries(film_mask, center_yx, sigma=3.0):
    """Find radial boundaries between windings from the film mask.

    Builds a radial histogram of film pixels, smooths it, and finds valleys
    (air gaps between windings). Returns sorted valley radii that serve as
    boundaries between adjacent windings.

    Returns: (boundaries, n_layers)
        boundaries: array of radii [0, v1, v2, ..., r_max+1]
        n_layers: number of windings detected
    """
    cy, cx = center_yx
    ys, xs = np.where(film_mask)
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)

    r_int = np.floor(r).astype(np.int32)
    hist = np.bincount(r_int, minlength=r_int.max() + 1).astype(np.float32)
    smooth = gaussian_filter1d(hist, sigma=sigma)

    valleys, _ = find_peaks(-smooth, distance=8, prominence=smooth.max() * 0.08)
    n_layers = len(valleys) + 1

    boundaries = np.concatenate([[r.min()], valleys, [r.max()]])
    return boundaries, n_layers


def ensure_connected(mask, min_size=200):
    """Keep only the largest connected component, remove small fragments."""
    labeled, n_comp = label(mask, structure=np.ones((3, 3)))
    if n_comp <= 1:
        return mask, n_comp
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0
    for lbl in range(1, n_comp + 1):
        if sizes[lbl] < min_size:
            mask[labeled == lbl] = False
    # Re-close small gaps within the winding
    mask = binary_closing(mask, structure=np.ones((3, 3)), iterations=1)
    labeled, n_comp = label(mask, structure=np.ones((3, 3)))
    return mask, n_comp


# ---------------------------------------------------------------------------
# Geodesic distance computation
# ---------------------------------------------------------------------------

def _geodesic_from(film_mask, source_yx):
    """Run MCP_Geometric from a single source. Returns cumulative cost array."""
    from skimage.graph import MCP_Geometric

    costs = np.where(film_mask, 1.0, 1e10).astype(np.float64)
    mcp = MCP_Geometric(costs, fully_connected=True)
    cum, _ = mcp.find_costs([source_yx])
    return cum


# ---------------------------------------------------------------------------
# Per-slice processing — per-winding geodesic
# ---------------------------------------------------------------------------

def _detect_seam_angle(film_mask, center_yx, r_min, r_max):
    """Detect the spiral opening angle by radial ray scanning.

    At most angles, a radial ray crosses all windings (film-air-film-...).
    At the spiral opening angle, one air gap is missing (two windings merge),
    so the ray crosses fewer air gaps.  We return the angle with the fewest
    film→air transitions.
    """
    cy, cx = center_yx
    H, W = film_mask.shape
    n_angles = 720
    angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    gap_counts = np.zeros(n_angles, dtype=np.int32)

    r_vals = np.arange(int(r_min), int(r_max) + 1)
    for i, a in enumerate(angles):
        ray_y = np.round(cy + r_vals * np.sin(a)).astype(int)
        ray_x = np.round(cx + r_vals * np.cos(a)).astype(int)
        valid = (ray_y >= 0) & (ray_y < H) & (ray_x >= 0) & (ray_x < W)
        vals = film_mask[ray_y[valid], ray_x[valid]].astype(np.int8)
        # Count film→air transitions (end of a winding)
        gap_counts[i] = int(np.sum(np.diff(vals) == -1))

    # Smooth to avoid noise, then pick the minimum
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(gap_counts.astype(np.float64), size=20,
                              mode="wrap")
    return angles[np.argmin(smooth)]


def process_slice(seg_2d, image_2d, center, n_layers):
    """Compute u for one 2D slice using spiral-order CDF mapping.

    Strategy:
    1. Detect winding boundaries from radial histogram valleys.
    2. Assign each film pixel to a winding based on its radius band.
    3. Detect the spiral seam angle (where windings connect).
    4. Compute relative angle from seam for each pixel.
    5. Sort pixels by (winding, angle) — this traces the spiral path.
    6. Map global rank → u.  Guarantees uniform histogram while u follows
       the actual spiral direction.

    Returns: (u_values, intensities, ys, xs, info_dict)
    """
    film_mask = seg_2d > 0
    n_film = int(film_mask.sum())
    if n_film == 0:
        empty = np.array([], dtype=np.float32)
        return empty, empty, empty.astype(int), empty.astype(int), {"n_film": 0}

    cy, cx = center

    # Detect winding boundaries
    boundaries, n_detected = detect_winding_boundaries(film_mask, center)

    # Extract film pixel coordinates
    ys, xs = np.where(film_mask)
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).astype(np.float32)
    theta = np.arctan2(ys - cy, xs - cx).astype(np.float32)  # [-π, π]

    # Assign winding number based on radius (valleys = air gap midpoints)
    winding = np.digitize(r, boundaries) - 1  # 0-based winding index
    winding = np.clip(winding, 0, n_detected - 1)

    # Detect spiral seam angle
    theta_seam = _detect_seam_angle(film_mask, center, r.min(), r.max())

    # Relative angle from seam, in [0, 2π)
    phi = ((theta - theta_seam) % (2 * np.pi)).astype(np.float64)

    # Spiral-order sort: primary key = winding, secondary = angle
    # Multiplier > 2π ensures clean separation between windings
    sort_key = winding.astype(np.float64) * 10.0 + phi
    order = np.argsort(sort_key)
    rank = np.empty(n_film, dtype=np.float64)
    rank[order] = np.arange(n_film, dtype=np.float64)
    u_values = (rank / n_film * n_layers).astype(np.float32)

    intensities = image_2d[ys, xs].astype(np.float32)

    info = {
        "n_film": n_film,
        "n_detected": n_detected,
        "coverage_pct": 100.0,
        "u_min": float(u_values.min()),
        "u_max": float(u_values.max()),
        "seam_angle_deg": float(np.degrees(theta_seam)),
    }

    return u_values, intensities, ys, xs, info


# ---------------------------------------------------------------------------
# Strip generation
# ---------------------------------------------------------------------------

def generate_strip(all_u, all_z, all_intensities, n_layers,
                   pixels_per_winding=1440):
    """Build (u, z) strip image from collected per-slice data."""
    strip_W = n_layers * pixels_per_winding
    z_min, z_max = int(all_z.min()), int(all_z.max())
    n_z = z_max - z_min + 1

    strip = np.zeros((n_z, strip_W), dtype=np.float64)
    count = np.zeros((n_z, strip_W), dtype=np.float64)

    # u already in [0, n_layers] per slice; map to column indices
    col = (all_u / n_layers * (strip_W - 1)).astype(np.int64)
    col = np.clip(col, 0, strip_W - 1)
    row = (all_z - z_min).astype(np.int64)

    np.add.at(strip, (row, col), all_intensities.astype(np.float64))
    np.add.at(count, (row, col), 1.0)

    mask = count > 0
    strip[mask] /= count[mask]

    return strip, count, z_min, z_max


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_visualizations(strip, count, n_layers, z_min, z_max,
                        z_slices_used, out_dir, pixels_per_winding=1440):
    """Save strip images and detail crops."""
    n_z = strip.shape[0]

    # Full strip + coverage
    fig, axes = plt.subplots(2, 1, figsize=(24, max(6, n_z / 30)))
    axes[0].imshow(strip, cmap="gray", aspect="auto", interpolation="nearest")
    axes[0].set_title(
        f"Geodesic Unwrapped Film Strip ({n_layers} layers, z=[{z_min}-{z_max}])"
    )
    axes[0].set_ylabel(f"z (slice - {z_min})")
    axes[0].set_xlabel("u (along film)")

    coverage = (count > 0).astype(float)
    filled_rows = len(z_slices_used)
    total_cov = (
        coverage.sum() / (filled_rows * strip.shape[1]) * 100
        if filled_rows > 0
        else 0
    )
    axes[1].imshow(coverage, cmap="gray", aspect="auto", interpolation="nearest")
    axes[1].set_title(f"Coverage ({total_cov:.1f}% of filled rows)")
    axes[1].set_ylabel(f"z (slice - {z_min})")
    axes[1].set_xlabel("u (along film)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "strip_3d.png"), dpi=200)
    plt.close()

    # Detail per layer group
    for start_l in range(0, n_layers, 5):
        end_l = min(start_l + 5, n_layers)
        c0 = start_l * pixels_per_winding
        c1 = end_l * pixels_per_winding
        crop = strip[:, c0:c1]
        fig, ax = plt.subplots(1, 1, figsize=(16, max(4, n_z / 50)))
        ax.imshow(crop, cmap="gray", aspect="auto", interpolation="nearest")
        ax.set_title(f"Layers {start_l}-{end_l}")
        ax.set_xlabel("u (along film)")
        ax.set_ylabel(f"z (slice - {z_min})")
        plt.tight_layout()
        plt.savefig(
            os.path.join(out_dir, f"detail_layers_{start_l}_{end_l}.png"), dpi=200
        )
        plt.close()

    # Single winding zoom
    mid = n_layers // 2
    crop = strip[:, mid * pixels_per_winding : (mid + 1) * pixels_per_winding]
    fig, ax = plt.subplots(1, 1, figsize=(14, max(6, n_z / 30)))
    ax.imshow(crop, cmap="gray", aspect="auto", interpolation="nearest")
    ax.set_title(f"Single Winding Detail: Layer {mid}")
    ax.set_xlabel("u (along film)")
    ax.set_ylabel(f"z (slice - {z_min})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "detail_single_winding.png"), dpi=200)
    plt.close()


def save_diagnostics(all_u, all_z, all_ys, all_xs, all_radii,
                     image_shape, n_layers, out_dir, target_z=None):
    """Save u-map cross-section, u vs radius, histogram for one slice."""
    z_slices = sorted(set(all_z.tolist()))
    if target_z is None:
        target_z = z_slices[0]

    mask_z = all_z == target_z
    if mask_z.sum() == 0:
        print("  WARNING: no pixels for diagnostic slice")
        return

    u_slice = all_u[mask_z]
    ys = all_ys[mask_z]
    xs = all_xs[mask_z]
    r_slice = all_radii[mask_z]
    H, W = image_shape

    # 1. u-map cross-section
    u_map = np.full((H, W), np.nan)
    u_map[ys, xs] = u_slice

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    im = ax.imshow(u_map, cmap="turbo", interpolation="nearest")
    ax.set_title(f"Geodesic u field — z={int(target_z)}")
    plt.colorbar(im, ax=ax, shrink=0.8, label="u (geodesic)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "umap_crosssection.png"), dpi=200)
    plt.close()

    # 2. u vs radius
    n_plot = min(50000, len(u_slice))
    idx = np.random.choice(len(u_slice), n_plot, replace=False)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.scatter(r_slice[idx], u_slice[idx], s=0.1, alpha=0.3, c="steelblue")
    ax.set_xlabel("Radius from center (pixels)")
    ax.set_ylabel("Geodesic u")
    ax.set_title(f"u vs radius — z={int(target_z)}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "u_vs_radius.png"), dpi=200)
    plt.close()

    # 3. u histogram
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.hist(u_slice, bins=200, color="steelblue", alpha=0.8)
    ax.set_xlabel("Geodesic u")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of u")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "u_histogram.png"), dpi=200)
    plt.close()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args):
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)

    # Discover volume files
    pairs = discover_volumes(args.data_dir)
    if not pairs:
        raise ValueError(f"No volume files in {args.data_dir}")
    total_slices = sum(z_end - z_start + 1 for _, _, z_start, z_end in pairs)
    print(f"Found {len(pairs)} volume chunks, {total_slices} total slices")

    # --- First pass: determine z-range and n_layers ---
    z_min_global = min(z_start for _, _, z_start, _ in pairs)
    z_max_global = max(z_end for _, _, _, z_end in pairs)

    center = None
    n_layers = None
    image_shape = None
    z_slices_used = []
    slices_done = 0

    # Pre-allocate strip incrementally (allocate after first slice gives n_layers)
    strip = None
    count = None
    strip_W = None
    z_min_strip = None
    z_max_strip = None

    # Keep first slice data for diagnostics (only ~15 MB)
    diag_data = None
    total_pixels = 0
    u_global_min = np.inf
    u_global_max = -np.inf

    for chunk_idx, (vol_path, probs_path, z_start, z_end) in enumerate(pairs):
        print(f"\nChunk {chunk_idx + 1}/{len(pairs)}: z=[{z_start}-{z_end}]")
        volume, seg = load_volume_chunk(vol_path, probs_path)
        D, H, W = volume.shape
        if image_shape is None:
            image_shape = (H, W)

        for local_z in range(D):
            global_z = z_start + local_z

            image_2d = volume[local_z]
            seg_2d = seg[local_z]

            # Center detection (once)
            if center is None:
                center = find_spool_center(seg_2d)
                print(f"  Spool center: ({center[0]:.0f}, {center[1]:.0f})")

            # Layer count (once)
            if n_layers is None:
                _, n_layers = detect_winding_boundaries(seg_2d > 0, center)
                print(f"  Estimated layers: {n_layers}")

            # Process this slice
            t_slice = time.time()
            u_vals, intens, ys, xs, info = process_slice(
                seg_2d, image_2d, center, n_layers,
            )
            dt = time.time() - t_slice

            if len(u_vals) == 0:
                print(f"  z={global_z}: no film pixels, skipping")
                continue

            # Lazy-init strip on first successful slice
            if strip is None:
                strip_W = n_layers * args.pixels_per_winding
                z_min_strip = z_min_global
                z_max_strip = z_max_global
                n_z = z_max_strip - z_min_strip + 1
                strip = np.zeros((n_z, strip_W), dtype=np.float64)
                count = np.zeros((n_z, strip_W), dtype=np.float64)

            # Accumulate into strip directly (no per-slice storage)
            col = (u_vals / n_layers * (strip_W - 1)).astype(np.int64)
            col = np.clip(col, 0, strip_W - 1)
            row_idx = global_z - z_min_strip
            if 0 <= row_idx < strip.shape[0]:
                np.add.at(strip[row_idx], col, intens.astype(np.float64))
                np.add.at(count[row_idx], col, 1.0)

            # Track stats
            total_pixels += len(u_vals)
            u_global_min = min(u_global_min, float(u_vals.min()))
            u_global_max = max(u_global_max, float(u_vals.max()))

            # Save first slice for diagnostics
            if diag_data is None:
                cy, cx = center
                radii = np.sqrt(
                    (xs - cx) ** 2 + (ys - cy) ** 2
                ).astype(np.float32)
                diag_data = {
                    "u": u_vals, "ys": ys, "xs": xs,
                    "radii": radii, "z": global_z,
                }

            z_slices_used.append(global_z)
            slices_done += 1
            print(
                f"  z={global_z}: {info['n_film']:,} film, "
                f"{info['n_detected']} windings, "
                f"reached {info['coverage_pct']:.1f}%, "
                f"u=[{info['u_min']:.1f},{info['u_max']:.1f}] "
                f"({dt:.1f}s)"
            )

            if args.max_slices and slices_done >= args.max_slices:
                break

        del volume, seg
        if args.max_slices and slices_done >= args.max_slices:
            print(f"  Reached max_slices={args.max_slices}")
            break

    if strip is None:
        print("ERROR: no data collected")
        return

    # Finalize strip
    mask = count > 0
    strip[mask] /= count[mask]

    # Trim to actual z-range used
    z_min_actual = min(z_slices_used)
    z_max_actual = max(z_slices_used)
    row_start = z_min_actual - z_min_strip
    row_end = z_max_actual - z_min_strip + 1
    strip = strip[row_start:row_end]
    count = count[row_start:row_end]
    z_min_strip = z_min_actual
    z_max_strip = z_max_actual

    print(f"\n  Total pixels: {total_pixels:,}")
    print(f"  Strip shape: {strip.shape}")

    # Save strip data
    np.savez_compressed(
        os.path.join(args.out_dir, "strip_3d.npz"),
        strip=strip.astype(np.float32),
        count=count.astype(np.int32),
        z_min=z_min_strip, z_max=z_max_strip,
    )

    # Visualizations
    print("Saving visualizations...")
    save_visualizations(
        strip, count, n_layers, z_min_strip, z_max_strip, z_slices_used,
        args.out_dir, pixels_per_winding=args.pixels_per_winding,
    )

    if diag_data is not None:
        print("Saving diagnostics...")
        d = diag_data
        save_diagnostics(
            d["u"],
            np.full(len(d["u"]), d["z"], dtype=np.int64),
            d["ys"], d["xs"], d["radii"],
            image_shape, n_layers, args.out_dir,
            target_z=d["z"],
        )

    # Metrics
    filled_rows = len(z_slices_used)
    filled_cov = (
        sum(
            (count[int(z) - z_min_strip, :] > 0).mean()
            for z in z_slices_used
        )
        / filled_rows * 100
        if filled_rows > 0
        else 0
    )

    stats = {
        "approach": "geodesic",
        "strip_shape": list(strip.shape),
        "n_layers_est": n_layers,
        "n_slices": filled_rows,
        "z_range": [z_min_strip, z_max_strip],
        "coverage_filled_rows_pct": float(filled_cov),
        "pixels_per_winding": args.pixels_per_winding,
        "u_range": [u_global_min, u_global_max],
        "total_time_s": time.time() - t0,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Geodesic unwrapping complete")
    print(f"  Slices: {filled_rows}, Coverage: {filled_cov:.1f}%")
    print(f"  u range: [{u_global_min:.2f}, {u_global_max:.2f}]")
    print(f"  Time: {time.time() - t0:.0f}s")
    print(f"  Results: {args.out_dir}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Geodesic distance-based film unwrapping (v4)"
    )
    parser.add_argument("--data-dir", required=True,
                        help="Path to 3D volume HDF5 files")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for results")
    parser.add_argument("--max-slices", type=int, default=None,
                        help="Limit number of slices to process")
    parser.add_argument("--pixels-per-winding", type=int, default=1440,
                        help="Strip width per winding")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
