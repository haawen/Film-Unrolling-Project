"""
Multi-angle ray-casting winding detector.

For each ray from the spool center, traverse outward and identify connected
runs of EMULSION class. Each emulsion run corresponds to one winding's
emulsion layer (the thin brighter sub-band in the segmentation).

Counting emulsion runs is more robust than counting film→air transitions
because:
  - Air gaps between windings can vanish (touching windings) → no air to
    transition through, so transition-counting misses those windings.
  - Each winding still has its own emulsion sub-band even when touching
    its neighbour (separated by film_base of the adjacent winding).
  - Pinch points in the segmentation only affect a small angular range,
    so median voting across many angles smooths them out.

Output:
  n_windings           — mode of per-ray emulsion-run counts
  boundary_radii       — median radius of the k-th emulsion run, k=0..n-1
  per_ray_counts       — distribution of counts (for diagnostics)

Usage:
    python -m unwrapping.inr.winding_raycast <vol.h5> <probs.h5>
        [--n-rays 720] [--out DIR] [--center-yx Y X]
"""

import argparse
import json
import os

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_seg_slice(probs_path, z=None):
    with h5py.File(probs_path, "r") as f:
        probs = f["exported_data"]
        if z is None:
            z = probs.shape[0] // 2
        p = probs[z]
    seg = np.argmax(p, axis=-1).astype(np.uint8)
    return seg, z


def detect_center(seg):
    ys, xs = np.where(seg > 0)
    return float(ys.mean()), float(xs.mean())


def cast_rays(seg, cy, cx, n_rays=720, r_max=None):
    """For each ray angle θ_k = 2πk/n_rays, traverse outward and return
    a list of segmentation values along the ray.

    Returns a list of arrays (length n_rays), each shape (R,) where R is
    the number of pixel samples from r=0 to r=r_max along that ray.
    """
    H, W = seg.shape
    if r_max is None:
        r_max = int(min(cy, cx, H - cy, W - cx)) - 2
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    radii = np.arange(r_max)
    samples = np.empty((n_rays, r_max), dtype=np.uint8)
    for k, theta in enumerate(angles):
        ys = (cy + radii * np.sin(theta)).astype(np.int32)
        xs = (cx + radii * np.cos(theta)).astype(np.int32)
        ys = np.clip(ys, 0, H - 1)
        xs = np.clip(xs, 0, W - 1)
        samples[k] = seg[ys, xs]
    return angles, radii, samples


def emulsion_runs(ray_seg, min_run=2):
    """Find connected runs of emulsion class (=2) in a 1D ray.

    Returns list of (r_start, r_end, r_center) tuples in pixel units.
    Runs shorter than `min_run` are dropped (segmentation noise).
    """
    is_emul = (ray_seg == 2).astype(np.int32)
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(is_emul):
        if v and not in_run:
            start = i
            in_run = True
        elif not v and in_run:
            length = i - start
            if length >= min_run:
                runs.append((start, i - 1, (start + i - 1) / 2.0))
            in_run = False
    if in_run:
        length = len(is_emul) - start
        if length >= min_run:
            runs.append((start, len(is_emul) - 1,
                         (start + len(is_emul) - 1) / 2.0))
    return runs


def aggregate_windings(samples, min_run=2):
    """Across all rays, count emulsion runs and compute per-ray run radii.

    Returns:
        per_ray_counts: (n_rays,) int — number of emulsion runs per ray
        per_ray_radii:  list of n_rays lists of float r_centers
        n_windings:     mode of per_ray_counts
        boundary_radii: (n_windings,) median radius of k-th run across rays
    """
    n_rays = samples.shape[0]
    per_ray_counts = np.zeros(n_rays, dtype=np.int32)
    per_ray_radii = []
    for k in range(n_rays):
        runs = emulsion_runs(samples[k], min_run=min_run)
        per_ray_counts[k] = len(runs)
        per_ray_radii.append([r[2] for r in runs])

    # Mode of per-ray counts → expected number of windings
    counts, freqs = np.unique(per_ray_counts, return_counts=True)
    n_windings = int(counts[np.argmax(freqs)])

    # For each winding index 0..n-1, take median of the k-th run's radius
    # across all rays that have at least k+1 runs.
    boundary_radii = np.full(n_windings, np.nan)
    for k in range(n_windings):
        rs = [r[k] for r in per_ray_radii if len(r) > k]
        if rs:
            boundary_radii[k] = float(np.median(rs))

    return per_ray_counts, per_ray_radii, n_windings, boundary_radii


def plot_diagnostic(seg, cy, cx, samples, per_ray_counts,
                    boundary_radii, out_path, n_rays_to_plot=8):
    """3-panel: cross-section + boundary circles + count histogram."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Panel 1: segmentation with overlaid boundary circles
    axes[0].imshow(seg, cmap="gray")
    axes[0].plot(cx, cy, "r+", markersize=15)
    theta = np.linspace(0, 2 * np.pi, 200)
    for k, r in enumerate(boundary_radii):
        if not np.isnan(r):
            xs = cx + r * np.cos(theta)
            ys = cy + r * np.sin(theta)
            axes[0].plot(xs, ys, "-", color="cyan", linewidth=0.5, alpha=0.6)
    axes[0].set_title(f"Segmentation + {len(boundary_radii)} winding "
                      f"boundary circles (median radii)")
    axes[0].set_xlim(0, seg.shape[1])
    axes[0].set_ylim(seg.shape[0], 0)

    # Panel 2: distribution of per-ray emulsion-run counts
    counts, freqs = np.unique(per_ray_counts, return_counts=True)
    axes[1].bar(counts, freqs, width=0.8)
    mode_count = int(counts[np.argmax(freqs)])
    axes[1].axvline(mode_count, color="r", linestyle="--",
                    label=f"mode = {mode_count}")
    axes[1].set_xlabel("emulsion runs per ray")
    axes[1].set_ylabel("number of rays")
    axes[1].set_title(f"Per-ray emulsion-run count distribution")
    axes[1].legend()

    # Panel 3: a few sample rays plotted as 1D (radius vs class)
    rs = np.arange(samples.shape[1])
    n = samples.shape[0]
    sample_idx = np.linspace(0, n - 1, n_rays_to_plot, dtype=int)
    for j, k in enumerate(sample_idx):
        offset = j * 0.4
        axes[2].plot(rs, samples[k] + offset, linewidth=0.6,
                     label=f"θ={k * 360 / n:.0f}°")
    for k, r in enumerate(boundary_radii):
        if not np.isnan(r):
            axes[2].axvline(r, color="r", alpha=0.2, linewidth=0.5)
    axes[2].set_xlabel("radius (px)")
    axes[2].set_ylabel("seg class (offset per ray)")
    axes[2].set_title(f"{n_rays_to_plot} sample rays (red = winding boundary)")
    axes[2].legend(fontsize=6, ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("vol", help="Path to volume_*.h5 (unused, kept for symmetry)")
    p.add_argument("probs", help="Path to volume_*_Probabilities.h5")
    p.add_argument("--z", type=int, default=None)
    p.add_argument("--n-rays", type=int, default=720)
    p.add_argument("--min-run", type=int, default=2,
                   help="Min emulsion-run length in pixels (drops noise)")
    p.add_argument("--center-yx", type=float, nargs=2, default=None,
                   help="Override centroid with explicit (cy cx)")
    p.add_argument("--out", default=".")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.probs} (slice {args.z})...")
    seg, z_used = load_seg_slice(args.probs, args.z)
    print(f"  loaded slice {z_used}: shape={seg.shape}, "
          f"film={int((seg>0).sum())}, emul={int((seg==2).sum())}")

    if args.center_yx:
        cy, cx = args.center_yx
        print(f"  center (override): ({cy:.1f}, {cx:.1f})")
    else:
        cy, cx = detect_center(seg)
        print(f"  center (centroid): ({cy:.1f}, {cx:.1f})")

    print(f"\nCasting {args.n_rays} rays...")
    angles, radii, samples = cast_rays(seg, cy, cx, n_rays=args.n_rays)
    print(f"  samples shape: {samples.shape}, r_max={radii.max()}")

    print("\nAggregating emulsion runs...")
    per_ray_counts, per_ray_radii, n_windings, boundary_radii = \
        aggregate_windings(samples, min_run=args.min_run)
    counts_unique, freqs = np.unique(per_ray_counts, return_counts=True)
    print(f"  per-ray count distribution: "
          f"{dict(zip(counts_unique.tolist(), freqs.tolist()))}")
    print(f"  → n_windings (mode) = {n_windings}")
    print(f"  per-ray count: mean={per_ray_counts.mean():.2f}, "
          f"median={int(np.median(per_ray_counts))}, "
          f"min={int(per_ray_counts.min())}, max={int(per_ray_counts.max())}")
    print(f"\nBoundary radii (median across rays):")
    for k, r in enumerate(boundary_radii):
        n_present = sum(1 for rr in per_ray_radii if len(rr) > k)
        spacing = boundary_radii[k] - boundary_radii[k - 1] if k > 0 else 0.0
        print(f"  winding {k:2d}: r={r:7.1f}px  "
              f"(spacing={spacing:5.1f}, present in {n_present}/{args.n_rays} rays)")

    diag_path = os.path.join(args.out, "raycast_diagnostic.png")
    plot_diagnostic(seg, cy, cx, samples, per_ray_counts,
                    boundary_radii, diag_path)
    print(f"\nDiagnostic saved: {diag_path}")

    summary = {
        "probs": args.probs,
        "slice": int(z_used),
        "center_yx": [float(cy), float(cx)],
        "n_rays": int(args.n_rays),
        "n_windings_mode": int(n_windings),
        "per_ray_count_distribution": dict(
            zip(counts_unique.tolist(), freqs.tolist())
        ),
        "boundary_radii": [float(r) if not np.isnan(r) else None
                           for r in boundary_radii],
    }
    with open(os.path.join(args.out, "raycast_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {os.path.join(args.out, 'raycast_summary.json')}")


if __name__ == "__main__":
    main()
