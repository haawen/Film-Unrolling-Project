"""
Standalone winding detector via connected components.

Compares two strategies for counting/locating spiral windings in CT data:

  1. Radial histogram + valley peak-finding (current method in
     surface_data.detect_winding_boundaries_normalized). Misses inner
     valleys when the segmentation is noisy / pinch-points connect
     adjacent windings.

  2. CC labeling on eroded film mask. Each surviving connected component
     after erosion is one winding. More robust to pinch points (erosion
     breaks the 1-2px bridges between windings) and gives a per-pixel
     winding label, not just a 1D histogram.

Usage:
    python -m unwrapping.inr.winding_detect <volume.h5> <probs.h5> [--out DIR]

Reports per slice (and aggregated across slices):
    n_windings_hist      from histogram detector
    n_windings_cc_eN     from CC with erode_iters=N (sweeps 1..6)
    centerline_radii     mean radius per CC
    coverage_per_winding pixels per CC

Saves diagnostics:
    detection_compare.png  — histogram + CC overlays for a single slice
    summary.json           — per-slice counts for all strategies
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import (
    binary_erosion, distance_transform_edt, gaussian_filter1d,
    label as cc_label,
)
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def load_seg_slice(vol_path, probs_path, z=None):
    """Load a single segmentation slice from probability HDF5.

    Returns the argmax of the 3-class probability map (0=air, 1=base, 2=emul).
    If z is None, picks the middle slice.
    """
    with h5py.File(probs_path, "r") as f:
        # exported_data shape: (n_z, H, W, 3)
        if "exported_data" not in f.keys():
            raise KeyError(f"{probs_path} has no 'exported_data' key")
        n_z = f["exported_data"].shape[0]
        if z is None:
            z = n_z // 2
        probs = f["exported_data"][z]  # (H, W, 3)
    seg = np.argmax(probs, axis=-1).astype(np.uint8)
    return seg, z


def detect_center(seg):
    """Centroid of the entire film mask. Quick & dirty."""
    film = seg > 0
    ys, xs = np.where(film)
    if len(ys) == 0:
        H, W = seg.shape
        return H / 2.0, W / 2.0
    return float(ys.mean()), float(xs.mean())


def histogram_windings(seg, cy, cx, prominence_frac=0.03):
    """Histogram-based winding detector (matches surface_data.detect_winding_boundaries_normalized)."""
    film = seg > 0
    H, W = seg.shape
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_film = r[film]
    if len(r_film) == 0:
        return 0, np.array([]), None, None

    r_max = int(r_film.max() + 2)
    bins = np.arange(0, r_max + 1)
    hist, edges = np.histogram(r_film, bins=bins)
    # circumference normalization
    centers = (edges[:-1] + edges[1:]) / 2.0
    norm = hist.astype(np.float64) / np.maximum(2 * np.pi * centers, 1.0)
    norm = norm / max(norm.max(), 1e-9)
    smooth = gaussian_filter1d(norm, sigma=2.0)
    valleys, _ = find_peaks(
        -smooth, distance=8, prominence=prominence_frac * smooth.max()
    )
    n_windings = len(valleys) + 1
    return n_windings, valleys, centers, smooth


def cc_windings(seg, cy, cx, erode_iters=2, min_pixels=8, mask_class=None):
    """CC-based winding detector. Returns (n_windings, mean_radii, sizes).

    mask_class: None=film (class>0), 1=base only, 2=emulsion only.
    Emulsion is thinner so less prone to pinch-points; recommended.
    """
    if mask_class is None:
        m = (seg > 0)
    else:
        m = (seg == mask_class)
    if not m.any():
        return 0, [], []
    if erode_iters > 0:
        eroded = binary_erosion(m, iterations=erode_iters)
        if not eroded.any():
            return 0, [], []
    else:
        eroded = m
    cc, n_cc = cc_label(eroded)
    if n_cc == 0:
        return 0, [], []
    yy, xx = np.mgrid[0:seg.shape[0], 0:seg.shape[1]]
    comps = []
    for c in range(1, n_cc + 1):
        m = (cc == c)
        sz = int(m.sum())
        if sz < min_pixels:
            continue
        r = float(np.sqrt(
            (yy[m] - cy) ** 2 + (xx[m] - cx) ** 2
        ).mean())
        comps.append((r, sz, c))
    comps.sort(key=lambda t: t[0])
    radii = [c[0] for c in comps]
    sizes = [c[1] for c in comps]
    return len(comps), radii, sizes


def cc_label_map(seg, cy, cx, erode_iters, mask_class=None):
    """Return a 2D winding-label map (-1 = no label)."""
    if mask_class is None:
        m = (seg > 0)
    else:
        m = (seg == mask_class)
    if erode_iters > 0:
        eroded = binary_erosion(m, iterations=erode_iters)
    else:
        eroded = m
    cc, n_cc = cc_label(eroded)
    yy, xx = np.mgrid[0:seg.shape[0], 0:seg.shape[1]]
    comps = []
    for c in range(1, n_cc + 1):
        m = (cc == c)
        if m.sum() < 8:
            continue
        r = float(np.sqrt(
            (yy[m] - cy) ** 2 + (xx[m] - cx) ** 2
        ).mean())
        comps.append((r, c))
    comps.sort(key=lambda t: t[0])
    label_2d = np.full(seg.shape, -1, dtype=np.int32)
    for new_idx, (_, c) in enumerate(comps):
        label_2d[cc == c] = new_idx
    return label_2d, len(comps)


def make_diagnostic(seg, cy, cx, hist_result, cc_results, out_path):
    """Side-by-side: segmentation + histogram + CC label map."""
    n_w_hist, valleys, centers, smooth = hist_result
    # Pick the erode_iters whose n_windings is largest in cc_results
    # (most interesting result, likely the one closest to the true count).
    erode_for_plot = max(cc_results.keys(), key=lambda e: cc_results[e]["n_w"])
    label_2d = cc_results[erode_for_plot]["label_2d"]
    n_w_cc = cc_results[erode_for_plot]["n_w"]

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    axes[0].imshow(seg, cmap="gray")
    axes[0].plot(cx, cy, "r+", markersize=15)
    axes[0].set_title(f"Segmentation (slice center ({cy:.0f},{cx:.0f}))")

    if smooth is not None:
        axes[1].plot(centers, smooth, "b-", linewidth=0.8, label="film fraction (smoothed)")
        for v in valleys:
            axes[1].axvline(centers[v], color="r", alpha=0.4, linewidth=0.5)
        axes[1].set_xlabel("radius (px)")
        axes[1].set_ylabel("normalized film fraction")
        axes[1].set_title(f"Histogram detector → {n_w_hist} windings")
        axes[1].legend(fontsize=8)

    if label_2d is not None:
        # Show only valid labels with turbo colormap
        masked = np.where(label_2d >= 0, label_2d, np.nan)
        im = axes[2].imshow(masked, cmap="turbo", interpolation="nearest")
        axes[2].plot(cx, cy, "k+", markersize=15)
        axes[2].set_title(f"CC detector (erode={erode_for_plot}) → {n_w_cc} windings")
        plt.colorbar(im, ax=axes[2], fraction=0.046, label="winding index")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("vol", help="Path to volume_*.h5")
    p.add_argument("probs", help="Path to volume_*_Probabilities.h5")
    p.add_argument("--z", type=int, default=None,
                   help="Slice index (default: middle)")
    p.add_argument("--out", default=".",
                   help="Output directory for diagnostics")
    p.add_argument("--erode-sweep", type=str, default="0,1,2,3,4,6",
                   help="Comma-separated erode_iters values to sweep")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {args.probs} (slice {args.z})...")
    seg, z_used = load_seg_slice(args.vol, args.probs, args.z)
    print(f"  loaded slice {z_used}: shape={seg.shape}, "
          f"film_pixels={int((seg>0).sum())}, "
          f"emul_pixels={int((seg==2).sum())}")

    cy, cx = detect_center(seg)
    print(f"  centroid: ({cy:.1f}, {cx:.1f})")

    print("\n=== Histogram detector ===")
    hist_result = histogram_windings(seg, cy, cx)
    n_w_hist = hist_result[0]
    print(f"  n_windings = {n_w_hist}")

    print("\n=== CC detector (full film, erode sweep) ===")
    erode_values = [int(x) for x in args.erode_sweep.split(",")]
    cc_results = {}
    for e in erode_values:
        n_w, radii, sizes = cc_windings(seg, cy, cx, erode_iters=e)
        label_2d, _ = cc_label_map(seg, cy, cx, e)
        cc_results[e] = {
            "n_w": n_w, "radii": radii, "sizes": sizes,
            "label_2d": label_2d,
        }
        max_size = max(sizes) if sizes else 0
        n_total = sum(sizes) if sizes else 1
        biggest_frac = max_size / n_total if n_total else 0
        print(f"  erode={e:2d}: n_w={n_w:2d}  "
              f"r=[{min(radii) if radii else 0:.0f}, "
              f"{max(radii) if radii else 0:.0f}]  "
              f"size_min/max={min(sizes) if sizes else 0}/{max_size}  "
              f"biggest_pix_frac={biggest_frac:.2%}")

    print("\n=== CC detector (emulsion only, erode sweep) ===")
    cc_emul_results = {}
    for e in erode_values:
        n_w, radii, sizes = cc_windings(seg, cy, cx, erode_iters=e, mask_class=2)
        label_2d, _ = cc_label_map(seg, cy, cx, e, mask_class=2)
        cc_emul_results[e] = {
            "n_w": n_w, "radii": radii, "sizes": sizes,
            "label_2d": label_2d,
        }
        max_size = max(sizes) if sizes else 0
        n_total = sum(sizes) if sizes else 1
        biggest_frac = max_size / n_total if n_total else 0
        print(f"  erode={e:2d}: n_w={n_w:2d}  "
              f"r=[{min(radii) if radii else 0:.0f}, "
              f"{max(radii) if radii else 0:.0f}]  "
              f"size_min/max={min(sizes) if sizes else 0}/{max_size}  "
              f"biggest_pix_frac={biggest_frac:.2%}")

    diag_path = os.path.join(args.out, "detection_compare.png")
    make_diagnostic(seg, cy, cx, hist_result, cc_results, diag_path)
    print(f"\nDiagnostic (full film): {diag_path}")

    diag_emul_path = os.path.join(args.out, "detection_compare_emul.png")
    make_diagnostic(seg, cy, cx, hist_result, cc_emul_results, diag_emul_path)
    print(f"Diagnostic (emulsion only): {diag_emul_path}")

    summary = {
        "volume": args.vol,
        "slice": int(z_used),
        "shape": list(seg.shape),
        "center_yx": [float(cy), float(cx)],
        "n_windings_hist": int(n_w_hist),
        "cc_results_film": {
            str(e): {
                "n_windings": cc_results[e]["n_w"],
                "radii": cc_results[e]["radii"],
                "sizes": cc_results[e]["sizes"],
            }
            for e in erode_values
        },
        "cc_results_emulsion": {
            str(e): {
                "n_windings": cc_emul_results[e]["n_w"],
                "radii": cc_emul_results[e]["radii"],
                "sizes": cc_emul_results[e]["sizes"],
            }
            for e in erode_values
        },
    }
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
