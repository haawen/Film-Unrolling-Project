"""
Sanity check: transfer a winding snippet of the SEGMENTATION mask into a strip.

This is the foundational primitive for the "fit-a-smooth-curve-to-the-seg"
unwrapping approach. It does NOT use the analytical Archimedean spiral, the
spool-center arc model, or any trained INR. It only:

  1. Detects the spool center on a reference slice.
  2. Runs the raycast detector to get, for each winding, the emulsion-centerline
     radius as a function of ray angle (per_angle[k, ray]) — a purely
     segmentation-derived curve.
  3. Picks one winding k and a SMALL angular window, samples the CT intensity
     along that winding's emulsion centerline (optionally averaging a few radial
     taps through the emulsion thickness), and lays it out as a strip:
         rows = z slices (film-height / across-frame direction)
         cols = arc position along the winding (film-length / movie direction)
  4. Repeats for every loaded z slice and stacks them.

If the resulting strip shows coherent movie content (synthetic) or coherent
film texture (real Mickey), the mechanism is validated and a smooth spline fit
to per_angle[k] can replace the analytical base.

Outputs (per run):
  snippet_strip.png      — the winding-snippet strip (z × arc), row-upsampled
  arc_overlay.png        — sampled arc points drawn on the reference seg + image
  full_strip.png         — (optional, --full) every winding, full 360°,
                           concatenated using only seg centerlines
  gt_strip.png           — (optional, --gt-npz) the synthetic GT strip for eyeball

Usage:
  python -m unwrapping.inr.sanity_winding_strip \
      --data-dir <dir with volume_*.h5> --out-dir <out> \
      --max-slices 48 --winding 10 --angle-extent-deg 120 \
      --multitap-n 5 --multitap-delta-px 3 [--gt-npz <ground_truth.npz>] [--full]
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.center_detection import find_spool_center
from unwrapping.inr.surface_data import detect_winding_boundaries_raycast
from unwrapping.inr.unwrap_data_3d import discover_volumes, load_volume_chunk


def load_stack(data_dir, max_slices):
    """Load (image_stack, seg_stack) as (Z, H, W) float/int arrays."""
    pairs = discover_volumes(data_dir)
    if not pairs:
        raise ValueError(f"No volume_*.h5 found in {data_dir}")
    vols, segs = [], []
    done = 0
    for vol_path, probs_path, z0, z1 in pairs:
        remaining = None if max_slices is None else max(0, max_slices - done)
        if remaining == 0:
            break
        volume, seg = load_volume_chunk(vol_path, probs_path, max_slices=remaining)
        for lz in range(volume.shape[0]):
            vols.append(volume[lz])
            segs.append(seg[lz])
            done += 1
            if max_slices and done >= max_slices:
                break
        del volume, seg
        if max_slices and done >= max_slices:
            break
    image_stack = np.stack(vols, 0).astype(np.float32)
    seg_stack = np.stack(segs, 0).astype(np.int64)
    return image_stack, seg_stack


def bilinear_sample(img, xs, ys):
    """Bilinear-sample img (H, W) at float coords xs, ys (1-D arrays).

    Out-of-bounds samples return 0.0.
    """
    H, W = img.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    inb = (xs >= 0) & (xs <= W - 1) & (ys >= 0) & (ys <= H - 1)
    x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y1, 0, H - 1)
    wx = xs - x0
    wy = ys - y0
    Ia = img[y0c, x0c]; Ib = img[y0c, x1c]
    Ic = img[y1c, x0c]; Id = img[y1c, x1c]
    val = (Ia * (1 - wx) * (1 - wy) + Ib * wx * (1 - wy)
           + Ic * (1 - wx) * wy + Id * wx * wy)
    return np.where(inb, val, 0.0)


def circ_interp_radius(per_angle_k, angles):
    """Circular linear interpolation of a per-ray radius table at arbitrary angles.

    per_angle_k: (n_rays,) emulsion-centerline radius for one winding, where
                 ray i corresponds to angle 2pi*i/n_rays.
    angles:      (N,) angles in radians (any range).
    Returns:     (N,) interpolated radii.
    """
    n_rays = per_angle_k.shape[0]
    ray_angles = np.arange(n_rays) * (2.0 * math.pi / n_rays)
    a = np.mod(angles, 2.0 * math.pi)
    # Pad with wrap-around so np.interp handles the seam at 0/2pi.
    xp = np.concatenate([ray_angles - 2 * math.pi, ray_angles, ray_angles + 2 * math.pi])
    fp = np.concatenate([per_angle_k, per_angle_k, per_angle_k])
    return np.interp(a, xp, fp)


def arclength_angles(per_angle_k, a0, ext, n_cols, n_dense=8000):
    """Return angles spaced uniformly in ARC LENGTH along the winding curve.

    The winding is r(theta) over [a0, a0+ext]. Arc length grows as
    ds = sqrt(r^2 + (dr/dtheta)^2) dtheta, so equal angle steps are NOT equal
    physical steps (it matters across windings; within one winding r is nearly
    constant so the correction is small). We build a dense theta grid, integrate
    arc length, then invert s(theta) to place n_cols columns at equal arc length.

    n_cols <= 0  -> auto: one column per pixel of true arc length (~1:1 sampling).
    Returns (angles, s_total, n_cols_used).
    """
    theta_dense = np.linspace(a0, a0 + ext, n_dense)
    r_dense = circ_interp_radius(per_angle_k, theta_dense)
    dr = np.gradient(r_dense, theta_dense)
    ds = np.sqrt(r_dense ** 2 + dr ** 2)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(theta_dense))])
    s_total = float(s[-1])
    if n_cols is None or n_cols <= 0:
        n_cols = max(2, int(round(s_total)))     # ~1 column per arc pixel
    col_s = np.linspace(0.0, s_total, n_cols)
    angles = np.interp(col_s, s, theta_dense)
    return angles, s_total, n_cols


def sample_winding_strip(image_stack, center, per_angle_k, angles,
                         multitap_n=1, multitap_delta_px=3.0):
    """Build a (Z, n_cols) strip by sampling the CT along one winding centerline.

    image_stack: (Z, H, W)
    center:      (cy, cx)
    per_angle_k: (n_rays,) emulsion centerline radius for the chosen winding
    angles:      (n_cols,) ray angles to sample (the arc window)
    multitap_n:  odd number of radial taps averaged through the emulsion.
    """
    cy, cx = center
    Z = image_stack.shape[0]
    r_c = circ_interp_radius(per_angle_k, angles)          # (n_cols,)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    K = (multitap_n - 1) // 2
    taps = np.arange(-K, K + 1) * multitap_delta_px        # radial offsets (px)

    strip = np.zeros((Z, angles.shape[0]), dtype=np.float32)
    for z in range(Z):
        acc = np.zeros(angles.shape[0], dtype=np.float64)
        for off in taps:
            r = r_c + off
            xs = cx + r * cos_a
            ys = cy + r * sin_a
            acc += bilinear_sample(image_stack[z], xs, ys)
        strip[z] = (acc / len(taps)).astype(np.float32)
    # Sample coordinates of the centerline at the reference (for overlay).
    xs_c = cx + r_c * cos_a
    ys_c = cy + r_c * sin_a
    return strip, (xs_c, ys_c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-slices", type=int, default=48)
    ap.add_argument("--winding", type=int, default=None,
                    help="Winding index k (0=innermost). Default: middle winding.")
    ap.add_argument("--angle-start-deg", type=float, default=0.0)
    ap.add_argument("--angle-extent-deg", type=float, default=120.0,
                    help="Arc span of the snippet in degrees.")
    ap.add_argument("--n-cols", type=int, default=0,
                    help="Strip columns (arc resolution). <=0 (default) = auto: "
                         "one column per pixel of true arc length (~1:1).")
    ap.add_argument("--n-rays", type=int, default=1440,
                    help="Raycast detector ray count (finer = smoother curve).")
    ap.add_argument("--multitap-n", type=int, default=5)
    ap.add_argument("--multitap-delta-px", type=float, default=3.0)
    ap.add_argument("--gt-npz", type=str, default=None)
    ap.add_argument("--full", action="store_true",
                    help="Also render the full strip (all windings, 360 each).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading up to {args.max_slices} slices from {args.data_dir} ...")
    image_stack, seg_stack = load_stack(args.data_dir, args.max_slices)
    Z, H, W = image_stack.shape
    print(f"  loaded {Z} slices, {H}x{W}")

    ref = Z // 2
    cy, cx = find_spool_center(seg_stack[ref])
    print(f"  spool center (cy, cx) = ({cy:.1f}, {cx:.1f})")

    boundaries, n_layers, per_angle = detect_winding_boundaries_raycast(
        seg_stack[ref], (cy, cx), n_rays=args.n_rays,
        out_dir=args.out_dir, return_per_angle=True,
    )
    print(f"  raycast: {n_layers} windings, per_angle shape {per_angle.shape}")

    k = args.winding if args.winding is not None else n_layers // 2
    k = int(np.clip(k, 0, n_layers - 1))
    print(f"  using winding k={k}, r range "
          f"[{per_angle[k].min():.1f}, {per_angle[k].max():.1f}] px")

    a0 = math.radians(args.angle_start_deg)
    ext = math.radians(args.angle_extent_deg)
    angles, s_total, n_cols = arclength_angles(per_angle[k], a0, ext, args.n_cols)
    print(f"  arc-length-uniform columns: arc={s_total:.0f}px -> {n_cols} cols "
          f"({'auto ~1:1' if args.n_cols <= 0 else 'fixed'})")

    strip, (xs_c, ys_c) = sample_winding_strip(
        image_stack, (cy, cx), per_angle[k], angles,
        multitap_n=args.multitap_n, multitap_delta_px=args.multitap_delta_px,
    )

    # ── snippet strip figure (true z height: one row per slice, no upsampling) ──
    n_rows = strip.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(min(24, max(6, strip.shape[1] / 60)),
                                          max(2.0, n_rows / 12)))
    ax.imshow(strip, cmap="gray", aspect="auto", interpolation="nearest")
    ax.set_title(f"Winding {k} snippet  "
                 f"({args.angle_start_deg:.0f}°–{args.angle_start_deg + args.angle_extent_deg:.0f}°, "
                 f"{Z} z-slices, multitap={args.multitap_n})")
    ax.set_xlabel("arc position along winding  →")
    ax.set_ylabel("z (slice)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "snippet_strip.png"), dpi=150)
    plt.close()

    # Pixel-exact version: literally Z rows tall (true 1:1 in both axes).
    plt.imsave(os.path.join(args.out_dir, "snippet_strip_raw.png"),
               strip, cmap="gray")

    # ── overlay of sampled arc on reference slice ──
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    for ax, base, title in [
        (axes[0], image_stack[ref], "CT image (ref slice) + sampled arc"),
        (axes[1], seg_stack[ref], "Segmentation + sampled arc"),
    ]:
        ax.imshow(base, cmap="gray")
        ax.plot(xs_c, ys_c, "-", color="red", linewidth=1.5)
        ax.plot(cx, cy, "c+", markersize=14)
        ax.set_title(title)
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "arc_overlay.png"), dpi=120)
    plt.close()

    # ── optional GT strip for eyeball comparison ──
    if args.gt_npz and os.path.exists(args.gt_npz):
        gt = np.load(args.gt_npz)
        gt_strip = gt["strip"]
        fig, ax = plt.subplots(1, 1, figsize=(24, 4))
        ax.imshow(gt_strip, cmap="gray", aspect="auto", interpolation="nearest")
        ax.set_title(f"GT strip (full)  {gt_strip.shape}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "gt_strip.png"), dpi=120)
        plt.close()

    # ── optional full strip from seg centerlines only ──
    if args.full:
        print("  rendering full strip from seg centerlines (all windings)...")
        # Arc-length-uniform per winding; auto cols (1:1) preserves the true
        # relative scale (inner windings are physically shorter -> fewer cols).
        parts = []
        for kk in range(n_layers):
            full_angles, _, _ = arclength_angles(
                per_angle[kk], 0.0, 2 * math.pi, args.n_cols)
            s, _ = sample_winding_strip(
                image_stack, (cy, cx), per_angle[kk], full_angles,
                multitap_n=args.multitap_n, multitap_delta_px=args.multitap_delta_px,
            )
            parts.append(s)
        full = np.concatenate(parts, axis=1)
        fig, ax = plt.subplots(1, 1, figsize=(28, max(4, Z / 40)))
        ax.imshow(full, cmap="gray", aspect="auto", interpolation="nearest")
        ax.set_title(f"Full strip from seg centerlines only "
                     f"({n_layers} windings, no INR, no analytical base)")
        ax.set_xlabel("u (winding × angle)")
        ax.set_ylabel("z")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "full_strip.png"), dpi=130)
        plt.close()
        np.savez_compressed(os.path.join(args.out_dir, "full_strip.npz"),
                            strip=full.astype(np.float32), n_layers=n_layers)

    np.savez_compressed(os.path.join(args.out_dir, "snippet_strip.npz"),
                        strip=strip.astype(np.float32), winding=k,
                        angle_start_deg=args.angle_start_deg,
                        angle_extent_deg=args.angle_extent_deg)
    print(f"Done. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
