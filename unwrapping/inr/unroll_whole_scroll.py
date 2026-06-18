"""
Unroll the WHOLE Mickey scroll from all available volume chunks.

The roll is ~z-invariant (center drifts a few px, r_max tapers slightly over the
full z-span; the physical seam is fixed). So we:

  1. Fit the spline base ONCE on a reference chunk and convert the user's manual
     START/END marks to a fixed spiral arc-range [phi_start, phi_end] and a fixed
     true-1:1 column count.
  2. For every chunk, re-detect center + re-fit r(phi) (tracking the mild drift)
     but keep the SAME fixed seam, arc-range and column count, so every z-slice
     samples the same film positions and the rows stay content-aligned.
  3. Render each slice's row (single-tap, 1:1) and stack all slices -> the whole
     unrolled movie strip (n_slices x n_cols).

Outputs: full_strip.npy (+ z index), overview.png (downscaled), and true-1:1
zoom crops at several arc positions.

Usage:
  python -m unwrapping.inr.unroll_whole_scroll \
      --data-dir data/01_Mickey_3d --out-dir <out> \
      --start-xy 2145,1600 --end-xy 1200,178 \
      --smooth-per-pt 0.35 --n-rays 2880 --seam-exclude-deg 8
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.center_detection import find_spool_center
from unwrapping.inr.geodesic_unwrap import _detect_seam_angle
from unwrapping.inr.surface_data import detect_winding_boundaries_raycast
from unwrapping.inr.unwrap_data_3d import discover_volumes, load_volume_chunk
from unwrapping.inr.sanity_winding_strip import bilinear_sample
from unwrapping.inr.spline_base_unwrap import (
    build_spiral_cloud, fit_radius_spline, arclength_table, cols_from_arc,
    xy_to_phi, snap_to_emulsion, parse_xy,
)


def fit_chunk(seg_mid, n_rays, seam, seam_exclude_deg, smooth_per_pt):
    """Detect center + fit r(phi) on one chunk's mid slice (fixed seam)."""
    cy, cx = find_spool_center(seg_mid)
    _, n_layers, per_angle = detect_winding_boundaries_raycast(
        seg_mid, (cy, cx), n_rays=n_rays, return_per_angle=True)
    phi, r = build_spiral_cloud(per_angle, seam, n_rays,
                                seam_exclude_deg=seam_exclude_deg)
    spl = fit_radius_spline(phi, r, smooth_per_pt=smooth_per_pt)
    return cx, cy, spl, float(phi.max()), n_layers


def render_rows(image_block, cx, cy, col_phi, col_r, seam, multitap_n, delta):
    """Render (D, n_cols) rows for a chunk's image block at the curve."""
    D = image_block.shape[0]
    theta = seam + col_phi
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    K = (multitap_n - 1) // 2
    taps = np.arange(-K, K + 1) * delta
    rows = np.zeros((D, col_phi.shape[0]), dtype=np.float32)
    for d in range(D):
        acc = np.zeros(col_phi.shape[0], dtype=np.float64)
        for off in taps:
            rr = col_r + off
            acc += bilinear_sample(image_block[d], cx + rr * cos_t, cy + rr * sin_t)
        rows[d] = (acc / len(taps)).astype(np.float32)
    return rows


def save_overview(strip, out_dir, max_w=6000):
    """Downscaled overview of the whole strip (percentile-stretched)."""
    H, W = strip.shape
    step = max(1, W // max_w)
    small = strip[:, ::step]
    lo, hi = np.percentile(small, [1, 99])
    small = np.clip((small - lo) / (hi - lo + 1e-6), 0, 1)
    fig, ax = plt.subplots(1, 1, figsize=(min(26, small.shape[1] / 250),
                                          max(3, H / 60)))
    ax.imshow(small, cmap="gray", aspect="auto", interpolation="nearest")
    ax.set_title(f"Whole scroll unrolled — {H} z-slices x {W} arc px "
                 f"(overview, every {step}th col)")
    ax.set_xlabel(f"arc length (downsampled x{step})  →"); ax.set_ylabel("z slice")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "overview.png"), dpi=130)
    plt.close()


def save_zoom(strip, out_dir, frac, tag, win=2400, up=6):
    """True-1:1 arc crop, z upscaled for visibility, contrast-stretched."""
    H, W = strip.shape
    c0 = int(np.clip(frac * W - win / 2, 0, max(0, W - win)))
    crop = strip[:, c0:c0 + win]
    lo, hi = np.percentile(crop, [1, 99])
    crop = np.clip((crop - lo) / (hi - lo + 1e-6), 0, 1)
    Image.fromarray((np.repeat(crop, up, 0) * 255).astype(np.uint8)).save(
        os.path.join(out_dir, f"zoom_{tag}_raw.png"))
    fig, ax = plt.subplots(1, 1, figsize=(14, max(3, H * up / 130)))
    ax.imshow(np.repeat(crop, up, 0), cmap="gray", aspect="auto",
              interpolation="nearest")
    ax.set_title(f"Zoom {tag}: arc cols {c0}-{c0+win} (1:1), "
                 f"{H} z-slices x{up}, contrast-stretched")
    ax.set_xlabel("arc length (px)  →"); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"zoom_{tag}.png"), dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start-xy", required=True)
    ap.add_argument("--end-xy", required=True)
    ap.add_argument("--smooth-per-pt", type=float, default=0.35)
    ap.add_argument("--n-rays", type=int, default=2880)
    ap.add_argument("--seam-exclude-deg", type=float, default=8.0)
    ap.add_argument("--multitap-n", type=int, default=1)
    ap.add_argument("--multitap-delta-px", type=float, default=1.5)
    ap.add_argument("--no-snap", action="store_true")
    ap.add_argument("--max-cols", type=int, default=0,
                    help="Cap n_cols (0 = true 1:1, ~arc length px).")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pairs = discover_volumes(args.data_dir)
    print(f"{len(pairs)} chunks, z {pairs[0][2]}-{pairs[-1][3]}")

    # ── reference fit on the first chunk (where the marks were made) ──
    vol0, seg0 = load_volume_chunk(pairs[0][0], pairs[0][1])
    ref = seg0.shape[0] // 2
    cy0, cx0 = find_spool_center(seg0[ref])
    film = seg0[ref] > 0; ys, xs = np.where(film); rf = np.hypot(ys - cy0, xs - cx0)
    seam = _detect_seam_angle(film, (cy0, cx0), rf.min(), rf.max())
    print(f"  fixed seam = {math.degrees(seam):.1f} deg")
    cx0, cy0, spl0, phimax0, nl0 = fit_chunk(
        seg0[ref], args.n_rays, seam, args.seam_exclude_deg, args.smooth_per_pt)

    sx, sy = parse_xy(args.start_xy); ex, ey = parse_xy(args.end_xy)
    if not args.no_snap:
        sx, sy = snap_to_emulsion(sx, sy, seg0[ref])
        ex, ey = snap_to_emulsion(ex, ey, seg0[ref])
    phi_lo = xy_to_phi(sx, sy, cx0, cy0, seam, spl0, nl0)
    phi_hi = xy_to_phi(ex, ey, cx0, cy0, seam, spl0, nl0)
    if phi_lo > phi_hi:
        phi_lo, phi_hi = phi_hi, phi_lo
    _, _, s_total0 = arclength_table(spl0, phi_hi, phi_min=phi_lo)
    n_cols = int(round(s_total0)) if args.max_cols <= 0 else min(int(round(s_total0)), args.max_cols)
    print(f"  arc range phi=[{phi_lo:.2f},{phi_hi:.2f}] "
          f"({(phi_hi-phi_lo)/(2*math.pi):.2f} turns), n_cols={n_cols} (1:1)")
    del vol0, seg0

    # ── render every chunk with its own fit, fixed seam/arc-range/n_cols ──
    all_rows, z_index = [], []
    for ci, (vp, pp, z0, z1) in enumerate(pairs):
        vol, seg = load_volume_chunk(vp, pp)
        D = vol.shape[0]; mid = D // 2
        cx, cy, spl, phimax, nl = fit_chunk(
            seg[mid], args.n_rays, seam, args.seam_exclude_deg, args.smooth_per_pt)
        hi = min(phi_hi, phimax)
        phid, sd, st = arclength_table(spl, hi, phi_min=phi_lo)
        col_s = np.linspace(0.0, st, n_cols)
        col_phi, col_r = cols_from_arc(spl, phid, sd, col_s)
        rows = render_rows(vol, cx, cy, col_phi, col_r, seam,
                           args.multitap_n, args.multitap_delta_px)
        all_rows.append(rows)
        z_index.extend(range(z0, z0 + D))
        print(f"  [{ci+1}/{len(pairs)}] z{z0}-{z1}: center=({cx:.0f},{cy:.0f}) "
              f"nl={nl} -> {D} rows")
        del vol, seg

    strip = np.concatenate(all_rows, axis=0)
    z_index = np.array(z_index)
    print(f"FULL STRIP: {strip.shape}  ({strip.nbytes/1e6:.0f} MB)")

    np.save(os.path.join(args.out_dir, "full_strip.npy"), strip)
    np.save(os.path.join(args.out_dir, "z_index.npy"), z_index)
    save_overview(strip, args.out_dir)
    for frac, tag in [(0.04, "start"), (0.33, "early"), (0.5, "mid"),
                      (0.66, "late"), (0.96, "end")]:
        save_zoom(strip, args.out_dir, frac, tag)
    print(f"Done. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
