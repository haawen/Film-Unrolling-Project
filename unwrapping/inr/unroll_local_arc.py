"""Unroll ONE local winding-arc over the whole continuous z-axis.

Unlike `unroll_continuous.py` (which sliced a GLOBAL whole-roll r(phi) spline and
inherited its per-winding fit errors -> wriggly arcs), this traces a SINGLE
winding's emulsion centerline directly from each slice's raycast detector
(`per_angle[k]`, exactly like `sanity_winding_strip.py`) over a small
absolute-angle window, then interpolates that LOCAL arc's (center + radius
profile) across z to every raw-CT slice. No global spline, no seam unwrapping ->
the arc stays as smooth as the sanity-check arc.

Pipeline:
  1. Anchors: on each of the 25 segmented chunks' mid slices, detect center +
     raycast per_angle (n_layers, n_rays). Cache to anchor_perangle.npz — the
     seg load is the slow ~40-min part, so any later winding/window is instant.
  2. Pick winding k + absolute-angle window [a0, a0+ext]. Reference = middle
     anchor; columns placed uniformly in ARC LENGTH (sanity convention).
  3. Per anchor, radius profile r(angle) over the window = circ_interp_radius of
     that anchor's per_angle[k]. Interpolate (center, radius profile) across z.
  4. Render the close-up by sampling raw CT along the interpolated local arc at
     every z. Also draw the (smooth) arc on a reference CT slice to verify.

Usage:
  python -m unwrapping.inr.unroll_local_arc \
      --ct-dir 01_Mickey_hdf --seg-dir 01_Mickey_3d --out-dir <out> \
      --winding 16 --angle-start-deg 0 --angle-extent-deg 120 \
      --n-rays 2880 --multitap-n 1 [--use-cached-anchors]
"""

import argparse
import math
import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d, UnivariateSpline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.center_detection import find_spool_center
from unwrapping.inr.surface_data import detect_winding_boundaries_raycast
from unwrapping.inr.unwrap_data_3d import discover_volumes
from unwrapping.inr.sanity_winding_strip import (
    circ_interp_radius, arclength_angles, bilinear_sample,
)
from unwrapping.inr.unroll_continuous import (
    load_ct, load_seg_mid, norm_slice, discover_ct_slices,
)


def local_radius_profile(per_angle_k, angles_eval, smooth_per_pt):
    """Radius r(angle) for ONE winding over the eval angles.

    smooth_per_pt <= 0  -> raw raycast centerline (sanity-check behavior).
    smooth_per_pt  > 0  -> a LOCAL per-slice, per-winding smoothing spline over
        just this winding's angle window (target ~smooth_per_pt px RMS). This is
        NOT the rejected whole-roll r(phi) spline; it only denoises the single
        arc's per-ray jitter on the noisier mid-roll chunks so every slice looks
        like the clean early chunks.
    """
    if smooth_per_pt is None or smooth_per_pt <= 0:
        return circ_interp_radius(per_angle_k, angles_eval)
    n = per_angle_k.shape[0]
    ray = np.arange(n) * (2.0 * math.pi / n)
    a = np.mod(angles_eval, 2.0 * math.pi)
    lo, hi = a.min(), a.max()
    pad = math.radians(3.0)
    m = (ray >= lo - pad) & (ray <= hi + pad)
    xa, ya = ray[m], per_angle_k[m]
    good = ~np.isnan(ya)
    xa, ya = xa[good], ya[good]
    s = (smooth_per_pt ** 2) * len(xa)
    spl = UnivariateSpline(xa, ya, k=3, s=s)
    return spl(a)


def build_anchor_cache(seg_dir, n_rays, cache_path):
    """Detect center + raycast per_angle on each chunk's mid slice; cache it."""
    pairs = discover_volumes(seg_dir)
    print(f"{len(pairs)} segmented chunks, z {pairs[0][2]}-{pairs[-1][3]}")
    z_anchor, cx_a, cy_a, per_angle_list, nls = [], [], [], [], []
    for vp, pp, z0, z1 in pairs:
        seg_mid, mid = load_seg_mid(pp)
        cy, cx = find_spool_center(seg_mid)
        _, nl, per_angle = detect_winding_boundaries_raycast(
            seg_mid, (cy, cx), n_rays=n_rays, return_per_angle=True)
        z_anchor.append(z0 + mid); cx_a.append(cx); cy_a.append(cy)
        per_angle_list.append(per_angle); nls.append(nl)
        print(f"  anchor z{z0 + mid}: center=({cx:.0f},{cy:.0f}) nl={nl}", flush=True)
    maxnl = max(nls)
    PA = np.full((len(pairs), maxnl, n_rays), np.nan, dtype=np.float32)
    for i, pa in enumerate(per_angle_list):
        PA[i, :pa.shape[0]] = pa
    np.savez_compressed(cache_path, z_anchor=np.array(z_anchor, float),
                        cx_a=cx_a, cy_a=cy_a, per_angle=PA, nls=nls, n_rays=n_rays)
    print(f"  cached anchors -> {cache_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-dir", required=True)
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--anchor-cache", default=None,
                    help="Path to anchor_perangle.npz (default: <out-dir>/..).")
    ap.add_argument("--use-cached-anchors", action="store_true")
    ap.add_argument("--winding", type=int, default=None,
                    help="Winding index k (0=innermost). Default: middle.")
    ap.add_argument("--angle-start-deg", type=float, default=0.0)
    ap.add_argument("--angle-extent-deg", type=float, default=120.0)
    ap.add_argument("--n-rays", type=int, default=2880)
    ap.add_argument("--multitap-n", type=int, default=1)
    ap.add_argument("--multitap-delta-px", type=float, default=3.0)
    ap.add_argument("--smooth-per-pt", type=float, default=0.0,
                    help="Local per-slice per-winding centerline smoothing (px "
                         "RMS). 0 = raw raycast (sanity behavior).")
    ap.add_argument("--z-step", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Anchor cache lives one level up so it is shared across per-arc out-dirs.
    cache_path = args.anchor_cache or os.path.join(
        os.path.dirname(os.path.normpath(args.out_dir)),
        f"anchor_perangle_r{args.n_rays}.npz")
    if not (args.use_cached_anchors and os.path.exists(cache_path)):
        build_anchor_cache(args.seg_dir, args.n_rays, cache_path)
    d = np.load(cache_path)
    z_anchor = d["z_anchor"]; cx_a = d["cx_a"]; cy_a = d["cy_a"]
    PA = d["per_angle"]; nls = list(d["nls"]); n_rays = int(d["n_rays"])
    print(f"anchors: {len(z_anchor)}  windings per anchor: {nls}")

    k = args.winding if args.winding is not None else min(nls) // 2
    print(f"winding k={k}  window [{args.angle_start_deg:.0f}, "
          f"{args.angle_start_deg + args.angle_extent_deg:.0f}] deg")

    # Reference = middle anchor; arc-length-uniform columns over the window.
    ref_i = len(z_anchor) // 2
    a0 = math.radians(args.angle_start_deg)
    ext = math.radians(args.angle_extent_deg)
    angles, s_total, n_cols = arclength_angles(PA[ref_i, k], a0, ext, n_cols=0)
    print(f"  ref anchor z{int(z_anchor[ref_i])}: arc={s_total:.0f}px -> {n_cols} cols")

    # Radius profile per anchor over the fixed angle grid; interpolate across z.
    # Each profile is a LOCAL single-winding centerline (optionally smoothed).
    Rprof = np.array([local_radius_profile(PA[i, k], angles, args.smooth_per_pt)
                      for i in range(len(z_anchor))])           # (n_anchor, n_cols)
    fR = interp1d(z_anchor, Rprof, axis=0, bounds_error=False,
                  fill_value=(Rprof[0], Rprof[-1]))
    fcx = interp1d(z_anchor, cx_a, bounds_error=False, fill_value=(cx_a[0], cx_a[-1]))
    fcy = interp1d(z_anchor, cy_a, bounds_error=False, fill_value=(cy_a[0], cy_a[-1]))
    cos_a, sin_a = np.cos(angles), np.sin(angles)
    K = (args.multitap_n - 1) // 2
    taps = np.arange(-K, K + 1) * args.multitap_delta_px

    z_to_path = discover_ct_slices(args.ct_dir)
    z_list = sorted(z_to_path)[:: args.z_step]
    print(f"{len(z_to_path)} raw-CT slices; rendering {len(z_list)} (step {args.z_step})")

    # ── arc-overlay on a reference CT slice (smoothness check) ──
    z_ov = int(z_anchor[ref_i])
    z_ov = min(z_to_path, key=lambda zz: abs(zz - z_ov))
    img_ov = load_ct(z_to_path[z_ov])
    r_ov = float(np.nanmean(Rprof[ref_i])); cx_ov, cy_ov = float(fcx(z_ov)), float(fcy(z_ov))
    rr = fR(z_ov)
    fig, ax = plt.subplots(1, 1, figsize=(11, 11))
    lo, hi = np.percentile(img_ov, [1, 99])
    ax.imshow(np.clip((img_ov - lo) / (hi - lo + 1e-6), 0, 1), cmap="gray")
    ax.plot(cx_ov + rr * cos_a, cy_ov + rr * sin_a, "-", color="red", lw=2.0)
    ax.plot(cx_ov, cy_ov, "c+", ms=14)
    ax.set_title(f"Local winding-{k} arc on raw CT z={z_ov} "
                 f"({args.angle_extent_deg:.0f} deg, {n_cols}px arc)")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "arc_overlay.png"), dpi=140)
    plt.close()

    # ── render the close-up over the whole z ──
    strip = np.empty((len(z_list), n_cols), dtype=np.float32)
    for i, z in enumerate(z_list):
        img = norm_slice(load_ct(z_to_path[z]))
        r = fR(z); cx, cy = float(fcx(z)), float(fcy(z))
        acc = np.zeros(n_cols, dtype=np.float64)
        for off in taps:
            acc += bilinear_sample(img, cx + (r + off) * cos_a, cy + (r + off) * sin_a)
        strip[i] = (acc / len(taps)).astype(np.float32)
        if i % 100 == 0 or i == len(z_list) - 1:
            print(f"  [{i + 1}/{len(z_list)}] z{z}", flush=True)

    np.save(os.path.join(args.out_dir, "strip.npy"), strip)
    np.save(os.path.join(args.out_dir, "z_index.npy"), np.array(z_list))
    lo, hi = np.percentile(strip, [1, 99])
    disp = np.clip((strip - lo) / (hi - lo + 1e-6), 0, 1)
    from PIL import Image as _Im
    _Im.fromarray((disp * 255).astype(np.uint8)).save(
        os.path.join(args.out_dir, "strip_raw.png"))
    fig, ax = plt.subplots(1, 1, figsize=(14, max(4, len(z_list) / 90)))
    ax.imshow(disp, cmap="gray", aspect="auto", interpolation="nearest")
    ax.set_title(f"Local winding-{k} arc unrolled over {len(z_list)} z-slices "
                 f"(arc {n_cols}px, multitap {args.multitap_n})")
    ax.set_xlabel("arc length (px)  →"); ax.set_ylabel("z slice")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "strip.png"), dpi=130)
    plt.close()
    print(f"Done. strip {strip.shape}. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
