"""Whole-z close-up of ONE winding using the SIMPLE robust poly arc fit.

Generalizes `fit_arc_simple.py` across z (NOT the whole roll — one winding):
  1. Per segmented anchor (25 chunk-mids), fit r(theta) = robust low-order poly to
     the emulsion (class 2) pixels in the wedge (coarse median-per-bin track + IRLS
     poly). Seed identity is PROPAGATED across z from the reference anchor (each
     anchor seeded by the neighbour's poly at the wedge mid) so one physical ring is
     followed without index-counting.
  2. Draw each anchor's fitted arc on its CT slice -> per-anchor inspection montage.
  3. Interpolate the per-anchor polys across z to every raw-CT slice; render the
     close-up strip over the whole 1184-slice z-axis.

Wedge emulsion points + fitted coefs cached to arc_poly_cache.npz; the slow part is
the one-time seg load (~40 min, 2.2 GB probability maps). `--use-cached-anchors`
re-fits/re-renders instantly.

Usage:
  python -m unwrapping.inr.unroll_arc_simple \
      --ct-dir 01_Mickey_hdf --seg-dir 01_Mickey_3d --out-dir <out> \
      --seed-r 1078 --angle-start-deg 0 --angle-extent-deg 140 --degree 3
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.center_detection import find_spool_center
from unwrapping.inr.unwrap_data_3d import discover_volumes
from unwrapping.inr.sanity_winding_strip import bilinear_sample
from unwrapping.inr.unroll_continuous import (
    load_ct, load_seg_mid, norm_slice, discover_ct_slices,
)
from unwrapping.inr.fit_arc_simple import (
    coarse_bin_track, robust_polyfit, arclen_angles,
)


def fit_anchor(thw, rw, a0, a1, seed_r, n_bins, gate, degree):
    tb, rb = coarse_bin_track(thw, rw, a0, a1, seed_r, n_bins=n_bins, gate=gate)
    if len(tb) < degree + 1:
        return None, 0.0
    coef = robust_polyfit(tb, rb, degree=degree)
    rms = float(np.sqrt(np.mean((rb - np.polyval(coef, tb)) ** 2)))
    return coef, rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-dir", required=True)
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed-r", type=float, required=True,
                    help="Approx radius of the target winding at the REF anchor wedge.")
    ap.add_argument("--angle-start-deg", type=float, default=0.0)
    ap.add_argument("--angle-extent-deg", type=float, default=140.0)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--gate-px", type=float, default=12.0)
    ap.add_argument("--multitap-n", type=int, default=1)
    ap.add_argument("--multitap-delta-px", type=float, default=3.0)
    ap.add_argument("--z-step", type=int, default=1)
    ap.add_argument("--use-cached-anchors", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    a0 = math.radians(args.angle_start_deg)
    a1 = a0 + math.radians(args.angle_extent_deg)
    a_mid = 0.5 * (a0 + a1)

    cache = os.path.join(args.out_dir, "arc_poly_cache.npz")
    if args.use_cached_anchors and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        z_anchor = d["z_anchor"]; cx_a = list(d["cx_a"]); cy_a = list(d["cy_a"])
        TH = list(d["thw"]); RR = list(d["rw"])
        print(f"loaded cached wedge points for {len(z_anchor)} anchors")
    else:
        pairs = discover_volumes(args.seg_dir)
        print(f"{len(pairs)} anchors; loading seg (slow)...")
        z_anchor, cx_a, cy_a, TH, RR = [], [], [], [], []
        for vp, pp, z0, z1 in pairs:
            seg, mid = load_seg_mid(pp)
            cy, cx = find_spool_center(seg)
            ey, ex = np.where(seg == 2)
            th = np.mod(np.arctan2(ey - cy, ex - cx), 2 * math.pi)
            r = np.hypot(ey - cy, ex - cx)
            inw = (th >= a0) & (th <= a1)
            z_anchor.append(z0 + mid); cx_a.append(cx); cy_a.append(cy)
            TH.append(th[inw].astype(np.float32)); RR.append(r[inw].astype(np.float32))
            print(f"  anchor z{z0 + mid}: center=({cx:.0f},{cy:.0f}) "
                  f"{inw.sum()} emul px in wedge", flush=True)
        z_anchor = np.array(z_anchor, float)
        np.savez(cache, z_anchor=z_anchor, cx_a=cx_a, cy_a=cy_a,
                 thw=np.array(TH, dtype=object), rw=np.array(RR, dtype=object))

    n = len(z_anchor); ref = n // 2
    # ── seed-propagated robust poly fit, ref outward in z ──
    coefs = [None] * n; rmss = [0.0] * n
    coefs[ref], rmss[ref] = fit_anchor(TH[ref], RR[ref], a0, a1, args.seed_r,
                                       args.n_bins, args.gate_px, args.degree)
    for i in range(ref + 1, n):
        seed = np.polyval(coefs[i - 1], a_mid)
        coefs[i], rmss[i] = fit_anchor(TH[i], RR[i], a0, a1, seed,
                                       args.n_bins, args.gate_px, args.degree)
    for i in range(ref - 1, -1, -1):
        seed = np.polyval(coefs[i + 1], a_mid)
        coefs[i], rmss[i] = fit_anchor(TH[i], RR[i], a0, a1, seed,
                                       args.n_bins, args.gate_px, args.degree)
    print("  per-anchor poly RMS (px): " +
          " ".join(f"{r:.1f}" for r in rmss))

    # ── fixed arc-length columns from the ref poly ──
    ang_u, s_total = arclen_angles(coefs[ref], a0, a1)
    n_cols = len(ang_u)
    R = np.array([np.polyval(coefs[i], ang_u) for i in range(n)])   # (n_anchor, n_cols)
    fR = interp1d(z_anchor, R, axis=0, bounds_error=False, fill_value=(R[0], R[-1]))
    fcx = interp1d(z_anchor, cx_a, bounds_error=False, fill_value=(cx_a[0], cx_a[-1]))
    fcy = interp1d(z_anchor, cy_a, bounds_error=False, fill_value=(cy_a[0], cy_a[-1]))
    cos_a, sin_a = np.cos(ang_u), np.sin(ang_u)
    print(f"  arc {s_total:.0f}px -> {n_cols} cols")

    z_to_path = discover_ct_slices(args.ct_dir)
    z_list = sorted(z_to_path)[:: args.z_step]

    # ── per-anchor inspection overlays + montage ──
    from PIL import Image as _Im
    tiles = []
    for i in range(n):
        za = min(z_to_path, key=lambda zz: abs(zz - z_anchor[i]))
        img = norm_slice(load_ct(z_to_path[za]))
        r = np.polyval(coefs[i], ang_u)
        xs = cx_a[i] + r * cos_a; ys = cy_a[i] + r * sin_a
        x0 = max(0, int(xs.min()) - 40); x1 = min(img.shape[1], int(xs.max()) + 40)
        y0 = max(0, int(ys.min()) - 40); y1 = min(img.shape[0], int(ys.max()) + 40)
        fig, ax = plt.subplots(1, 1, figsize=(4, 4 * (y1 - y0) / max(1, x1 - x0)))
        ax.imshow(img[y0:y1, x0:x1], cmap="gray")
        ax.plot(xs - x0, ys - y0, "-", color="red", lw=1.2)
        ax.set_title(f"z{za} (a{i}) RMS{rmss[i]:.1f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        p = os.path.join(args.out_dir, f"anchor_{i:02d}_z{za}.png")
        plt.tight_layout(); plt.savefig(p, dpi=85); plt.close()
        tiles.append(p)
    ims = [_Im.open(t).convert("RGB").resize((300, 300)) for t in tiles]
    cols = 5; rows = int(np.ceil(len(ims) / cols))
    mon = _Im.new("RGB", (300 * cols, 300 * rows), "white")
    for j, im in enumerate(ims):
        mon.paste(im, ((j % cols) * 300, (j // cols) * 300))
    mon.save(os.path.join(args.out_dir, "anchors_montage.png"))
    print(f"  wrote {len(tiles)} anchor overlays + montage")

    # ── full-z render ──
    K = (args.multitap_n - 1) // 2
    taps = np.arange(-K, K + 1) * args.multitap_delta_px
    strip = np.empty((len(z_list), n_cols), dtype=np.float32)
    for j, z in enumerate(z_list):
        img = norm_slice(load_ct(z_to_path[z]))
        r = fR(z); cx, cy = float(fcx(z)), float(fcy(z))
        acc = np.zeros(n_cols)
        for off in taps:
            rr = r + off
            acc += bilinear_sample(img, cx + rr * cos_a, cy + rr * sin_a)
        strip[j] = (acc / len(taps)).astype(np.float32)
        if j % 200 == 0 or j == len(z_list) - 1:
            print(f"  [{j + 1}/{len(z_list)}] z{z}", flush=True)

    np.save(os.path.join(args.out_dir, "strip.npy"), strip)
    np.save(os.path.join(args.out_dir, "z_index.npy"), np.array(z_list))
    lo, hi = np.percentile(strip, [1, 99])
    disp = np.clip((strip - lo) / (hi - lo + 1e-6), 0, 1)
    _Im.fromarray((disp * 255).astype(np.uint8)).save(
        os.path.join(args.out_dir, "strip_raw.png"))
    print(f"Done. strip {strip.shape}. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
