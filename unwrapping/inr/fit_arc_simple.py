"""Simplest-possible smooth short-arc fit from the segmentation (single slice).

Reset (2026-06-17): drop the fragile per-ray band-COUNTING (raycast detector,
continuity tracker, caches) that wriggles/skips on the dashed mid-roll emulsion.
Instead, for ONE winding over a short angular wedge, FIT a smooth low-order curve:

  1. Emulsion (class 2) pixels -> polar (theta, r) about the spool center.
  2. Isolate ONE winding: a coarse robust track = median emulsion radius per
     angular bin, propagated from a manual seed within +-half-spacing (gap-tolerant,
     only ~B points so it cannot wriggle).
  3. Fit r(theta) = polynomial degree d (2-3) by robust IRLS to the bin points.
     Low order => smooth by construction; fitting class 2 => lands on the emulsion.
  4. Sample CT along the curve (arc-length-uniform) over all loaded z -> mini strip.

Validate on a CLEAN (z846) and a NOISY (z1605) chunk before generalizing.

Usage:
  python -m unwrapping.inr.fit_arc_simple \
      --vol 01_Mickey_3d/volume_1595-1614.h5 \
      --probs 01_Mickey_3d/volume_1595-1614_Probabilities.h5 \
      --out-dir <out> --seed-r 1000 --angle-start-deg 0 --angle-extent-deg 140 \
      --degree 3
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
from unwrapping.inr.unwrap_data_3d import load_volume_chunk
from unwrapping.inr.sanity_winding_strip import bilinear_sample


def coarse_bin_track(thw, rw, a0, a1, seed_r, n_bins=24, gate=11.0):
    """Median emulsion radius per angular bin, propagated from the seed bin.

    Returns (theta_bin, r_bin) for bins that had emulsion pixels within +-gate of
    the running estimate. Robust to dashed emulsion (median) and to neighbour
    windings (gate < layer spacing). Seed bin = the one containing the wedge mid.
    """
    edges = np.linspace(a0, a1, n_bins + 1)
    cen = 0.5 * (edges[:-1] + edges[1:])
    seed_bin = n_bins // 2
    tb, rb = [], []
    r_prev = seed_r
    # walk right from the seed bin, then left, so the estimate propagates both ways
    for order in (range(seed_bin, n_bins), range(seed_bin - 1, -1, -1)):
        r_run = r_prev
        for b in order:
            m = (thw >= edges[b]) & (thw < edges[b + 1]) & (np.abs(rw - r_run) < gate)
            if m.sum() >= 3:
                r_run = float(np.median(rw[m]))
                tb.append(cen[b]); rb.append(r_run)
        if order.start == seed_bin:      # remember the seed-bin radius for the left walk
            r_prev = rb[0] if rb else seed_r
    o = np.argsort(tb)
    return np.array(tb)[o], np.array(rb)[o]


def robust_polyfit(x, y, degree=3, iters=3):
    """Iteratively-reweighted least-squares polynomial r(theta)."""
    c = np.polyfit(x, y, degree)
    for _ in range(iters):
        res = y - np.polyval(c, x)
        s = 1.4826 * np.median(np.abs(res)) + 1e-6
        w = 1.0 / (1.0 + (res / (2.0 * s)) ** 2)
        c = np.polyfit(x, y, degree, w=w)
    return c


def arclen_angles(coef, a0, a1, n_dense=8000):
    """Arc-length-uniform sample angles along r(theta)=poly, ~1 col per arc px."""
    th = np.linspace(a0, a1, n_dense)
    r = np.polyval(coef, th)
    dr = np.polyval(np.polyder(coef), th)
    ds = np.sqrt(r ** 2 + dr ** 2)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(th))])
    n_cols = max(2, int(round(s[-1])))
    return np.interp(np.linspace(0, s[-1], n_cols), s, th), float(s[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed-r", type=float, required=True,
                    help="Approx radius (px) of the target winding at the wedge.")
    ap.add_argument("--angle-start-deg", type=float, default=0.0)
    ap.add_argument("--angle-extent-deg", type=float, default=140.0)
    ap.add_argument("--degree", type=int, default=3)
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--gate-px", type=float, default=11.0)
    ap.add_argument("--multitap-n", type=int, default=1)
    ap.add_argument("--multitap-delta-px", type=float, default=3.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    vol, seg = load_volume_chunk(args.vol, args.probs)
    Z = vol.shape[0]; mid = Z // 2
    cy, cx = find_spool_center(seg[mid])
    a0 = math.radians(args.angle_start_deg)
    a1 = a0 + math.radians(args.angle_extent_deg)
    print(f"loaded {Z} slices; center=({cx:.0f},{cy:.0f}); wedge "
          f"[{args.angle_start_deg:.0f},{args.angle_start_deg + args.angle_extent_deg:.0f}]deg "
          f"seed_r={args.seed_r:.0f}")

    ey, ex = np.where(seg[mid] == 2)
    th = np.mod(np.arctan2(ey - cy, ex - cx), 2 * math.pi)
    r = np.hypot(ey - cy, ex - cx)
    inw = (th >= a0) & (th <= a1)
    thw, rw = th[inw], r[inw]

    tb, rb = coarse_bin_track(thw, rw, a0, a1, args.seed_r,
                              n_bins=args.n_bins, gate=args.gate_px)
    print(f"  coarse track: {len(tb)} bins, r {rb.min():.0f}-{rb.max():.0f}")
    coef = robust_polyfit(tb, rb, degree=args.degree)
    fit_resid = float(np.sqrt(np.mean((rb - np.polyval(coef, tb)) ** 2)))
    print(f"  poly degree {args.degree}: RMS(bin resid)={fit_resid:.2f}px")

    ang, s_total = arclen_angles(coef, a0, a1)
    r_col = np.polyval(coef, ang)
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    xs = cx + r_col * cos_a; ys = cy + r_col * sin_a

    # ── overlay (full + zoom) on mid CT: emulsion px, bin track, fitted curve ──
    img = vol[mid]; lo, hi = np.percentile(img, [1, 99])
    disp = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
    fig, ax = plt.subplots(1, 1, figsize=(11, 11))
    ax.imshow(disp, cmap="gray")
    ax.plot(xs, ys, "-", color="red", lw=2.0, label="fitted r(theta)")
    ax.plot(cx + rb * np.cos(tb), cy + rb * np.sin(tb), "o", mfc="none",
            mec="lime", ms=5, label="bin track")
    ax.plot(cx, cy, "c+", ms=14); ax.legend(loc="upper right", fontsize=9)
    ax.set_title(f"Simple poly arc fit (deg {args.degree})"); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "fit_overlay.png"), dpi=140)
    plt.close()
    # zoom on the arc bbox
    x0 = max(0, int(xs.min()) - 50); x1 = min(img.shape[1], int(xs.max()) + 50)
    y0 = max(0, int(ys.min()) - 50); y1 = min(img.shape[0], int(ys.max()) + 50)
    fig, ax = plt.subplots(1, 1, figsize=(14, 14 * (y1 - y0) / max(1, x1 - x0)))
    ax.imshow(disp[y0:y1, x0:x1], cmap="gray")
    ax.plot(xs - x0, ys - y0, "-", color="red", lw=1.5)
    ax.scatter(ex[inw] - x0, ey[inw] - y0, s=2, c="cyan", alpha=0.25)
    ax.set_xlim(0, x1 - x0); ax.set_ylim(y1 - y0, 0)
    ax.set_title("zoom: fitted curve (red) over emulsion px (cyan)")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "fit_zoom.png"), dpi=150)
    plt.close()

    # ── mini strip over all z (single/multi-tap) ──
    K = (args.multitap_n - 1) // 2
    taps = np.arange(-K, K + 1) * args.multitap_delta_px
    strip = np.zeros((Z, len(ang)), dtype=np.float32)
    for z in range(Z):
        acc = np.zeros(len(ang))
        for off in taps:
            rr = r_col + off
            acc += bilinear_sample(vol[z], cx + rr * cos_a, cy + rr * sin_a)
        strip[z] = (acc / len(taps)).astype(np.float32)
    from PIL import Image as _Im
    lo, hi = np.percentile(strip, [1, 99])
    sd = np.clip((strip - lo) / (hi - lo + 1e-6), 0, 1)
    _Im.fromarray((np.repeat(sd, 6, 0) * 255).astype(np.uint8)).save(
        os.path.join(args.out_dir, "strip_raw.png"))
    print(f"Done. arc {s_total:.0f}px, strip {strip.shape}. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
