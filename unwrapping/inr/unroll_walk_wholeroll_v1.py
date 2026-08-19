"""Whole-roll unroll via the center-free EMULSION WALK (walk_emulsion.py).

Segmentation exists only for the 25 gapped chunks; the film is ~z-invariant. So:
  1. Per anchor (25 chunk mid-slices): film-band seed -> walk each winding along the
     emulsion (both dirs, meet at seam) -> smooth. Center-free per-winding (x,y) paths.
  2. Match windings across anchors by median radius (robust, monotone). Resample each
     winding's path to a common arc-fraction grid; interpolate (x,y) across z to all
     1184 raw-CT slices.
  3. Render the raw CT along each interpolated path; concatenate windings inner->outer.

The walk (not a polar r(theta) fit) is what changed: no spool center, so eccentricity
is irrelevant, and each band is followed as a curve so curves can't wander between
windings. Anchor walks cached (slow seg load) -> `--use-cached-anchors` re-renders fast.

Usage:
  python -m unwrapping.inr.unroll_walk_wholeroll \
      --ct-dir 01_Mickey_hdf --seg-dir 01_Mickey_3d --out-dir <out> \
      [--inspect-anchors] [--use-cached-anchors] [--smooth-px 0.6]
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
from unwrapping.inr.geodesic_unwrap import _detect_seam_angle
from unwrapping.inr.unwrap_data_3d import discover_volumes
from unwrapping.inr.sanity_winding_strip import bilinear_sample
from unwrapping.inr.unroll_continuous import (
    load_ct, norm_slice, discover_ct_slices, load_seg_mid,
)
from unwrapping.inr.walk_emulsion import (
    film_seed_azimuth, walk, smooth_path,
)

TWO_PI = 2.0 * math.pi


def walk_anchor(seg2, cx, cy, r0, r1, film_min_thick, snap_max, step, coast_max,
                stop_extent, smooth_px):
    """All per-winding emulsion paths for one seg slice. Returns list of (N,2)."""
    emul = (seg2 == 2).astype(np.float32)
    film = (seg2 > 0).astype(np.float32)
    seed_az, seeds = film_seed_azimuth(film, emul, cx, cy, r0, r1,
                                       film_min_thick, snap_max)
    tan = np.array([-math.sin(seed_az), math.cos(seed_az)])
    paths = []
    for rs in seeds:
        p0 = np.array([cx + rs * math.cos(seed_az), cy + rs * math.sin(seed_az)])
        fwd = walk(emul, p0, tan, cx, cy, step, snap_max, coast_max, stop_extent)
        bwd = walk(emul, p0, -tan, cx, cy, step, snap_max, coast_max, stop_extent)
        p = np.vstack([bwd[::-1], fwd[1:]])
        paths.append(smooth_path(p, smooth_px))
    return paths


def median_radius(path, cx, cy):
    return float(np.median(np.hypot(path[:, 0] - cx, path[:, 1] - cy)))


def resample_arc(path, K):
    """Resample a path to K points uniform in arc length (fraction 0..1)."""
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1])))]
    if d[-1] <= 0:
        return np.repeat(path[:1], K, axis=0)
    d /= d[-1]
    u = np.linspace(0, 1, K)
    return np.column_stack([np.interp(u, d, path[:, 0]), np.interp(u, d, path[:, 1])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-dir", required=True)
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smooth-px", type=float, default=0.6)
    ap.add_argument("--snap-max", type=float, default=8.0)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--coast-max", type=int, default=40)
    ap.add_argument("--film-min-thick", type=float, default=6.0)
    ap.add_argument("--seam-exclude-deg", type=float, default=6.0)
    ap.add_argument("--match-tol-px", type=float, default=12.0,
                    help="Max median-radius diff to match a winding across anchors.")
    ap.add_argument("--z-step", type=int, default=1)
    ap.add_argument("--use-cached-anchors", action="store_true")
    ap.add_argument("--inspect-anchors", action="store_true",
                    help="Draw all winding walks on each anchor CT + montage, exit.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    stop_extent = math.pi - math.radians(args.seam_exclude_deg)
    cache = os.path.join(args.out_dir, "walk_anchors.npz")

    # ── anchor walks (slow seg load; cached) ──
    if args.use_cached_anchors and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        z_anchor = d["z_anchor"]; cx_a = list(d["cx_a"]); cy_a = list(d["cy_a"])
        PATHS = list(d["paths"]); seam = float(d["seam"])
        print(f"loaded cached walks for {len(z_anchor)} anchors")
    else:
        pairs = discover_volumes(args.seg_dir)
        seg_ref, _ = load_seg_mid(pairs[0][1])
        cy0, cx0 = find_spool_center(seg_ref); f0 = seg_ref > 0
        ys0, xs0 = np.where(f0); rf0 = np.hypot(ys0 - cy0, xs0 - cx0)
        seam = _detect_seam_angle(f0, (cy0, cx0), rf0.min(), rf0.max())
        print(f"{len(pairs)} anchors; seam={math.degrees(seam):.1f}deg (loading seg...)")
        z_anchor, cx_a, cy_a, PATHS = [], [], [], []
        for vp, pp, z0, z1 in pairs:
            seg2, mid = load_seg_mid(pp)
            cy, cx = find_spool_center(seg2)
            ys, xs = np.where(seg2 > 0); rf = np.hypot(ys - cy, xs - cx)
            paths = walk_anchor(seg2, cx, cy, rf.min() + 5, rf.max() - 5,
                                args.film_min_thick, args.snap_max, args.step,
                                args.coast_max, stop_extent, args.smooth_px)
            z_anchor.append(z0 + mid); cx_a.append(cx); cy_a.append(cy)
            PATHS.append(paths)
            print(f"  anchor z{z0 + mid}: {len(paths)} winding walks", flush=True)
        z_anchor = np.array(z_anchor, float)
        np.savez(cache, z_anchor=z_anchor, cx_a=cx_a, cy_a=cy_a, seam=seam,
                 paths=np.array(PATHS, dtype=object))

    n = len(z_anchor); ref = n // 2
    z_to_path = discover_ct_slices(args.ct_dir)

    # ── match windings across anchors by median radius (monotone, robust) ──
    ref_paths = PATHS[ref]
    ref_r = np.array([median_radius(p, cx_a[ref], cy_a[ref]) for p in ref_paths])
    order = np.argsort(ref_r); ref_paths = [ref_paths[i] for i in order]
    ref_r = ref_r[order]; nw = len(ref_paths)
    # matched[k][a] = path of winding k at anchor a (or None)
    matched = [[None] * n for _ in range(nw)]
    for a in range(n):
        for p in PATHS[a]:
            rr = median_radius(p, cx_a[a], cy_a[a])
            k = int(np.argmin(np.abs(ref_r - rr)))
            if abs(ref_r[k] - rr) <= args.match_tol_px and matched[k][a] is None:
                matched[k][a] = p
    for k in range(nw):
        matched[k][ref] = ref_paths[k]
    covered = [sum(p is not None for p in matched[k]) for k in range(nw)]
    print(f"  {nw} windings; per-winding anchor coverage min/med/max = "
          f"{min(covered)}/{int(np.median(covered))}/{max(covered)}")

    # ── inspect: draw all winding walks on each anchor CT + montage ──
    if args.inspect_anchors:
        from PIL import Image as _Im
        cmap = plt.get_cmap("hsv"); tiles = []
        for a in range(n):
            za = min(z_to_path, key=lambda zz: abs(zz - z_anchor[a]))
            img = norm_slice(load_ct(z_to_path[za]))
            fig, ax = plt.subplots(1, 1, figsize=(6, 6)); ax.imshow(img, cmap="gray")
            for k in range(nw):
                p = matched[k][a]
                if p is not None:
                    ax.plot(p[:, 0], p[:, 1], "-", color=cmap(k / nw), lw=0.5)
            ax.set_title(f"z{za} — {sum(matched[k][a] is not None for k in range(nw))}"
                         f"/{nw}w", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            pth = os.path.join(args.out_dir, f"insp_{a:02d}_z{za}.png")
            plt.tight_layout(); plt.savefig(pth, dpi=95); plt.close(); tiles.append(pth)
            print(f"  drew anchor {a} z{za}", flush=True)
        ims = [_Im.open(t).convert("RGB").resize((360, 360)) for t in tiles]
        cols = 5; rows = int(np.ceil(len(ims) / cols))
        mon = _Im.new("RGB", (360 * cols, 360 * rows), "white")
        for j, im in enumerate(ims):
            mon.paste(im, ((j % cols) * 360, (j // cols) * 360))
        mon.save(os.path.join(args.out_dir, "walk_anchors_montage.png"))
        for t in tiles:
            os.remove(t)
        print(f"Done (inspect). montage in {args.out_dir}")
        return

    # ── resample each winding to a common arc grid; interp (x,y) across z ──
    fcx = interp1d(z_anchor, cx_a, bounds_error=False, fill_value=(cx_a[0], cx_a[-1]))
    fcy = interp1d(z_anchor, cy_a, bounds_error=False, fill_value=(cy_a[0], cy_a[-1]))
    win = []; total_cols = 0
    for k in range(nw):
        present = [a for a in range(n) if matched[k][a] is not None]
        if len(present) < 2:
            continue
        K = max(2, int(round(
            np.median([np.hypot(np.diff(matched[k][a][:, 0]),
                                np.diff(matched[k][a][:, 1])).sum() for a in present]))))
        X = np.empty((n, K), np.float32); Y = np.empty((n, K), np.float32)
        for a in range(n):
            src = matched[k][a] if matched[k][a] is not None else \
                matched[k][min(present, key=lambda p: abs(p - a))]
            rp = resample_arc(src, K); X[a] = rp[:, 0]; Y[a] = rp[:, 1]
        fX = interp1d(z_anchor, X, axis=0, bounds_error=False, fill_value=(X[0], X[-1]))
        fY = interp1d(z_anchor, Y, axis=0, bounds_error=False, fill_value=(Y[0], Y[-1]))
        win.append((fX, fY, K)); total_cols += K
    print(f"  whole roll: {len(win)} windings, {total_cols} cols")

    z_list = sorted(z_to_path)[:: args.z_step]
    strip = np.empty((len(z_list), total_cols), dtype=np.float32)
    for j, z in enumerate(z_list):
        img = norm_slice(load_ct(z_to_path[z]))
        c = 0
        for fX, fY, K in win:
            strip[j, c:c + K] = bilinear_sample(img, fX(z), fY(z)); c += K
        if j % 200 == 0 or j == len(z_list) - 1:
            print(f"  [{j + 1}/{len(z_list)}] z{z}", flush=True)

    np.save(os.path.join(args.out_dir, "wholeroll.npy"), strip)
    np.save(os.path.join(args.out_dir, "z_index.npy"), np.array(z_list))
    from PIL import Image as _Im
    stepc = max(1, total_cols // 6000)
    sm = strip[:, ::stepc]; lo, hi = np.percentile(sm, [1, 99])
    sm = np.clip((sm - lo) / (hi - lo + 1e-6), 0, 1)
    _Im.fromarray((sm * 255).astype(np.uint8)).save(
        os.path.join(args.out_dir, "wholeroll_overview.png"))
    print(f"Done. whole roll {strip.shape}. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
