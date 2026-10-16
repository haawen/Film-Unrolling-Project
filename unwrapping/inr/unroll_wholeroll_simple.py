"""Whole-roll unroll using the SIMPLE robust poly fit, one winding at a time.

Generalizes `unroll_arc_simple.py` from a 140-deg wedge to the FULL roll:
  1. Per segmented anchor, collect ALL emulsion (class 2) pixels (full circle) in
     polar (theta, r) about the spool center. Detect the seam once on the ref slice.
  2. Reference per-winding seed radii = radius clusters of the emulsion at a
     reference azimuth (phi = pi, opposite the seam).
  3. For each winding, parameterise by phi = wrap(theta - seam) in [0, 2pi) and fit
     r(phi) over [eps, 2pi-eps] (seam band excluded) with the SIMPLE robust fit
     (coarse median-per-bin track seeded at phi=pi + IRLS polynomial). Seed identity
     propagated across z from the middle anchor.
  4. Interpolate each winding's poly across z; render each winding's full-turn strip
     (arc-length-uniform) over all raw-CT slices and CONCATENATE inner->outer ->
     the whole unrolled roll.

Full-circle emulsion points + per-winding coefs cached to wholeroll_cache.npz; the
seg load is the slow ~40-min part. `--use-cached-anchors` re-fits/re-renders fast.

Usage:
  python -m unwrapping.inr.unroll_wholeroll_simple \
      --ct-dir 01_Mickey_hdf --seg-dir 01_Mickey_3d --out-dir <out> \
      --degree 6 --seam-exclude-deg 8
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
    load_ct, load_seg_mid, norm_slice, discover_ct_slices,
)
from unwrapping.inr.fit_arc_simple import coarse_bin_track, robust_polyfit


TWO_PI = 2.0 * math.pi


def winding_seeds(phi, r, ref_phi=math.pi, dphi=math.radians(10.0), max_gap=14.0):
    """Per-winding seed radii = radius clusters of emulsion near a reference phi.

    Window is +-dphi around ref_phi (phi=pi, opposite the seam). dphi=10deg keeps
    the spiral growth within the window ~1px (windings stay 1-spacing apart and
    separable) while covering enough azimuth that the dashed emulsion still has at
    least one pixel per winding (a +-2deg window missed most windings -> only 5).
    """
    m = np.abs(phi - ref_phi) < dphi
    rr = np.sort(r[m])
    if rr.size == 0:
        return np.array([])
    seeds, cur = [], [rr[0]]
    for v in rr[1:]:
        if v - cur[-1] < max_gap:
            cur.append(v)
        else:
            seeds.append(np.median(cur)); cur = [v]
    seeds.append(np.median(cur))
    return np.array(seeds)


def filmband_seeds(FB_ref, rayi, n_rays, half_deg=6.0):
    """Per-winding seed radii: most-complete single ray + spacing infill.

    Pooling rays smears windings (eccentricity sweeps a winding ~one spacing over
    ~20deg). So instead: scan +-half_deg of rays and pick the ONE ray with the most
    detected film bands (windings stay separable on a single ray), then INFILL
    windings the detector missed there — a missing winding shows up as a gap that is
    ~k x the median single-winding spacing, so split such gaps evenly. Recovers
    skipped windings without merging neighbours. Returns sorted seed radii.
    """
    half = max(1, int(round(half_deg / 360.0 * n_rays)))
    best, bestn = rayi, -1
    for d in range(-half, half + 1):
        r = (rayi + d) % n_rays
        c = int(np.sum(~np.isnan(FB_ref[r])))
        if c > bestn:
            bestn, best = c, r
    bands = np.sort(FB_ref[best][~np.isnan(FB_ref[best])])
    if bands.size < 2:
        return bands
    med = float(np.median(np.diff(bands)))                 # typical 1-winding spacing
    out = [bands[0]]
    for i in range(1, len(bands)):
        g = bands[i] - bands[i - 1]
        nfill = max(0, int(round(g / med)) - 1)            # missing windings in gap
        for j in range(1, nfill + 1):
            out.append(bands[i - 1] + g * j / (nfill + 1))
        out.append(bands[i])
    return np.array(out)


def fit_winding(phi_w, r_w, seed_r, eps, n_bins, gate, degree):
    tb, rb = coarse_bin_track(phi_w, r_w, eps, TWO_PI - eps, seed_r,
                              n_bins=n_bins, gate=gate)
    if len(tb) < degree + 1:
        return None
    return robust_polyfit(tb, rb, degree=degree)


def arclen_phi(coef, eps):
    phid = np.linspace(eps, TWO_PI - eps, 12000)
    r = np.polyval(coef, phid); dr = np.polyval(np.polyder(coef), phid)
    ds = np.sqrt(r ** 2 + dr ** 2)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(phid))])
    n = max(2, int(round(s[-1])))
    return np.interp(np.linspace(0, s[-1], n), s, phid), float(s[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-dir", required=True)
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--degree", type=int, default=6)
    ap.add_argument("--n-bins", type=int, default=60)
    ap.add_argument("--gate-px", type=float, default=12.0)
    ap.add_argument("--seam-exclude-deg", type=float, default=8.0)
    ap.add_argument("--filmband-cache", default=None,
                    help="anchor_filmband_*.npz — robust per-winding seed radii "
                         "(continuous film bands; avoids the dashed-emulsion "
                         "seeding failure).")
    ap.add_argument("--seed-ray-halfdeg", type=float, default=6.0,
                    help="Pool film bands over +-this many deg of rays for seeding "
                         "(recovers windings that drop out at a single ray).")
    ap.add_argument("--multitap-n", type=int, default=1)
    ap.add_argument("--multitap-delta-px", type=float, default=3.0)
    ap.add_argument("--z-step", type=int, default=1)
    ap.add_argument("--use-cached-anchors", action="store_true")
    ap.add_argument("--inspect-anchors", action="store_true",
                    help="Draw ALL windings' fitted curves on each anchor's CT "
                         "slice + montage (whole-roll fit check), then exit.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    eps = math.radians(args.seam_exclude_deg)

    cache = os.path.join(args.out_dir, "wholeroll_cache.npz")
    if args.use_cached_anchors and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        z_anchor = d["z_anchor"]; cx_a = list(d["cx_a"]); cy_a = list(d["cy_a"])
        PHI = list(d["phi"]); RR = list(d["r"]); seam = float(d["seam"])
        print(f"loaded cached full-circle emulsion for {len(z_anchor)} anchors; "
              f"seam={math.degrees(seam):.1f}")
    else:
        pairs = discover_volumes(args.seg_dir)
        print(f"{len(pairs)} anchors; loading seg (slow)...")
        ref_seg, _ = load_seg_mid(pairs[len(pairs) // 2][1])
        cyr, cxr = find_spool_center(ref_seg)
        film = ref_seg > 0; ys, xs = np.where(film)
        rf = np.hypot(ys - cyr, xs - cxr)
        seam = _detect_seam_angle(film, (cyr, cxr), rf.min(), rf.max())
        print(f"  seam = {math.degrees(seam):.1f} deg")
        z_anchor, cx_a, cy_a, PHI, RR = [], [], [], [], []
        for vp, pp, z0, z1 in pairs:
            seg, mid = load_seg_mid(pp)
            cy, cx = find_spool_center(seg)
            ey, ex = np.where(seg == 2)
            th = np.arctan2(ey - cy, ex - cx)
            r = np.hypot(ey - cy, ex - cx)
            phi = np.mod(th - seam, TWO_PI)
            z_anchor.append(z0 + mid); cx_a.append(cx); cy_a.append(cy)
            PHI.append(phi.astype(np.float32)); RR.append(r.astype(np.float32))
            print(f"  anchor z{z0 + mid}: center=({cx:.0f},{cy:.0f}) {len(ey)} emul px",
                  flush=True)
        z_anchor = np.array(z_anchor, float)
        np.savez(cache, z_anchor=z_anchor, cx_a=cx_a, cy_a=cy_a, seam=seam,
                 phi=np.array(PHI, dtype=object), r=np.array(RR, dtype=object))

    n = len(z_anchor); ref = n // 2
    if args.filmband_cache:
        fb = np.load(args.filmband_cache)
        FB = fb["FB"]; nr = int(fb["n_rays"])
        rayi = int(round((np.mod(seam + math.pi, TWO_PI)) / TWO_PI * nr)) % nr
        seeds = filmband_seeds(FB[ref], rayi, nr, half_deg=args.seed_ray_halfdeg)
        print(f"  seeds from film-band cache (ray {rayi} +-{args.seed_ray_halfdeg}deg, "
              f"pooled+clustered)")
    else:
        seeds = winding_seeds(PHI[ref], RR[ref])
    nw = len(seeds)
    print(f"  {nw} windings (seed radii {seeds.min():.0f}-{seeds.max():.0f})")

    # ── per-winding, seed-propagated poly fit across z ──
    # coefs[k][i] = poly for winding k at anchor i
    coefs = [[None] * n for _ in range(nw)]
    for k in range(nw):
        coefs[k][ref] = fit_winding(PHI[ref], RR[ref], seeds[k], eps,
                                    args.n_bins, args.gate_px, args.degree)
        for i in range(ref + 1, n):
            prev = coefs[k][i - 1]
            sd = np.polyval(prev, math.pi) if prev is not None else seeds[k]
            coefs[k][i] = fit_winding(PHI[i], RR[i], sd, eps,
                                      args.n_bins, args.gate_px, args.degree)
        for i in range(ref - 1, -1, -1):
            nxt = coefs[k][i + 1]
            sd = np.polyval(nxt, math.pi) if nxt is not None else seeds[k]
            coefs[k][i] = fit_winding(PHI[i], RR[i], sd, eps,
                                      args.n_bins, args.gate_px, args.degree)
    print("  fitted all windings x anchors")

    z_to_path = discover_ct_slices(args.ct_dir)

    # ── whole-roll fit inspection: all windings drawn on each anchor's CT ──
    if args.inspect_anchors:
        from PIL import Image as _Im
        phig = np.linspace(eps, TWO_PI - eps, 3000)
        cos_t, sin_t = np.cos(seam + phig), np.sin(seam + phig)
        tiles = []
        for i in range(n):
            za = min(z_to_path, key=lambda zz: abs(zz - z_anchor[i]))
            img = norm_slice(load_ct(z_to_path[za]))
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.imshow(img, cmap="gray")
            for k in range(nw):
                if coefs[k][i] is None:
                    continue
                r = np.polyval(coefs[k][i], phig)
                ax.plot(cx_a[i] + r * cos_t, cy_a[i] + r * sin_t, "-",
                        color="red", lw=0.4)
            ax.plot(cx_a[i], cy_a[i], "c+", ms=10)
            ax.set_title(f"z{za} (a{i}) — {nw} windings", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            p = os.path.join(args.out_dir, f"fit_anchor_{i:02d}_z{za}.png")
            plt.tight_layout(); plt.savefig(p, dpi=95); plt.close()
            tiles.append(p)
            print(f"  anchor {i} z{za} drawn", flush=True)
        ims = [_Im.open(t).convert("RGB").resize((360, 360)) for t in tiles]
        cols = 5; rows = int(np.ceil(len(ims) / cols))
        mon = _Im.new("RGB", (360 * cols, 360 * rows), "white")
        for j, im in enumerate(ims):
            mon.paste(im, ((j % cols) * 360, (j // cols) * 360))
        mon.save(os.path.join(args.out_dir, "fit_anchors_montage.png"))
        print(f"Done (inspect-anchors). {len(tiles)} overlays + montage in {args.out_dir}")
        return

    fcx = interp1d(z_anchor, cx_a, bounds_error=False, fill_value=(cx_a[0], cx_a[-1]))
    fcy = interp1d(z_anchor, cy_a, bounds_error=False, fill_value=(cy_a[0], cy_a[-1]))

    # Per winding: fixed arc-length phi grid from the ref poly; interp coef-eval'd
    # radius across z. Stack the phi grids/interpolators; remember column ranges.
    z_list = sorted(z_to_path)[:: args.z_step]
    win = []                                   # (phi_grid, cos, sin, fR) per winding
    total_cols = 0
    for k in range(nw):
        cref = coefs[k][ref]
        if cref is None:
            continue
        phig, s_total = arclen_phi(cref, eps)
        R = np.array([np.polyval(coefs[k][i], phig) if coefs[k][i] is not None
                      else np.polyval(cref, phig) for i in range(n)])
        fR = interp1d(z_anchor, R, axis=0, bounds_error=False,
                      fill_value=(R[0], R[-1]))
        th = seam + phig
        win.append((np.cos(th), np.sin(th), fR, len(phig)))
        total_cols += len(phig)
    print(f"  whole roll: {len(win)} windings, {total_cols} cols, {len(z_list)} z")

    K = (args.multitap_n - 1) // 2
    taps = np.arange(-K, K + 1) * args.multitap_delta_px
    strip = np.empty((len(z_list), total_cols), dtype=np.float32)
    for j, z in enumerate(z_list):
        img = norm_slice(load_ct(z_to_path[z]))
        cx, cy = float(fcx(z)), float(fcy(z))
        c = 0
        for cos_t, sin_t, fR, w in win:
            r = fR(z); acc = np.zeros(w)
            for off in taps:
                rr = r + off
                acc += bilinear_sample(img, cx + rr * cos_t, cy + rr * sin_t)
            strip[j, c:c + w] = (acc / len(taps)).astype(np.float32)
            c += w
        if j % 200 == 0 or j == len(z_list) - 1:
            print(f"  [{j + 1}/{len(z_list)}] z{z}", flush=True)

    np.save(os.path.join(args.out_dir, "wholeroll.npy"), strip)
    np.save(os.path.join(args.out_dir, "z_index.npy"), np.array(z_list))
    # downscaled overview (too wide for 1:1 png)
    from PIL import Image as _Im
    step = max(1, total_cols // 6000)
    sm = strip[:, ::step]; lo, hi = np.percentile(sm, [1, 99])
    sm = np.clip((sm - lo) / (hi - lo + 1e-6), 0, 1)
    _Im.fromarray((sm * 255).astype(np.uint8)).save(
        os.path.join(args.out_dir, "wholeroll_overview.png"))
    print(f"Done. whole roll {strip.shape}. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
