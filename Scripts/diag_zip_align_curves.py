"""Pin the OLD -> NEW-reconstruction transform using the DELIVERED WINDING CURVES
as the template, and report the acceptance test in one go.

    p_zip = s * R(theta) @ (p_old - c_old) + c_zip          (4 free params)

WHY THIS ESTIMATOR. Correlating the two reconstructions' IMAGES failed twice:
(1) high-passed 4x-decimated slices -> ncc 0.06-0.09, rotation pinned at the grid
edge (decimation aliased the 21/26px ring texture); (2) polar mask/intensity
correlation -> scale fine (1.2515 +- 0.1%) but rotation only ~-63 +-4deg with gross
outliers, because the two reconstructions differ in resolution and noise so their
TEXTURE does not correlate -- only their GEOMETRY does. So use geometry directly:
map the 36 validated winding curves into a zip slice and maximise the zip intensity
sampled along them. The curves sit on the EMULSION, a ~5 zip-px bright sublayer, so
the objective is sharp exactly where mask correlation was flat, and 36 curves
spanning r623-1347 (old) constrain 4 parameters heavily.

It doubles as the acceptance test: for the best transform it reports the signed
radial offset from each mapped curve point to the nearest local intensity maximum
(the band centre). Target: |median| and p90 within a few px -- the sampling line
must stay inside the 18px FILM layer (perf bands carry no picture content, so the
4px emulsion tolerance does NOT apply there).

Runs on Merlin7 where both inputs already live: matched_walks.npz and the uploaded
zip slices. Uses the PICTURE-BAND OVERLAP of the upload (zip z 470-540 and
1930-1975 -> old z ~832-890 and ~1990-2015): real walked geometry at nearly the
right z, and the two ends the perf-band extrapolation anchors to.

Usage (see Scripts/slurm/slurm_zip_align.sh):
  python Scripts/diag_zip_align_curves.py \
      --geom unwrapping/inr/results/walk_full_final/matched_walks.npz \
      --zip-dir 01_Mickey_sprockets --out-dir unwrapping/inr/results/zip_align \
      --zip-z 480 500 520 1940 1960 --s0 1.2515 --theta0 -63
"""
import argparse
import glob
import json
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter, map_coordinates


def load_zip_slice(zip_dir, z):
    p = os.path.join(zip_dir, f"Mickey_merged_{z:04d}.h5")
    if not os.path.exists(p):
        raise SystemExit(f"missing {p}")
    with h5py.File(p, "r") as f:
        return np.asarray(f["image"], np.float32)


def otsu(a, nbins=512):
    h, e = np.histogram(a, bins=nbins)
    c = 0.5 * (e[1:] + e[:-1])
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    s0 = np.cumsum(h * c); s1 = s0[-1] - s0
    ok = (w0 > 0) & (w1 > 0)
    var = np.zeros(nbins)
    var[ok] = w0[ok] * w1[ok] * (s0[ok] / w0[ok] - s1[ok] / w1[ok]) ** 2
    return float(c[int(np.argmax(var))])


def zip_centre(img):
    """Provisional zip centre: centroid of the film mask (single Otsu is correct
    for the zip -- it has no background spike, unlike the old stitch)."""
    m = img > otsu(img)
    ys, xs = np.nonzero(m)
    return float(ys.mean()), float(xs.mean())


def ridge(img, sigma=8.0):
    """High-pass so only the thin EMULSION sublayer peaks.

    The raw intensity is a terrible objective here: the film layer is 18 old px =
    22 zip px on a 26 zip px winding spacing, so a curve displaced anywhere within
    +-11px is still on film and the mean intensity hardly changes -- a plateau, not
    a peak (measured: offsets scattered ~uniformly over +-20px, median +0.5 px
    meaning nothing). Subtracting a Gaussian leaves a sharp ~5px ridge at the
    emulsion, giving a 26px-periodic objective with a real maximum.
    """
    return img - gaussian_filter(img, sigma)


def transform(P, c_old, s, theta_deg, c_zip, flip=False):
    """P (N,2) old (x,y) -> zip (x,y). `flip` mirrors x before rotating.

    A similarity CANNOT absorb a handedness change, and this dataset family has
    already produced one (the H<->W seg transpose = a reflection about the near
    diagonal). With a mirror present every scale/rotation estimator fails in a
    different way, which is exactly what was observed -- so flip is searched.
    """
    th = math.radians(theta_deg)
    ct, st = math.cos(th), math.sin(th)
    dx = P[:, 0] - c_old[1]
    dy = P[:, 1] - c_old[0]
    if flip:
        dx = -dx
    return np.stack([s * (ct * dx - st * dy) + c_zip[1],
                     s * (st * dx + ct * dy) + c_zip[0]], 1)


def objective(img, P, c_old, s, theta, c_zip, flip=False):
    Q = transform(P, c_old, s, theta, c_zip, flip)
    v = map_coordinates(img, [Q[:, 1], Q[:, 0]], order=1, mode="constant", cval=0.0)
    return float(v.mean())


def band_offsets(img, P, c_old, s, theta, c_zip, half=13.0, step=0.5, flip=False):
    """Signed radial offset (zip px) from each mapped point to the nearest local
    intensity maximum along the RADIAL direction, searched +-half (< half the 26px
    winding spacing, so a neighbouring band can never be picked)."""
    Q = transform(P, c_old, s, theta, c_zip, flip)
    rx = Q[:, 0] - c_zip[1]; ry = Q[:, 1] - c_zip[0]
    L = np.hypot(rx, ry) + 1e-9
    ux, uy = rx / L, ry / L
    ts = np.arange(-half, half + 1e-9, step)
    prof = np.stack([map_coordinates(img, [Q[:, 1] + t * uy, Q[:, 0] + t * ux],
                                    order=1, mode="constant", cval=0.0)
                     for t in ts])                       # (nt, N)
    return ts[np.argmax(prof, axis=0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", required=True)
    ap.add_argument("--zip-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--zip-z", type=int, nargs="+", required=True)
    ap.add_argument("--z-map", type=float, nargs=2, default=[1.2722, -588.5],
                    help="z_zip = a*z_old + b (only ~1%% accuracy needed)")
    ap.add_argument("--s0", type=float, default=1.2515)
    ap.add_argument("--theta0", type=float, default=-63.0)
    ap.add_argument("--sub", type=int, default=4, help="use every Nth curve point")
    ap.add_argument("--sweeps", type=int, default=3)
    ap.add_argument("--ridge-sigma", type=float, default=8.0,
                    help="high-pass sigma (zip px) isolating the emulsion ridge")
    ap.add_argument("--edge-windings", type=int, default=2,
                    help="windings excluded at EACH end of the roll (film ends)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    g = np.load(args.geom, allow_pickle=True)
    z_anchor = np.asarray(g["z_anchor"], float)
    cx_a = np.asarray(g["cx_a"], float); cy_a = np.asarray(g["cy_a"], float)
    paths = g["paths"]
    print(f"geom: {len(paths)} anchors z{z_anchor.min():.0f}-{z_anchor.max():.0f}, "
          f"{len(paths[0])} windings", flush=True)

    a_z, b_z = args.z_map
    results = []
    for zz in args.zip_z:
        z_old = (zz - b_z) / a_z
        ai = int(np.argmin(np.abs(z_anchor - z_old)))
        if abs(z_anchor[ai] - z_old) > 30:
            print(f"  zip z{zz}: no anchor near old z{z_old:.0f} -- skipped")
            continue
        img_raw = load_zip_slice(args.zip_dir, zz)
        img = ridge(img_raw, args.ridge_sigma)
        # end windings carry the film's tongue/termination: partial, non-circular
        # and heavily CLAMPED (resample_phi repeats the endpoint), so their points
        # are frozen at an arbitrary place and must not drive the fit.
        Ps, goods = [], []
        nw = len(paths[ai])
        for wi, w in enumerate(paths[ai]):
            W = np.asarray(w, float)[:: args.sub]
            ds = np.hypot(np.diff(W[:, 0]), np.diff(W[:, 1]))
            g = np.concatenate([[True], ds > 1e-6])          # clamped -> False
            if wi < args.edge_windings or wi >= nw - args.edge_windings:
                g[:] = False
            Ps.append(W); goods.append(g)
        P = np.concatenate(Ps); good = np.concatenate(goods)
        print(f"    points {len(P)}, usable {int(good.sum())} "
              f"({100 * good.mean():.0f}%; dropped {args.edge_windings} winding(s) "
              f"per end + clamped)", flush=True)
        c_old = (cy_a[ai], cx_a[ai])
        c_zip = list(zip_centre(img_raw))
        s, th = args.s0, args.theta0
        flip = False
        scan = []
        for fl in (False, True):
            for th_c in np.arange(-180, 180, 1.0):
                scan.append((objective(img, P[good], c_old, s, th_c, c_zip, fl),
                             fl, float(th_c)))
        best, flip, th = max(scan)
        top = sorted(scan, reverse=True)[:3]
        print(f"    full-circle scan: best flip={flip} theta {th:+.1f} obj {best:.1f}"
              f" | top3 {[(f'{v:.1f}', fl, t) for v, fl, t in top]}", flush=True)
        # how decisive is the flip? compare the best of each handedness
        b_no = max(v for v, fl, _ in scan if not fl)
        b_fl = max(v for v, fl, _ in scan if fl)
        print(f"    flip test: best no-flip {b_no:.1f} vs flip {b_fl:.1f} "
              f"({100 * (max(b_no, b_fl) / max(min(b_no, b_fl), 1e-9) - 1):+.1f}% "
              f"in favour of {'FLIP' if b_fl > b_no else 'NO-FLIP'})", flush=True)
        print(f"  zip z{zz} <-> old z{z_anchor[ai]:.0f} (anchor {ai}), "
              f"{len(P)} pts | init obj {best:.1f} centre "
              f"{c_zip[0]:.1f},{c_zip[1]:.1f}", flush=True)

        # coordinate sweeps, coarse -> fine (4 params, sharp objective)
        for it in range(args.sweeps):
            f = 0.25 ** it
            for th_c in th + np.arange(-8, 8.001, 0.5) * f:
                v = objective(img, P[good], c_old, s, th_c, c_zip, flip)
                if v > best:
                    best, th = v, float(th_c)
            for s_c in s + np.arange(-0.008, 0.00801, 0.0005) * f:
                v = objective(img, P[good], c_old, s_c, th, c_zip, flip)
                if v > best:
                    best, s = v, float(s_c)
            for dy in np.arange(-12, 12.001, 1.0) * f:
                for dx in np.arange(-12, 12.001, 1.0) * f:
                    cz = [c_zip[0] + dy, c_zip[1] + dx]
                    v = objective(img, P[good], c_old, s, th, cz, flip)
                    if v > best:
                        best, c_zip = v, cz
            print(f"    sweep {it}: obj {best:.1f} s {s:.4f} theta {th:+.3f} "
                  f"centre {c_zip[0]:.2f},{c_zip[1]:.2f}", flush=True)

        # ── acceptance + the RADIAL TREND, which is the sharp scale estimator ──
        # A scale error puts the band at r_map*(s_true/s), so
        #   offset(r) = r * (s_true/s - 1)  =>  slope = s_true/s - 1.
        # The objective's argmax cannot see this (a radial ramp averages to ~0 in
        # the median), which is why the curve fit and the annulus-boundary fit
        # disagreed by 0.7%. Search half-window 20px (< half the 26px spacing) so
        # p90 is not pinned at the bound as it was at half=13.
        off = band_offsets(img, P, c_old, s, th, c_zip, half=20.0, flip=flip)
        Q = transform(P, c_old, s, th, c_zip, flip)
        r_map = np.hypot(Q[:, 0] - c_zip[1], Q[:, 1] - c_zip[0])
        keep = good & (np.abs(off) < 19.5)          # drop bound-saturated points
        A = np.stack([r_map[keep], np.ones(keep.sum())], 1)
        (slope, icpt), *_ = np.linalg.lstsq(A, off[keep], rcond=None)
        s_corr = s * (1.0 + slope)
        resid = off[keep] - (slope * r_map[keep] + icpt)
        med = float(np.median(off[keep]))
        p90 = float(np.percentile(np.abs(off[keep]), 90))
        print(f"    ACCEPTANCE: offset median {med:+.2f} zip px  p90|.| {p90:.2f} "
              f"({100 * keep.mean():.0f}% of pts used, "
              f"{100 * (np.abs(off) >= 19.5).mean():.1f}% bound-saturated)",
              flush=True)
        print(f"    RADIAL TREND: slope {slope * 1e3:+.4f}e-3 px/px, intercept "
              f"{icpt:+.2f} => scale correction s {s:.4f} -> {s_corr:.4f}; "
              f"residual about the trend med |.| "
              f"{float(np.median(np.abs(resid))):.2f} p90 "
              f"{float(np.percentile(np.abs(resid), 90)):.2f} zip px", flush=True)
        results.append(dict(zip_z=zz, old_z=float(z_anchor[ai]), anchor=ai,
                            s=s, theta_deg=th, c_zip=c_zip, c_old=list(c_old),
                            obj=best, off_med=med, off_p90=p90,
                            slope=float(slope), s_corr=float(s_corr),
                            flip=bool(flip)))

        # fit-check overlay (a real image, per the standing rule)
        Q = transform(P, c_old, s, th, c_zip, flip)
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        lo, hi = np.percentile(img, [1, 99.5])
        for a_, (y0, y1, x0, x1) in zip(ax, [(0, img.shape[0], 0, img.shape[1]),
                                             (int(c_zip[0]) - 300, int(c_zip[0]) + 60,
                                              int(c_zip[1]) + 700,
                                              int(c_zip[1]) + 1100)]):
            a_.imshow(np.clip((img[y0:y1, x0:x1] - lo) / (hi - lo), 0, 1), cmap="gray")
            a_.plot(Q[:, 0] - x0, Q[:, 1] - y0, ".", ms=0.4, color="red")
            a_.set_xlim(0, x1 - x0); a_.set_ylim(y1 - y0, 0)
            a_.set_xticks([]); a_.set_yticks([])
        ax[0].set_title(f"zip z{zz}: mapped old curves (s={s:.4f} th={th:+.2f})")
        ax[1].set_title(f"zoom -- offset med {med:+.2f} p90 {p90:.2f} zip px")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"align_z{zz}.png"), dpi=130)
        plt.close()

    if results:
        s = np.array([r["s"] for r in results]); th = np.array([r["theta_deg"]
                                                               for r in results])
        print(f"\nACROSS {len(results)} SLICES (a single physical transform must be "
              f"CONSISTENT -- scatter here is the real error bar):")
        print(f"  s     {s.mean():.4f} +- {s.std():.4f} "
              f"({100 * s.std() / max(s.mean(), 1e-9):.3f}%; target 0.3%)")
        print(f"  theta {th.mean():+.3f} +- {th.std():.3f} deg (target 0.2-0.5)")
        sc = np.array([r["s_corr"] for r in results])
        print(f"  s_corr (from the radial trend) {sc.mean():.4f} +- {sc.std():.4f} "
              f"({100 * sc.std() / sc.mean():.3f}%)  <- USE THIS")
        print(f"  flip: {[r['flip'] for r in results]}")
        print(f"  offset median {np.mean([r['off_med'] for r in results]):+.2f} "
              f"zip px, p90 {np.mean([r['off_p90'] for r in results]):.2f}")
    with open(os.path.join(args.out_dir, "zip_align.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
