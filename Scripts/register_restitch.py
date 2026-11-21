"""Map the restitched tomogram onto the ORIGINAL new-scan volume, slice by slice,
so the existing segmentation + walk geometry can be reused without recomputation.

Probe (2026-11-12, 3 slices): same orientation (identity NCC 0.93 vs <=0.70 for
all 7 flips/transposes); z offset drifts (+5 at z600, +7 at z1200, +8 at z1800);
in-plane shift drifts (x -1.7 -> -0.1 px). The RADIAL component of the local shift
field is a pure translation per slice (residual 0.06-0.09 px rms); tangential
phase-correlation along the rings is noise and is not used.

For each grid slice z_new: (1) coarse z match = best NCC over old slices
z_new+lo..z_new+hi on a 4x-downsampled annulus crop, refined to a FRACTIONAL
offset by a parabola through the NCC peak; (2) radial 1-D profile correlation at
5 radii x 24 azimuths (old at the nearest integer match vs new, 0.1 px steps);
(3) least-squares translation (tx, ty) from the radial shifts, with its residual.
Output npz: z_new, off (z_old = z_new + off), tx, ty (new = old + t), ncc, res.

  python Scripts/register_restitch.py --old 01_Mickey_sprockets \
      --new 01_Mickey_restitch/h5 --out .../restitch_map.npz
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unwrapping.inr.unroll_continuous import load_ct, discover_ct_slices


def prep(a):
    lo, hi = np.percentile(a[::4, ::4], [1, 99])
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))


def radial_shifts(o, n, cx, cy, radii, n_az=24):
    rows = []
    rs = np.arange(-60, 61, dtype=np.float32); ts = np.linspace(-40, 40, 41, dtype=np.float32)
    up = np.linspace(-60, 60, 1201)
    lags = np.arange(-50, 51)
    for r in radii:
        for k in range(n_az):
            th = 2 * math.pi * k / n_az
            ux, uy = math.cos(th), math.sin(th)
            x, y = cx + r * ux, cy + r * uy
            X = (x + np.outer(rs, np.full(41, ux)) - np.outer(np.ones_like(rs), uy * ts)).astype(np.float32)
            Y = (y + np.outer(rs, np.full(41, uy)) + np.outer(np.ones_like(rs), ux * ts)).astype(np.float32)
            po = cv2.remap(o, X, Y, cv2.INTER_LINEAR).mean(1)
            pn = cv2.remap(n, X, Y, cv2.INTER_LINEAR).mean(1)
            if po.std() < 1e-3 or pn.std() < 1e-3:
                continue
            a = np.interp(up, rs, po - po.mean()); b = np.interp(up, rs, pn - pn.mean())
            cc = np.array([np.dot(a[60:-60], np.roll(b, -l)[60:-60]) for l in lags])
            rows.append((ux, uy, lags[int(np.argmax(cc))] * 0.1))
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True); ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=20)
    ap.add_argument("--dense-at", default="487,974,1461,1948",
                    help="sub-scan junctions (restitch z): sampled every --dense-step "
                         "within +-30 slices, where the mapping may jump")
    ap.add_argument("--dense-step", type=int, default=4)
    ap.add_argument("--off-lo", type=int, default=0)
    ap.add_argument("--off-hi", type=int, default=15)
    ap.add_argument("--cx", type=float, default=1885.0)
    ap.add_argument("--cy", type=float, default=1885.0)
    args = ap.parse_args()
    O = discover_ct_slices(args.old); N = discover_ct_slices(args.new)
    print(f"old {len(O)} slices z{min(O)}-{max(O)}; new {len(N)} slices z{min(N)}-{max(N)}",
          flush=True)
    grid = set(range(min(N) + 10, max(N) - 10, args.step))
    for j in [int(v) for v in args.dense_at.split(",")]:
        grid |= set(range(j - 30, j + 31, args.dense_step))
    grid = sorted(z for z in grid if z in N)
    cache = {}

    def old(z):
        if z not in cache:
            if len(cache) > 40:
                cache.pop(next(iter(cache)))
            cache[z] = prep(load_ct(O[z]))
        return cache[z]

    out = []
    for i, zn in enumerate(grid):
        n = prep(load_ct(N[zn]))
        W = min(n.shape[1], old(min(O)).shape[1])
        n = n[:, :W]
        crop = (slice(600, 3100), slice(600, min(3100, W)))
        nd = cv2.resize(n[crop], None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        sc = []
        for d in range(args.off_lo, args.off_hi + 1):
            zo = zn + d
            if zo not in O:
                continue
            od = cv2.resize(old(zo)[:, :W][crop], None, fx=0.25, fy=0.25,
                            interpolation=cv2.INTER_AREA)
            sc.append((d, ncc(nd, od)))
        if len(sc) < 3:
            continue
        ds = np.array([s[0] for s in sc]); vs = np.array([s[1] for s in sc])
        k = int(np.argmax(vs))
        frac = 0.0
        if 0 < k < len(vs) - 1:
            den = vs[k - 1] - 2 * vs[k] + vs[k + 1]
            frac = float(np.clip(0.5 * (vs[k - 1] - vs[k + 1]) / den, -0.5, 0.5)) if den < 0 else 0.0
        off = float(ds[k]) + frac
        R = radial_shifts(old(zn + int(ds[k]))[:, :W], n, args.cx, args.cy,
                          (900, 1100, 1300, 1500, 1650))
        tx = ty = res = np.nan
        if len(R) >= 20:
            A = R[:, :2]; b = R[:, 2]
            t, *_ = np.linalg.lstsq(A, b, rcond=None)
            r = b - A @ t
            keep = np.abs(r) < 3 * np.median(np.abs(r)) + 0.1      # drop lone mismatches
            t, *_ = np.linalg.lstsq(A[keep], b[keep], rcond=None)
            tx, ty = t; res = float(np.sqrt(np.mean((b[keep] - A[keep] @ t) ** 2)))
        out.append((zn, off, tx, ty, float(vs[k]), res))
        if i % 10 == 0:
            print(f"  [{i + 1}/{len(grid)}] z_new {zn}: off {off:+.2f}  t=({tx:+.2f},{ty:+.2f})"
                  f"  ncc {vs[k]:.3f}  radial res {res:.2f}px", flush=True)
    a = np.array(out)
    np.savez(args.out, z_new=a[:, 0], off=a[:, 1], tx=a[:, 2], ty=a[:, 3], ncc=a[:, 4],
             res=a[:, 5])
    print(f"wrote {args.out}: {len(a)} slices; off {np.nanmin(a[:,1]):+.2f}..{np.nanmax(a[:,1]):+.2f}, "
          f"tx {np.nanmin(a[:,2]):+.2f}..{np.nanmax(a[:,2]):+.2f}, ty {np.nanmin(a[:,3]):+.2f}.."
          f"{np.nanmax(a[:,3]):+.2f}, radial res median {np.nanmedian(a[:,5]):.2f} px", flush=True)


if __name__ == "__main__":
    main()
