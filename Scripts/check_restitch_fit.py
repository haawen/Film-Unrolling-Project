"""Fit check for rendering the RESTITCHED tomogram with the original geometry.

For a list of original-frame z: map z -> (restitched slice, t) with the
register_restitch npz, take the delivered winding paths (matched_walks.npz from the
approved order_fix3 inspect run), and
  1. QUANTITATIVE: along every path, the emulsion peak position relative to the
     path (radial profile -4..+4 px, parabola-refined) in the OLD slice and in the
     mapped NEW slice. A correct mapping reproduces the old distribution; a 1-2 px
     mapping error shifts it by that much.
  2. VISUAL: full-slice overlay of all windings on the new slice + zoom crops at
     three azimuths (labelled), for user approval before any render.

  python Scripts/check_restitch_fit.py --matched .../order_fix3/matched_walks.npz \
      --remap .../restitch_map.npz --old 01_Mickey_sprockets --new 01_Mickey_restitch/h5 \
      --out-dir .../diag/restitch_fit
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unwrapping.inr.unroll_continuous import load_ct, norm_slice, discover_ct_slices
from unwrapping.inr.sanity_winding_strip import bilinear_sample
from unwrapping.inr.unroll_walk_wholeroll_v13 import _z_map


def peak_offsets(img, X, Y):
    dx, dy = np.gradient(X), np.gradient(Y)
    L = np.hypot(dx, dy) + 1e-9
    nx, ny = -dy / L, dx / L
    offs = np.arange(-4, 5)
    P = np.stack([bilinear_sample(img, X + o * nx, Y + o * ny) for o in offs])
    i = np.clip(P.argmax(0), 1, len(offs) - 2); c = np.arange(P.shape[1])
    y0, y1, y2 = P[i - 1, c], P[i, c], P[i + 1, c]
    den = y0 - 2 * y1 + y2
    sub = np.where(den < -1e-6, 0.5 * (y0 - y2) / np.where(den < -1e-6, den, -1), 0)
    return offs[i] + np.clip(sub, -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", required=True); ap.add_argument("--remap", required=True)
    ap.add_argument("--old", required=True); ap.add_argument("--new", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--z", default="520,700,900,980,1100,1300,1468,1500,1700,1900")
    ap.add_argument("--zoom-az", default="45,170,290")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    d = np.load(args.matched, allow_pickle=True)
    za = np.asarray(d["z_anchor"], float); P = d["paths"]
    cx, cy = np.asarray(d["cx_a"], float), np.asarray(d["cy_a"], float)
    nw = len(P[0])
    maps = []
    for k in range(nw):
        X = np.stack([np.asarray(P[a][k], np.float32)[:, 0] for a in range(len(za))])
        Y = np.stack([np.asarray(P[a][k], np.float32)[:, 1] for a in range(len(za))])
        maps.append((_z_map(za, X), _z_map(za, Y)))
    R = np.load(args.remap)
    zog = R["z_new"] + R["off"]; o = np.argsort(zog)
    zog, zng, txg, tyg = zog[o], R["z_new"][o], R["tx"][o], R["ty"][o]
    ok = np.isfinite(txg) & np.isfinite(tyg)
    O = discover_ct_slices(args.old); N = discover_ct_slices(args.new)
    navail = np.array(sorted(N))
    print(" z_orig  z_new     t(px)        peak-offset OLD med/p10/p90   NEW med/p10/p90   "
          "|NEW-OLD| med", flush=True)
    cmap = plt.get_cmap("hsv")
    for z in [int(v) for v in args.z.split(",")]:
        zn = float(np.interp(z, zog, zng)); zi = int(navail[np.argmin(np.abs(navail - round(zn)))])
        tx, ty = float(np.interp(z, zog[ok], txg[ok])), float(np.interp(z, zog[ok], tyg[ok]))
        io_, in_ = norm_slice(load_ct(O[z])), norm_slice(load_ct(N[zi]))
        po, pn = [], []
        curves = []
        for k, (fX, fY) in enumerate(maps):
            X, Y = fX(z), fY(z)
            if 0 < k < nw - 1:                      # film ends are partial: skip in stats
                po.append(peak_offsets(io_, X, Y)); pn.append(peak_offsets(in_, X + tx, Y + ty))
            curves.append((X + tx, Y + ty))
        po, pn = np.concatenate(po), np.concatenate(pn)
        print(f" {z:5d}  {zi:5d}  ({tx:+.2f},{ty:+.2f})   "
              f"{np.median(po):+.2f} {np.percentile(po,10):+.2f} {np.percentile(po,90):+.2f}      "
              f"{np.median(pn):+.2f} {np.percentile(pn,10):+.2f} {np.percentile(pn,90):+.2f}       "
              f"{np.median(np.abs(pn - po)):.2f}", flush=True)
        fig, ax = plt.subplots(figsize=(11, 11)); ax.imshow(in_, cmap="gray")
        for k, (X, Y) in enumerate(curves):
            ax.plot(X, Y, "-", color=cmap(k / nw), lw=0.6)
        ax.set_title(f"restitched z{zi} (orig z{z}), mapped walk, {nw} windings")
        ax.set_xticks([]); ax.set_yticks([]); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"full_z{z}.png"), dpi=110); plt.close()
        a0 = int(np.argmin(np.abs(za - z)))
        for azd in [float(v) for v in args.zoom_az.split(",")]:
            th = math.radians(azd); r0 = 1050
            mx, my = cx[a0] + r0 * math.cos(th), cy[a0] + r0 * math.sin(th)
            x0, y0 = int(mx - 160), int(my - 160)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.imshow(in_[y0:y0 + 320, x0:x0 + 320], cmap="gray", extent=[x0, x0 + 320, y0 + 320, y0])
            for k, (X, Y) in enumerate(curves):
                m = (X > x0) & (X < x0 + 320) & (Y > y0) & (Y < y0 + 320)
                if m.any():
                    q = np.where(m, X, np.nan); ax.plot(q, Y, "-", color=cmap(k / nw), lw=1.2)
            ax.set_xlim(x0, x0 + 320); ax.set_ylim(y0 + 320, y0)
            ax.set_title(f"restitched z{zi} (orig z{z}) az {azd:.0f}"); ax.set_xticks([]); ax.set_yticks([])
            plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, f"zoom_z{z}_az{int(azd)}.png"), dpi=110)
            plt.close()


if __name__ == "__main__":
    main()
