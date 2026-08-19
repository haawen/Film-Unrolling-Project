"""Overlay v12 (walk_dense_v11, ORIGINAL 25-chunk seg) vs s60 (walk_full_s60, FRESH
full seg) fits at the SAME z on the SAME CT, at zoom. Directly tests whether the
fresh full-segmentation makes the fit wobble (v12 smooth + s60 wobbly => fresh seg
is the culprit) or both look the same (=> not the seg).
"""
import argparse
import glob
import math
import os
import re

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_ct(ct_dir, z):
    for p in glob.glob(os.path.join(ct_dir, "*.h5")):
        m = re.search(r"_(\d+)\.h5$", os.path.basename(p))
        if m and int(m.group(1)) == z:
            with h5py.File(p, "r") as f:
                return np.asarray(f["image"], np.float32)
    return None


def anchor_at(cache, z):
    d = np.load(cache, allow_pickle=True)
    za = np.asarray(d["z_anchor"], float)
    a = int(np.argmin(np.abs(za - z)))
    return d["paths"][a], float(d["cx_a"][a]), float(d["cy_a"][a]), int(za[a])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-dir", default="01_Mickey_hdf")
    ap.add_argument("--v12", default="unwrapping/inr/results/walk_dense_v11/matched_walks.npz")
    ap.add_argument("--s60", default="unwrapping/inr/results/walk_full_s60/matched_walks.npz")
    ap.add_argument("--z", type=int, default=1432)
    ap.add_argument("--out", default="unwrapping/inr/results/walk_full_s60/fitcompare")
    ap.add_argument("--azs", default="90,150,210,270")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ct = load_ct(args.ct_dir, args.z)
    lo, hi = np.percentile(ct, [1, 99]); disp = np.clip((ct - lo) / (hi - lo + 1e-6), 0, 1)
    p_v12, cx1, cy1, z1 = anchor_at(args.v12, args.z)
    p_s60, cx2, cy2, z2 = anchor_at(args.s60, args.z)
    print(f"v12 anchor z{z1} ({len(p_v12)}w), s60 anchor z{z2} ({len(p_s60)}w)", flush=True)
    azs = [math.radians(float(x)) for x in args.azs.split(",")]
    wob = math.radians(11)
    fig, axes = plt.subplots(2, 2, figsize=(22, 20))
    for ax, az in zip(axes.ravel(), azs):
        cx, cy = cx2, cy2
        allp = np.vstack([np.asarray(p) for p in p_s60])
        rr = np.hypot(allp[:, 0] - cx, allp[:, 1] - cy)
        rs = np.linspace(rr.min(), rr.max(), 60); aa = np.linspace(az - wob, az + wob, 60)
        gx = cx + np.outer(rs, np.cos(aa)); gy = cy + np.outer(rs, np.sin(aa))
        x0, x1 = int(max(0, gx.min())), int(min(ct.shape[1], gx.max()))
        y0, y1 = int(max(0, gy.min())), int(min(ct.shape[0], gy.max()))
        ax.imshow(disp[y0:y1, x0:x1], cmap="gray")
        for p in p_v12:
            p = np.asarray(p)
            m = (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)
            ax.plot(p[m, 0] - x0, p[m, 1] - y0, "-", color="lime", lw=1.5)
        for p in p_s60:
            p = np.asarray(p)
            m = (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)
            ax.plot(p[m, 0] - x0, p[m, 1] - y0, "-", color="red", lw=1.0)
        ax.plot([], [], "-", color="lime", lw=1.5, label="v12 (orig seg)")
        ax.plot([], [], "-", color="red", lw=1.0, label="s60 (fresh full seg)")
        ax.legend(fontsize=11, loc="upper right")
        ax.set_title(f"z{args.z} az{math.degrees(az):.0f}deg", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    out = os.path.join(args.out, f"fitcompare_z{args.z}.png")
    plt.savefig(out, dpi=115); plt.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
