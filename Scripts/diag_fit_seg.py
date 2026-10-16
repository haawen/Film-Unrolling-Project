"""For each fit anchor: overlay CT + emulsion seg (red) + the walk fit, full-ring
AND a zoom wedge, so we can see whether (a) the emulsion seg is noisy/dashed, or
(b) the fit wobbles off a clean emulsion (= heal/track issue, not seg). Also draws
the RAW cached walk (pre-heal) vs the kept fit to isolate the z-heal's effect.
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

TWO_PI = 2 * math.pi


def load_ct(ct_dir, z):
    for p in glob.glob(os.path.join(ct_dir, "*.h5")):
        m = re.search(r"_(\d+)\.h5$", os.path.basename(p))
        if m and int(m.group(1)) == z:
            with h5py.File(p, "r") as f:
                return np.asarray(f["image"], np.float32)
    return None


def load_emul(seg_dir, z):
    for p in sorted(glob.glob(os.path.join(seg_dir, "volume_*_Probabilities.h5"))):
        m = re.search(r"volume_(\d+)-(\d+)_Prob", os.path.basename(p))
        z0, z1 = int(m.group(1)), int(m.group(2))
        if z0 <= z <= z1:
            with h5py.File(p, "r") as f:
                probs = f["exported_data"][z - z0].astype(np.float32)
            return np.argmax(probs, axis=-1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-dir", default="01_Mickey_hdf")
    ap.add_argument("--seg-dir", default="01_Mickey_full_3d")
    ap.add_argument("--cache", default="unwrapping/inr/results/walk_full_s60/matched_walks.npz")
    ap.add_argument("--rawcache", default="unwrapping/inr/results/walk_full_s60/walk_anchors.npz")
    ap.add_argument("--anchors", default="0,600,900")
    ap.add_argument("--out", default="unwrapping/inr/results/walk_full_s60/fitseg")
    ap.add_argument("--wedge-deg", type=float, default=140.0,
                    help="Zoom azimuth center (abs deg).")
    ap.add_argument("--wedge-half", type=float, default=14.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = np.load(args.cache, allow_pickle=True)
    za = np.asarray(d["z_anchor"], float)
    raw = np.load(args.rawcache, allow_pickle=True)
    zar = np.asarray(raw["z_anchor"], float)

    for a in [int(x) for x in args.anchors.split(",")]:
        z = int(za[a]); cx = float(d["cx_a"][a]); cy = float(d["cy_a"][a])
        kept = d["paths"][a]
        ar = int(np.argmin(np.abs(zar - z)))
        rawp = raw["paths"][ar]
        ct = load_ct(args.ct_dir, z); seg = load_emul(args.seg_dir, z)
        lo, hi = np.percentile(ct, [1, 99])
        disp = np.clip((ct - lo) / (hi - lo + 1e-6), 0, 1)
        ey, ex = np.where(seg == 2)
        # zoom window
        cw = math.radians(args.wedge_deg); wob = math.radians(args.wedge_half)
        allp = np.vstack([np.asarray(p) for p in kept])
        rr = np.hypot(allp[:, 0] - cx, allp[:, 1] - cy)
        rs = np.linspace(rr.min(), rr.max(), 60); aa = np.linspace(cw - wob, cw + wob, 60)
        gx = cx + np.outer(rs, np.cos(aa)); gy = cy + np.outer(rs, np.sin(aa))
        x0, x1 = int(max(0, gx.min())), int(min(ct.shape[1], gx.max()))
        y0, y1 = int(max(0, gy.min())), int(min(ct.shape[0], gy.max()))

        fig, axes = plt.subplots(1, 2, figsize=(24, 12))
        # LEFT: full ring, CT + emulsion(red) + kept fit(yellow)
        ax = axes[0]; ax.imshow(disp, cmap="gray")
        ax.scatter(ex[::9], ey[::9], s=0.4, c="red", alpha=0.25)
        for p in kept:
            p = np.asarray(p); ax.plot(p[:, 0], p[:, 1], "-", color="yellow", lw=0.6)
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   ec="cyan", lw=2))
        ax.set_title(f"z{z}: CT + emulsion(red) + fit(yellow)", fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
        # RIGHT: zoom, CT + emulsion(cyan pts) + kept fit(yellow) + RAW walk(magenta)
        ax = axes[1]; ax.imshow(disp[y0:y1, x0:x1], cmap="gray")
        m = (ex >= x0) & (ex < x1) & (ey >= y0) & (ey < y1)
        ax.scatter(ex[m] - x0, ey[m] - y0, s=3, c="cyan", alpha=0.4)
        for p in kept:
            p = np.asarray(p)
            mm = (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)
            ax.plot(p[mm, 0] - x0, p[mm, 1] - y0, "-", color="yellow", lw=1.4)
        for p in rawp:
            p = np.asarray(p)
            mm = (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)
            ax.plot(p[mm, 0] - x0, p[mm, 1] - y0, "--", color="magenta", lw=0.8)
        ax.plot([], [], "-", color="yellow", label="kept fit (healed)")
        ax.plot([], [], "--", color="magenta", label="raw walk (pre-heal)")
        ax.scatter([], [], s=3, c="cyan", label="emulsion px")
        ax.legend(fontsize=11, loc="upper right")
        ax.set_title(f"ZOOM z{z} az{args.wedge_deg:.0f}deg: does fit HUG emulsion?",
                     fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        out = os.path.join(args.out, f"fitseg_z{z}.png")
        plt.savefig(out, dpi=110); plt.close()
        print(f"z{z}: {len(kept)}w kept, {len(rawp)}w raw -> {out}", flush=True)


if __name__ == "__main__":
    main()
