"""Overlay CT + emulsion seg + walk fit for the full-density and v12 caches at a
common z, drawing both detected seam rays. Confirms the seam-detection bug: the
full-density walk detected seam=27.5deg (wrong) while v12 got 60deg (true), so the
full walk spans through the REAL seam at 60deg and crosses windings there.
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
            return (np.argmax(probs, axis=-1) == 2).astype(np.uint8)
    return None


def anchor_at(cache, z):
    d = np.load(cache, allow_pickle=True)
    za = np.asarray(d["z_anchor"], float)
    a = int(np.argmin(np.abs(za - z)))
    return (d["paths"][a], float(d["cx_a"][a]), float(d["cy_a"][a]),
            float(d["seam"]), int(za[a]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-dir", default="01_Mickey_hdf")
    ap.add_argument("--seg-dir", default="01_Mickey_full_3d")
    ap.add_argument("--full", default="unwrapping/inr/results/walk_dense_full/matched_walks.npz")
    ap.add_argument("--v12", default="unwrapping/inr/results/walk_dense_v11/matched_walks.npz")
    ap.add_argument("--z", type=int, default=1282)
    ap.add_argument("--out", default="unwrapping/inr/results/seam_fit")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    ct = load_ct(args.ct_dir, args.z)
    emul = load_emul(args.seg_dir, args.z)
    lo, hi = np.percentile(ct, [1, 99])
    disp = np.clip((ct - lo) / (hi - lo + 1e-6), 0, 1)

    for cache, tag in [(args.full, "full-density"), (args.v12, "v12")]:
        paths, cx, cy, seam, za = anchor_at(cache, args.z)
        fig, ax = plt.subplots(1, 1, figsize=(13, 13))
        ax.imshow(disp, cmap="gray")
        # emulsion in translucent red
        ys, xs = np.where(emul > 0)
        ax.scatter(xs[::7], ys[::7], s=0.5, c="red", alpha=0.15)
        cmap = plt.get_cmap("hsv")
        for k, p in enumerate(paths):
            p = np.asarray(p)
            ax.plot(p[:, 0], p[:, 1], "-", color=cmap(k / max(1, len(paths))), lw=0.8)
        # draw seam rays: this cache's seam (solid cyan) + the other (dashed yellow)
        R = 1500
        for ang, col, ls, lab in [(seam, "cyan", "-", f"seam {math.degrees(seam):.1f}deg"),
                                  (math.radians(60), "yellow", "--", "true seam 60deg")]:
            ax.plot([cx, cx + R * math.cos(ang)], [cy, cy + R * math.sin(ang)],
                    ls, color=col, lw=2, label=lab)
        ax.plot(cx, cy, "c+", ms=12)
        ax.legend(fontsize=11, loc="upper right")
        ax.set_title(f"{tag}: z{za}, {len(paths)} windings, "
                     f"detected seam {math.degrees(seam):.1f}deg", fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        out = os.path.join(args.out, f"seamfit_{tag}_z{za}.png")
        plt.savefig(out, dpi=120); plt.close()
        print(f"{tag}: z{za} seam={math.degrees(seam):.1f}deg {len(paths)}w -> {out}",
              flush=True)


if __name__ == "__main__":
    main()
