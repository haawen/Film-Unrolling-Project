"""Compare emulsion (class 2) segmentation between the original 25-chunk seg
(01_Mickey_3d) and the fresh full-scroll seg (01_Mickey_full_3d) at matched z.

Hypothesis: the full-density walk's dark-sweep artifacts come from the fresh seg
having patchier/dashier emulsion than the original, so the walk samples off the
emulsion. Reports per-z: emulsion pixel count, and dashiness (mean # of class-2
runs per ray over many rays) — more/short runs = dashier = worse for the walk.
Saves side-by-side emulsion masks.
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


def find_chunk(seg_dir, z):
    for p in sorted(glob.glob(os.path.join(seg_dir, "volume_*_Probabilities.h5"))):
        m = re.search(r"volume_(\d+)-(\d+)_Prob", os.path.basename(p))
        z0, z1 = int(m.group(1)), int(m.group(2))
        if z0 <= z <= z1:
            return p, z - z0
    return None, None


def load_emul(seg_dir, z):
    p, idx = find_chunk(seg_dir, z)
    if p is None:
        return None
    with h5py.File(p, "r") as f:
        ds = f["exported_data"]
        probs = ds[idx].astype(np.float32)          # (H, W, 3)
    seg = np.argmax(probs, axis=-1)
    return (seg == 2).astype(np.uint8)


def dashiness(emul, n_rays=360):
    """Mean number of class-2 runs per ray from the image center (more = dashier)."""
    H, W = emul.shape
    cy, cx = H / 2, W / 2
    rmax = min(H, W) / 2 - 5
    rr = np.arange(30, rmax, 1.0)
    counts = []
    for a in np.linspace(0, 2 * math.pi, n_rays, endpoint=False):
        xs = np.clip((cx + rr * math.cos(a)).astype(int), 0, W - 1)
        ys = np.clip((cy + rr * math.sin(a)).astype(int), 0, H - 1)
        on = emul[ys, xs] > 0
        runs = int(np.sum((~on[:-1]) & on[1:])) + int(on[0])
        counts.append(runs)
    return float(np.mean(counts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default="01_Mickey_3d")
    ap.add_argument("--full", default="01_Mickey_full_3d")
    ap.add_argument("--zs", default="846,1282,1732")
    ap.add_argument("--out", default="unwrapping/inr/results/seg_compare")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    zs = [int(x) for x in args.zs.split(",")]
    fig, axes = plt.subplots(len(zs), 2, figsize=(14, 6 * len(zs)))
    if len(zs) == 1:
        axes = axes[None, :]
    for i, z in enumerate(zs):
        eo = load_emul(args.orig, z)
        ef = load_emul(args.full, z)
        for j, (e, tag) in enumerate([(eo, "orig 25-chunk"), (ef, "full-scroll")]):
            ax = axes[i, j]
            if e is None:
                ax.set_title(f"z{z} {tag}: MISSING"); ax.axis("off"); continue
            npx = int(e.sum()); dash = dashiness(e)
            ax.imshow(e, cmap="gray")
            ax.set_title(f"z{z} {tag}: {npx} emul px, dashiness {dash:.1f} runs/ray",
                         fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            print(f"z{z} {tag}: emul_px={npx} dashiness={dash:.2f}", flush=True)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "seg_compare.png"), dpi=110)
    print(f"saved {args.out}/seg_compare.png")


if __name__ == "__main__":
    main()
