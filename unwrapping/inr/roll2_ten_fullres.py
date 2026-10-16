"""Render the ten Roll2 anchors at NATIVE resolution: seg and walk, separately.

The three-panel figures are a preview -- each panel is ~920 px for a 3064 px
slice, a 3.3x downscale that hides exactly the thing worth judging (whether the
emulsion is continuous, and whether a path sits on its own band). This writes,
per anchor, two full-3064x3064 images:

    seg_z####.png   segmentation, native, no resampling at all
    walk_z####.png  the walked paths over the CT, rendered 1:1

Walk paths come from the `walk_z####.npz` the panel run already saved, so this
does not re-walk anything.

Usage (Merlin):
  python -m unwrapping.inr.roll2_ten_fullres \
      --chunk-dir 02_Sample_3d --label-dir 02_Sample_ten2d/labels \
      --walk-dir unwrapping/inr/results/roll2_ten2d \
      --out-dir unwrapping/inr/results/roll2_ten2d/fullres
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

Image.MAX_IMAGE_PIXELS = None


def ct_slice(z0, idx, chunk_dir):
    cand = [f for f in glob.glob(os.path.join(chunk_dir, "volume_%04d-*.h5" % z0))
            if "Probabilities" not in f]
    with h5py.File(cand[0], "r") as f:
        key = "volume" if "volume" in f else list(f.keys())[0]
        return f[key][idx].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-dir", default="02_Sample_3d")
    ap.add_argument("--label-dir", default="02_Sample_ten2d/labels")
    ap.add_argument("--walk-dir",
                    default="unwrapping/inr/results/roll2_ten2d")
    ap.add_argument("--out-dir",
                    default="unwrapping/inr/results/roll2_ten2d/fullres")
    ap.add_argument("--chunk-z0s",
                    default="200,360,540,700,880,1060,1220,1400,1580,1740")
    ap.add_argument("--idx", type=int, default=10)
    ap.add_argument("--prefix", default="Roll2Ten")
    ap.add_argument("--lw", type=float, default=1.3)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import tifffile
    for z0 in [int(x) for x in args.chunk_z0s.split(",")]:
        z = z0 + args.idx
        seg = np.asarray(tifffile.imread(
            os.path.join(args.label_dir, "%s_%04d.tif" % (args.prefix, z))))
        if seg.ndim == 3:
            seg = seg.squeeze()
        H, W = seg.shape

        # --- segmentation, exact native pixels (no matplotlib, no resampling)
        rgb = np.zeros((H, W, 3), np.uint8)
        rgb[seg == 1] = (120, 120, 120)          # film base
        rgb[seg == 2] = (255, 40, 40)            # emulsion
        Image.fromarray(rgb).save(
            os.path.join(args.out_dir, "seg_z%04d.png" % z))

        # --- walk over CT, rendered 1:1
        ct = ct_slice(z0, args.idx, args.chunk_dir)
        lo, hi = np.percentile(ct, [1, 99])
        disp = np.clip((ct - lo) / (hi - lo + 1e-6), 0, 1)
        d = np.load(os.path.join(args.walk_dir, "walk_z%04d.npz" % z))
        keys = sorted([k for k in d.files
                       if k.startswith("w") and k[1:].isdigit()],
                      key=lambda k: int(k[1:]))
        cmap = plt.get_cmap("hsv")
        fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(disp, cmap="gray", interpolation="nearest")
        for k, key in enumerate(keys):
            p = d[key]
            ax.plot(p[:, 0], p[:, 1], "-", lw=args.lw,
                    color=cmap(k / max(1, len(keys))))
        ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
        fig.savefig(os.path.join(args.out_dir, "walk_z%04d.png" % z), dpi=100)
        plt.close(fig)
        print("  z=%d  seg+walk %dx%d  (%d windings)" % (z, W, H, len(keys)),
              flush=True)

    print("-> %s" % args.out_dir)


if __name__ == "__main__":
    main()
