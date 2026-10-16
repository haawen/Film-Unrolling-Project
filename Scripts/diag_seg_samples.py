"""Return ~50 samples across the full scroll: original CT + segmentation (no walk),
to judge the fresh full-scroll seg quality directly. For each sampled z: a
side-by-side [CT | seg-overlay] where emulsion(class2)=red, base(class1)=blue over
the CT. Saves individuals + a grid montage. Also prints per-slice class pixel
counts so a degenerate/empty seg is obvious.
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
from PIL import Image


def find_chunk(seg_dir, z):
    for p in sorted(glob.glob(os.path.join(seg_dir, "volume_*_Probabilities.h5"))):
        m = re.search(r"volume_(\d+)-(\d+)_Prob", os.path.basename(p))
        z0, z1 = int(m.group(1)), int(m.group(2))
        if z0 <= z <= z1:
            return p, z - z0
    return None, None


def load_ct(ct_dir, z):
    for p in glob.glob(os.path.join(ct_dir, "*.h5")):
        m = re.search(r"_(\d+)\.h5$", os.path.basename(p))
        if m and int(m.group(1)) == z:
            with h5py.File(p, "r") as f:
                return np.asarray(f["image"], np.float32)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-dir", default="01_Mickey_hdf")
    ap.add_argument("--seg-dir", default="01_Mickey_full_3d")
    ap.add_argument("--z0", type=int, default=832)
    ap.add_argument("--z1", type=int, default=2015)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="unwrapping/inr/results/seg_samples")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    zs = np.unique(np.linspace(args.z0, args.z1, args.n).round().astype(int))
    tiles = []
    for z in zs:
        ct = load_ct(args.ct_dir, z)
        pth, idx = find_chunk(args.seg_dir, z)
        if ct is None or pth is None:
            print(f"z{z}: MISSING (ct={ct is not None}, seg={pth is not None})",
                  flush=True)
            continue
        with h5py.File(pth, "r") as f:
            probs = f["exported_data"][idx].astype(np.float32)
        seg = np.argmax(probs, axis=-1)
        if seg.shape != ct.shape:              # fresh full-seg is stored TRANSPOSED
            seg = seg.T                        # (W,H) -> (H,W) to match the CT
        n_air = int((seg == 0).sum()); n_base = int((seg == 1).sum())
        n_emul = int((seg == 2).sum())
        lo, hi = np.percentile(ct, [1, 99])
        disp = np.clip((ct - lo) / (hi - lo + 1e-6), 0, 1)
        # seg overlay: CT gray + base(blue) + emulsion(red)
        rgb = np.stack([disp, disp, disp], -1)
        rgb[seg == 1] = rgb[seg == 1] * 0.4 + np.array([0.1, 0.3, 1.0]) * 0.6
        rgb[seg == 2] = np.array([1.0, 0.1, 0.1])
        # side-by-side, downscaled to 900px wide each
        ds = max(1, ct.shape[1] // 900)
        left = (disp[::ds, ::ds] * 255).astype(np.uint8)
        right = (rgb[::ds, ::ds] * 255).astype(np.uint8)
        left3 = np.stack([left] * 3, -1)
        pair = np.concatenate([left3, right], axis=1)
        # label
        img = Image.fromarray(pair)
        outp = os.path.join(args.out, f"seg_z{z:04d}.png")
        img.save(outp)
        tiles.append((outp, z, n_base, n_emul))
        print(f"z{z}: base={n_base} emul={n_emul} air={n_air} "
              f"({'OK' if n_emul > 50000 else 'LOW-EMUL?!'})", flush=True)
    # montage grid of the seg-overlay halves (right panel), small
    cols = 5
    rows = int(math.ceil(len(tiles) / cols))
    cell = 380
    mon = Image.new("RGB", (cell * cols, cell * rows), "black")
    for i, (outp, z, nb, ne) in enumerate(tiles):
        im = Image.open(outp)
        # take right half (seg overlay), resize
        w, h = im.size
        seg_half = im.crop((w // 2, 0, w, h)).resize((cell, cell))
        mon.paste(seg_half, ((i % cols) * cell, (i // cols) * cell))
    mon.save(os.path.join(args.out, "seg_montage.png"))
    print(f"saved {len(tiles)} samples + seg_montage.png to {args.out}")


if __name__ == "__main__":
    main()
