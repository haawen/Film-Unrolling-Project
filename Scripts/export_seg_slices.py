"""Export the full-resolution segmentation MASK for chosen slices of each chunk.

The walk's segmentation lives as nnU-Net softmax (`volume_*_Probabilities.h5`,
key `exported_data`, shape (Z,H,W,C)) at ~3.2 GB per 20-slice chunk. Pulling those
to look at one slice would be ~32 GB for ten chunks; argmaxing on the cluster and
shipping uint8 labels is ~14 MB per slice.

Emits, per requested slice:
  seg_z{z}.h5   key `seg`, uint8 (H,W), 0=air 1=film base 2=emulsion
  seg_z{z}.png  the same, colourised, for a quick look

Usage:
  python Scripts/export_seg_slices.py --pairs-dir newscan_z10/pairs \
      --slice-index 10 --out-dir unwrapping/inr/results/newscan_z10/segslices
"""
import argparse
import glob
import os
import re

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--slice-index", type=int, default=10,
                    help="Index WITHIN the chunk. The walk runs --slice-frac 1.0, "
                         "so anchor N of a chunk is its slice N; the fit overlays "
                         "were drawn at anchor 10.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    probs = sorted(glob.glob(os.path.join(args.pairs_dir, "*_Probabilities.h5")))
    if not probs:
        raise SystemExit(f"no *_Probabilities.h5 in {args.pairs_dir}")
    print(f"{len(probs)} chunks; slice index {args.slice_index}")

    for p in probs:
        m = re.search(r"volume_(\d+)-(\d+)_Probabilities\.h5$", os.path.basename(p))
        if not m:
            print(f"  skip (unparsable name): {os.path.basename(p)}")
            continue
        z0 = int(m.group(1))
        with h5py.File(p, "r") as f:
            ds = f["exported_data"]
            i = min(args.slice_index, ds.shape[0] - 1)
            seg = np.argmax(ds[i].astype(np.float32), axis=-1).astype(np.uint8)
        z = z0 + i
        out = os.path.join(args.out_dir, f"seg_z{z:04d}.h5")
        with h5py.File(out, "w") as f:
            d = f.create_dataset("seg", data=seg, compression="gzip",
                                 compression_opts=1)
            d.attrs["source"] = os.path.basename(p)
            d.attrs["slice_in_chunk"] = i
            d.attrs["z"] = z
            d.attrs["classes"] = "0=air 1=film base 2=emulsion"
        frac = [float((seg == c).mean()) for c in (0, 1, 2)]
        print(f"  z{z}: {seg.shape} air {frac[0]*100:.1f}% base {frac[1]*100:.1f}% "
              f"emulsion {frac[2]*100:.1f}% -> {os.path.basename(out)}", flush=True)

        if not args.no_png:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            rgb = np.zeros(seg.shape + (3,), np.uint8)
            rgb[seg == 1] = (90, 90, 90)       # film base
            rgb[seg == 2] = (255, 60, 60)      # emulsion
            plt.imsave(os.path.join(args.out_dir, f"seg_z{z:04d}.png"), rgb)


if __name__ == "__main__":
    main()
