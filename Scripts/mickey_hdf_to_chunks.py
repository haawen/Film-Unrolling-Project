"""Chunk the full continuous Mickey scroll (01_Mickey_hdf, 1184 per-slice HDF5s)
into CONTIGUOUS 20-slice volumes for nnU-Net 3D inference.

01_Mickey_hdf/01_Mickey_Stitch_Stitch_Export_{z:04d}.h5  (key 'image', (H,W) uint16)
    -> <out-dir>/volume_{z0:04d}-{z1:04d}.h5  (key 'volume', (Z,H,W) uint16)

Matches the format hdf5_to_nifti_for_predict.py expects (key 'volume'). Unlike the
gapped 01_Mickey_3d (25 windows), this covers EVERY slice so the whole scroll can be
segmented and walked densely.

Usage:
  python Scripts/mickey_hdf_to_chunks.py --in-dir 01_Mickey_hdf --out-dir 01_Mickey_full_3d \
      [--chunk-z 20] [--z-start 0832 --z-end 2015]   # z-range optional (smoke test)
"""
import argparse
import glob
import os
import re

import h5py
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chunk-z", type=int, default=20)
    ap.add_argument("--z-start", type=int, default=None, help="First z (inclusive).")
    ap.add_argument("--z-end", type=int, default=None, help="Last z (inclusive).")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    files = glob.glob(os.path.join(args.in_dir, "*.h5"))
    zof = []
    for f in files:
        m = re.search(r"(\d{4})\.h5$", os.path.basename(f))
        if m:
            zof.append((int(m.group(1)), f))
    zof.sort()
    if args.z_start is not None:
        zof = [(z, f) for z, f in zof if z >= args.z_start]
    if args.z_end is not None:
        zof = [(z, f) for z, f in zof if z <= args.z_end]
    if not zof:
        raise SystemExit("no matching slices")
    print(f"{len(zof)} slices z{zof[0][0]}-{zof[-1][0]}; chunk-z {args.chunk_z}")

    for i in range(0, len(zof), args.chunk_z):
        grp = zof[i:i + args.chunk_z]
        z0, z1 = grp[0][0], grp[-1][0]
        out = os.path.join(args.out_dir, f"volume_{z0:04d}-{z1:04d}.h5")
        if os.path.exists(out):
            print(f"  {os.path.basename(out)} exists, skip"); continue
        vol = []
        for _, f in grp:
            with h5py.File(f, "r") as h:
                vol.append(np.asarray(h["image"], dtype=np.uint16))
        vol = np.stack(vol, axis=0)                       # (Z, H, W)
        with h5py.File(out, "w") as h:
            h.create_dataset("volume", data=vol, compression="gzip", compression_opts=1)
        print(f"  wrote {os.path.basename(out)} {vol.shape}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
