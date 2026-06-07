"""Chunk a directory of per-slice TIFFs into HDF5 volumes of N consecutive z.

Output layout matches `01_Mickey_3d/`:
    volume_NNNN-MMMM.h5   key 'volume'  shape (chunk_z, H, W)  uint16

Filename numbers are the inclusive start/end z-index of the chunk in the
sorted TIFF file order. TIFFs are sorted by filename so they need to be
named with monotonic zero-padded indices (e.g. 0000.tiff ... 1935.tiff).
"""
import argparse
import os
import re
import sys
import numpy as np
import h5py
import tifffile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tiff-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--chunk-z", type=int, default=20,
                   help="Number of slices per HDF5 chunk (Mickey uses 20).")
    p.add_argument("--dtype", default="uint16")
    args = p.parse_args()

    files = sorted(
        f for f in os.listdir(args.tiff_dir)
        if f.lower().endswith((".tif", ".tiff"))
    )
    if not files:
        sys.exit(f"No .tif/.tiff in {args.tiff_dir}")
    print(f"Found {len(files)} TIFFs in {args.tiff_dir}")

    sample = tifffile.imread(os.path.join(args.tiff_dir, files[0]))
    print(f"Sample shape: {sample.shape} dtype={sample.dtype}")
    if sample.ndim != 2:
        sys.exit(f"Expected 2-D slices, got ndim={sample.ndim}")
    H, W = sample.shape

    os.makedirs(args.out_dir, exist_ok=True)
    z_total = len(files)
    chunk_z = args.chunk_z
    n_chunks = (z_total + chunk_z - 1) // chunk_z
    print(f"Producing {n_chunks} chunks of up to {chunk_z} slices each")

    for ci in range(n_chunks):
        z0 = ci * chunk_z
        z1 = min(z0 + chunk_z, z_total)
        out_name = f"volume_{z0:04d}-{z1 - 1:04d}.h5"
        out_path = os.path.join(args.out_dir, out_name)
        if os.path.exists(out_path):
            print(f"  [{ci+1}/{n_chunks}] {out_name} exists, skipping")
            continue

        vol = np.empty((z1 - z0, H, W), dtype=args.dtype)
        for k, fname in enumerate(files[z0:z1]):
            img = tifffile.imread(os.path.join(args.tiff_dir, fname))
            if img.shape != (H, W):
                sys.exit(f"TIFF {fname} has shape {img.shape} != {(H, W)}")
            vol[k] = img.astype(args.dtype, copy=False)
        with h5py.File(out_path, "w") as h:
            h.create_dataset(
                "volume", data=vol, compression="lzf",
                chunks=(min(8, vol.shape[0]), 512, 512),
            )
        print(f"  [{ci+1}/{n_chunks}] wrote {out_name}  shape={vol.shape}")

    print(f"Done. {n_chunks} chunks in {args.out_dir}")


if __name__ == "__main__":
    main()
