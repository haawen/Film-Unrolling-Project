"""Convert the restitched tomogram (Mickey_norm{z}.tif) to the HDF5 slices our
loaders read (`<out>/Mickey_norm_{z:04d}.h5`, key `image`, gzip-1), IN PLACE:
each TIF is deleted only after its HDF5 has been written and read back
bit-identical, so peak disk use stays ~ the TIF set (Merlin7 /data is tight).
Idempotent: slices whose HDF5 already verifies are skipped.

The `_` before the digits matters: discover_ct_slices matches r"_(\\d+)\\.h5$".

  python Scripts/restitch_tif_to_h5.py <tif_dir> <h5_dir> [<in_regex> <out_prefix>]

Defaults (restitch): in_regex "Mickey_norm(\\d+)\\.tif", out_prefix "Mickey_norm".
High-res scan (2026-11): "Mickey_HR_bin4_(\\d+)\\.tif" "Mickey_HR".
"""
import glob
import os
import re
import sys

import h5py
import numpy as np
import tifffile


def main():
    src, dst = sys.argv[1], sys.argv[2]
    in_re = sys.argv[3] if len(sys.argv) > 3 else r"Mickey_norm(\d+)\.tif"
    out_prefix = sys.argv[4] if len(sys.argv) > 4 else "Mickey_norm"
    os.makedirs(dst, exist_ok=True)
    files = []
    for p in glob.glob(os.path.join(src, "*.tif")):
        m = re.fullmatch(in_re, os.path.basename(p))
        if m:
            files.append((int(m.group(1)), p))
    files.sort()
    print(f"{len(files)} TIFs in {src}", flush=True)
    shape = None
    for i, (z, p) in enumerate(files):
        out = os.path.join(dst, f"{out_prefix}_{z:04d}.h5")
        a = tifffile.imread(p)
        if shape is None:
            shape = a.shape
            print(f"slice shape {a.shape} dtype {a.dtype}", flush=True)
        if a.shape != shape:
            raise SystemExit(f"z{z}: shape {a.shape} != {shape}")
        with h5py.File(out, "w") as f:
            d = f.create_dataset("image", data=a, compression="gzip",
                                 compression_opts=1, chunks=(256, 256))
            d.attrs["source"] = p
            d.attrs["z_restitch"] = z
        with h5py.File(out, "r") as f:
            ok = np.array_equal(f["image"][...], a)
        if not ok:
            raise SystemExit(f"z{z}: read-back mismatch, TIF kept")
        os.remove(p)
        if i % 200 == 0 or i == len(files) - 1:
            print(f"  [{i + 1}/{len(files)}] z{z}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
