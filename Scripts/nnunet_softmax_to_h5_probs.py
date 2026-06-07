"""Convert nnU-Net v2 saved softmax (.npz) back to the HDF5 Probabilities
format the unwrapping pipeline expects.

For every  <prefix>_NNNN.nii.gz (label) +  <prefix>_NNNN.npz (softmax)
pair in --nnunet-out, write
    volume_NNNN-MMMM_Probabilities.h5
to --pairs-dir (next to the existing volume_NNNN-MMMM.h5), with key
'exported_data' of shape (Z, H, W, 3) float32 — same convention as the
existing 01_Mickey_3d Probabilities files.

The .npz keys produced by nnU-Net v2 are usually 'probabilities' (shape
(C, Z, H, W) float32). We transpose to (Z, H, W, C).
"""
import argparse
import os
import re
import h5py
import nibabel as nib
import numpy as np


def find_pair_volume(pairs_dir, start_idx):
    """Find the volume_NNNN-MMMM.h5 whose NNNN matches start_idx."""
    needle = f"volume_{start_idx:04d}-"
    for f in os.listdir(pairs_dir):
        if f.startswith(needle) and f.endswith(".h5") and "Probabilities" not in f:
            return f
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nnunet-out", required=True,
                   help="Dir with nnU-Net predictions (NIfTI + .npz).")
    p.add_argument("--pairs-dir", required=True,
                   help="Dir containing volume_*.h5 to pair with.")
    p.add_argument("--prefix", default="Sample3D")
    args = p.parse_args()

    npz_files = sorted(
        f for f in os.listdir(args.nnunet_out) if f.endswith(".npz")
    )
    if not npz_files:
        raise SystemExit(f"No .npz softmax in {args.nnunet_out}")

    for i, nf in enumerate(npz_files):
        m = re.match(rf"{args.prefix}_(\d+)\.npz", nf)
        if not m:
            print(f"  skip {nf}: cannot parse"); continue
        start_idx = int(m.group(1))
        vol_name = find_pair_volume(args.pairs_dir, start_idx)
        if vol_name is None:
            print(f"  skip {nf}: no matching volume_{start_idx:04d}-*.h5"); continue
        out_name = vol_name.replace(".h5", "_Probabilities.h5")
        out_path = os.path.join(args.pairs_dir, out_name)
        if os.path.exists(out_path):
            # Treat tiny stubs (<10 MB) as broken from a failed previous run.
            size = os.path.getsize(out_path)
            if size > 10 * 1024 * 1024:
                print(f"  [{i+1}/{len(npz_files)}] {out_name} exists ({size/1e6:.0f} MB), skip"); continue
            print(f"  [{i+1}/{len(npz_files)}] {out_name} exists but only {size} bytes — treating as broken, overwriting")

        arr = np.load(os.path.join(args.nnunet_out, nf))
        if "probabilities" in arr.files:
            probs = arr["probabilities"]
        elif "softmax" in arr.files:
            probs = arr["softmax"]
        else:
            probs = arr[arr.files[0]]
        # Normalise to (Z, H, W, C). nnU-Net v2 may save as (C, Z, H, W),
        # (H, W, Z, C), (X, Y, Z, C), etc. depending on version.
        if probs.ndim != 4:
            raise ValueError(f"{nf}: expected 4-D softmax, got {probs.shape}")
        # 1) Find the class dim (a single dim with value 2-4) and move it last.
        class_dims = [i for i, s in enumerate(probs.shape) if 2 <= s <= 4]
        if len(class_dims) != 1:
            raise ValueError(
                f"{nf}: cannot identify class axis (shape={probs.shape})"
            )
        probs = np.moveaxis(probs, class_dims[0], -1)
        # 2) Of the remaining three spatial dims, Z is the smallest (chunk depth ~20
        #    vs H/W ~3064). Move it first.
        spatial = probs.shape[:-1]
        z_axis = int(np.argmin(spatial))
        probs = np.moveaxis(probs, z_axis, 0)
        # Now probs is (Z, H, W, C)
        probs = np.ascontiguousarray(probs.astype(np.float32, copy=False))

        # Clamp chunk dims to data dims (Z is small, H/W are large).
        Z, H, W, C = probs.shape
        chunks = (min(8, Z), min(512, H), min(512, W), C)
        with h5py.File(out_path, "w") as h:
            h.create_dataset(
                "exported_data", data=probs, compression="lzf",
                chunks=chunks,
            )
        print(f"  [{i+1}/{len(npz_files)}] wrote {out_name}  shape={probs.shape}")

    print(f"Done.")


if __name__ == "__main__":
    main()
