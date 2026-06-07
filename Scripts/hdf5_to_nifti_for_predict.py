"""HDF5 volume chunks -> NIfTI files for nnU-Net v2 inference.

For each `volume_NNNN-MMMM.h5` (key 'volume') in --in-dir, writes
    <prefix>_NNNN_0000.nii.gz
to --out-dir. No labels needed (this is inference, not training).

Pair with `nnUNetv2_predict --save_probabilities` and then
`nnunet_softmax_to_h5_probs.py` to package the softmax outputs.
"""
import argparse
import os
import re
import h5py
import nibabel as nib
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", default="Sample3D",
                   help="Case-id prefix; nnU-Net will produce one output per "
                        "input named <prefix>_NNNN.nii.gz.")
    args = p.parse_args()

    files = sorted(
        f for f in os.listdir(args.in_dir)
        if f.startswith("volume_") and f.endswith(".h5")
        and "Probabilities" not in f
    )
    if not files:
        raise SystemExit(f"No volume_*.h5 in {args.in_dir}")
    os.makedirs(args.out_dir, exist_ok=True)
    affine = np.eye(4)
    for i, vf in enumerate(files):
        m = re.search(r"volume_(\d+)-(\d+)", vf)
        if not m:
            print(f"  skip {vf}: cannot parse range"); continue
        case_id = f"{args.prefix}_{m.group(1)}"
        out = os.path.join(args.out_dir, f"{case_id}_0000.nii.gz")
        if os.path.exists(out):
            print(f"  [{i+1}/{len(files)}] {case_id} exists, skip"); continue
        with h5py.File(os.path.join(args.in_dir, vf), "r") as h:
            vol = h["volume"][...].astype(np.float32)
        nib.save(nib.Nifti1Image(vol, affine), out)
        print(f"  [{i+1}/{len(files)}] {case_id}  shape={vol.shape}")
    print(f"Done. Wrote {len(files)} NIfTIs to {args.out_dir}")


if __name__ == "__main__":
    main()
