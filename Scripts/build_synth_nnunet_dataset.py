"""Build a chunked nnU-Net v2 3D dataset (or predict-input NIfTIs) from a
synthetic preset's HDF5 volume + GT one-hot Probabilities.

Chunks the (Z, H, W) volume into CHUNK-slice sub-volumes so each case matches
the real-Mickey z-regime (~20 slices) that nnU-Net was configured for, and so
each prediction softmax stays small. The unwrapping pipeline reads the chunked
volume_NNNN-MMMM.h5 + _Probabilities.h5 pairs natively.

Modes:
  --mode train    : write imagesTr/<case>_0000.nii.gz + labelsTr/<case>.nii.gz
                    (label = argmax of GT Probabilities) + dataset.json
  --mode predict  : write only <prefix>_<start4>_0000.nii.gz images (no labels),
                    for nnUNetv2_predict --save_probabilities.

Case/file naming uses the chunk start index zero-padded to 4 (matches the
volume_NNNN-MMMM convention and the softmax->h5 packer's regex).
"""
import argparse
import json
import os
import glob
import re

import h5py
import numpy as np
import nibabel as nib


def load_volume_and_seg(syn_dir):
    """Load full (Z,H,W) uint16 volume and (Z,H,W) uint8 seg (argmax of probs)."""
    vol_files = sorted(
        f for f in glob.glob(os.path.join(syn_dir, "volume_*[0-9].h5"))
        if "Probabilities" not in f
    )
    if not vol_files:
        raise SystemExit(f"No volume_*.h5 in {syn_dir}")
    vols, segs = [], []
    for vf in vol_files:
        pf = vf.replace(".h5", "_Probabilities.h5")
        with h5py.File(vf, "r") as h:
            vols.append(h["volume"][...])
        with h5py.File(pf, "r") as h:
            probs = h["exported_data"][...]            # (Z,H,W,C)
            segs.append(np.argmax(probs, axis=-1).astype(np.uint8))
    vol = np.concatenate(vols, axis=0)
    seg = np.concatenate(segs, axis=0)
    return vol, seg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--syn-dir", required=True)
    p.add_argument("--mode", choices=["train", "predict"], required=True)
    p.add_argument("--chunk", type=int, default=24)
    p.add_argument("--prefix", default="Synth3D")
    # train mode
    p.add_argument("--raw-dir", default=None, help="nnUNet_raw/<DatasetXXX> dir")
    # predict mode
    p.add_argument("--out-dir", default=None, help="dir for predict-input NIfTIs")
    args = p.parse_args()

    print(f"Loading volume+seg from {args.syn_dir} ...")
    vol, seg = load_volume_and_seg(args.syn_dir)
    Z, H, W = vol.shape
    print(f"  volume {vol.shape} dtype={vol.dtype}  seg classes={np.unique(seg).tolist()}")
    affine = np.eye(4)
    starts = list(range(0, Z, args.chunk))

    if args.mode == "train":
        assert args.raw_dir, "--raw-dir required for train mode"
        img_dir = os.path.join(args.raw_dir, "imagesTr")
        lbl_dir = os.path.join(args.raw_dir, "labelsTr")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        n = 0
        for s in starts:
            e = min(s + args.chunk, Z)
            case = f"{args.prefix}_{s:04d}"
            img = vol[s:e].astype(np.float32)
            lbl = seg[s:e].astype(np.uint8)
            nib.save(nib.Nifti1Image(img, affine),
                     os.path.join(img_dir, f"{case}_0000.nii.gz"))
            nib.save(nib.Nifti1Image(lbl, affine),
                     os.path.join(lbl_dir, f"{case}.nii.gz"))
            n += 1
            print(f"  [{n}/{len(starts)}] {case}  img{img.shape} lbl-classes={np.unique(lbl).tolist()}")
        ds = {
            "channel_names": {"0": "XRay"},
            "labels": {"background": 0, "film_base": 1, "emulsion": 2},
            "numTraining": n,
            "file_ending": ".nii.gz",
        }
        with open(os.path.join(args.raw_dir, "dataset.json"), "w") as f:
            json.dump(ds, f, indent=2)
        print(f"Done. {n} training cases + dataset.json -> {args.raw_dir}")

    else:  # predict
        assert args.out_dir, "--out-dir required for predict mode"
        os.makedirs(args.out_dir, exist_ok=True)
        n = 0
        for s in starts:
            e = min(s + args.chunk, Z)
            case = f"{args.prefix}_{s:04d}"
            img = vol[s:e].astype(np.float32)
            nib.save(nib.Nifti1Image(img, affine),
                     os.path.join(args.out_dir, f"{case}_0000.nii.gz"))
            n += 1
            print(f"  [{n}/{len(starts)}] {case}  img{img.shape}")
        print(f"Done. {n} predict-input NIfTIs -> {args.out_dir}")


if __name__ == "__main__":
    main()
