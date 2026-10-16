"""Render an 'oracle' strip by sampling the CT volume at GT u_map positions.

The oracle is the strip you'd get if you knew the true (u, z) -> (x, y) mapping
exactly: each strip pixel = mean of CT intensities at every emulsion pixel
falling into that strip column. Has only CT noise + binning/aliasing error;
no geometric or learned-model error.

Used as a per-paper "irreducible reconstruction" reference.
Output: oracle_strip.npz with {strip, n_layers, pixels_per_winding}.
"""

import argparse
import os
import sys
import numpy as np
import h5py


def load_volume(syn_dir):
    """Load full CT volume from the per-slice HDF5 files in syn_dir."""
    files = sorted(
        f for f in os.listdir(syn_dir)
        if f.startswith("slice_") and f.endswith(".h5") and "Probabilities" not in f
    )
    if not files:
        raise FileNotFoundError(f"No slice_*.h5 in {syn_dir}")
    # Peek shape
    with h5py.File(os.path.join(syn_dir, files[0]), "r") as h:
        key = next(iter(h.keys()))
        first = h[key][...]
    # First file may be a chunk (Z, H, W) or single slice (H, W)
    if first.ndim == 3:
        # Chunked: one or more files, concatenate along axis 0
        chunks = [first]
        for f in files[1:]:
            with h5py.File(os.path.join(syn_dir, f), "r") as h:
                k = next(iter(h.keys()))
                chunks.append(h[k][...])
        vol = np.concatenate(chunks, axis=0)
    else:
        # One slice per file
        Z = len(files)
        H, W = first.shape
        vol = np.empty((Z, H, W), dtype=first.dtype)
        vol[0] = first
        for i, f in enumerate(files[1:], 1):
            with h5py.File(os.path.join(syn_dir, f), "r") as h:
                k = next(iter(h.keys()))
                vol[i] = h[k][...]
    return vol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--syn-dir", required=True)
    p.add_argument("--out", required=True, help="Path to write oracle_strip.npz")
    args = p.parse_args()

    print(f"Loading GT npz from {args.syn_dir}/ground_truth.npz ...")
    gt = np.load(os.path.join(args.syn_dir, "ground_truth.npz"))
    u_map = gt["u_map"]               # (H, W), in [0, 1) over strip
    seg = gt["seg"]                   # (H, W) uint8
    strip_gt = gt["strip"]            # (Z_strip, W_strip)
    n_windings = int(gt["n_windings"])
    Z_strip, W_strip = strip_gt.shape
    print(f"  u_map shape={u_map.shape}, strip GT shape={strip_gt.shape}, "
          f"n_windings={n_windings}")

    print(f"Loading CT volume from {args.syn_dir} ...")
    vol = load_volume(args.syn_dir)
    print(f"  volume shape={vol.shape} dtype={vol.dtype}")
    assert vol.shape[1:] == u_map.shape, \
        f"volume HW {vol.shape[1:]} != u_map {u_map.shape}"
    Z = vol.shape[0]
    if Z != Z_strip:
        print(f"  warn: volume Z={Z} != strip Z_strip={Z_strip}; using volume Z")
        Z_strip = Z

    # Emulsion pixels with valid u_map
    print("Selecting emulsion pixels with valid u_map ...")
    emul = (seg == 2) & np.isfinite(u_map)
    ys, xs = np.where(emul)
    u_vals = u_map[ys, xs]
    # Wrap to [0, 1)
    u_vals = np.mod(u_vals, 1.0)
    strip_cols = np.clip((u_vals * W_strip).astype(np.int64), 0, W_strip - 1)
    n_emul = ys.size
    print(f"  emulsion pixels: {n_emul:,}")
    counts = np.bincount(strip_cols, minlength=W_strip).astype(np.float64)
    print(f"  empty strip cols: {int((counts == 0).sum())} / {W_strip}")
    cov = (counts > 0).mean() * 100.0
    print(f"  column coverage: {cov:.2f}%")

    # Per-z bincount-accumulate
    print(f"Rendering oracle strip ({Z_strip} × {W_strip}) ...")
    oracle = np.zeros((Z_strip, W_strip), dtype=np.float32)
    safe_counts = np.maximum(counts, 1.0)
    for z in range(Z_strip):
        vals = vol[z, ys, xs].astype(np.float64)
        sums = np.bincount(strip_cols, weights=vals, minlength=W_strip)
        oracle[z] = (sums / safe_counts).astype(np.float32)
        if (z + 1) % 32 == 0 or z == Z_strip - 1:
            print(f"  z={z+1}/{Z_strip}")

    # Fill empty cols with neighbor average (cheap, optional)
    empty = counts == 0
    if empty.any():
        print("Filling empty cols with neighbor interpolation ...")
        # Per-row 1D interpolation in u
        idx = np.arange(W_strip)
        valid = ~empty
        for z in range(Z_strip):
            row = oracle[z]
            row[empty] = np.interp(idx[empty], idx[valid], row[valid])

    pixels_per_winding = int(round(W_strip / n_windings))
    print(f"Saving to {args.out} ...")
    np.savez_compressed(
        args.out,
        strip=oracle,
        n_layers=np.int64(n_windings),
        pixels_per_winding=np.int64(pixels_per_winding),
    )
    print(f"Done. strip range [{oracle.min():.3f}, {oracle.max():.3f}], "
          f"mean={oracle.mean():.3f}")


if __name__ == "__main__":
    main()
