"""CT / segmentation I/O and image sampling shared by the emulsion-walk pipeline.

Function bodies are verbatim copies of the originals in the thesis working
repository (unwrap_data_3d.py, unroll_continuous.py, sanity_winding_strip.py),
collected here so the final pipeline does not depend on legacy modules.

Data conventions
----------------
* CT slices:   one HDF5 per z-slice, file name ending in ``_{z:04d}.h5``, key
  ``image`` (H, W) uint16  -> ``discover_ct_slices`` / ``load_ct``.
* Segmentation chunks: ``volume_{z0:04d}-{z1:04d}_Probabilities.h5``, key
  ``exported_data`` (Z, H, W, 3); classes 0 = air, 1 = film base, 2 = emulsion.
  The class map is ``argmax`` over the last axis, so either float softmax
  probabilities or a compact uint8 one-hot encoding work unchanged.
"""
import glob
import os
import re

import h5py
import numpy as np


def discover_volumes(data_dir):
    """Find all volume + probability file pairs, sorted by z-range.

    Deliverable addition: segmentation-only directories are accepted too. A
    ``volume_*_Probabilities.h5`` without its CT chunk is returned with
    ``volume_path=None``; the emulsion walk reads only the segmentation (the CT
    for rendering comes from the single-slice files via ``discover_ct_slices``).
    """
    vol_files = sorted(glob.glob(os.path.join(data_dir, "volume_*[0-9].h5")))
    pairs = []
    for vf in vol_files:
        if "Probabilities" in vf:
            continue
        m = re.search(r"volume_(\d+)-(\d+)\.h5$", vf)
        if not m:
            continue
        z_start, z_end = int(m.group(1)), int(m.group(2))
        probs_path = vf.replace(".h5", "_Probabilities.h5")
        if os.path.exists(probs_path):
            pairs.append((vf, probs_path, z_start, z_end))
        else:
            print(f"  WARNING: no probabilities for {vf}, skipping")
    have = {p[1] for p in pairs}
    for pf in sorted(glob.glob(os.path.join(data_dir, "volume_*_Probabilities.h5"))):
        m = re.search(r"volume_(\d+)-(\d+)_Probabilities\.h5$", pf)
        if m and pf not in have:
            pairs.append((None, pf, int(m.group(1)), int(m.group(2))))
    pairs.sort(key=lambda x: x[2])
    return pairs


def discover_ct_slices(ct_dir):
    """Map raw-CT z-index -> file path from 01_Mickey_Stitch..._{z:04d}.h5."""
    z_to_path = {}
    for p in glob.glob(os.path.join(ct_dir, "*.h5")):
        m = re.search(r"_(\d+)\.h5$", os.path.basename(p))
        if m:
            z_to_path[int(m.group(1))] = p
    return z_to_path


def load_ct(path):
    with h5py.File(path, "r") as f:
        return np.asarray(f["image"], dtype=np.float32)


def load_seg_slices(probs_path, frac=1.0):
    """Read `frac` of a chunk's z-slices' segmentation, evenly spaced in z.

    frac=1.0 -> all 20 slices of the chunk; 0.5 -> ~10, etc. The 2.2 GB prob map
    is opened once and only the selected slices are decompressed. Returns a list of
    (seg_2d, slice_index) so the caller can walk many z-slices per chunk (dense
    anchors) instead of just the mid one. Used for the 50/75/100%-of-slices renders.
    """
    with h5py.File(probs_path, "r") as f:
        ds = f["exported_data"]
        Z = ds.shape[0]
        n = max(1, int(round(frac * Z)))
        idxs = np.unique(np.linspace(0, Z - 1, n).round().astype(int))
        # decompress the whole (z-chunked) prob map ONCE — per-slice ds[i] reads
        # re-decompress the entire 2.2 GB map each time (~20x slower per chunk).
        arr = ds[:]                                       # (Z, H, W, 3)
    return [(np.argmax(arr[int(i)], axis=-1).astype(np.int64), int(i)) for i in idxs]


def norm_slice(img):
    """Robust per-slice normalization to [0,1] (1-99 pct) -> kills z-banding."""
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def bilinear_sample(img, xs, ys):
    """Bilinear-sample img (H, W) at float coords xs, ys (1-D arrays).

    Out-of-bounds samples return 0.0.
    """
    H, W = img.shape
    x0 = np.floor(xs).astype(np.int64)
    y0 = np.floor(ys).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    inb = (xs >= 0) & (xs <= W - 1) & (ys >= 0) & (ys <= H - 1)
    x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y1, 0, H - 1)
    wx = xs - x0
    wy = ys - y0
    Ia = img[y0c, x0c]; Ib = img[y0c, x1c]
    Ic = img[y1c, x0c]; Id = img[y1c, x1c]
    val = (Ia * (1 - wx) * (1 - wy) + Ib * wx * (1 - wy)
           + Ic * (1 - wx) * wy + Id * wx * wy)
    return np.where(inb, val, 0.0)
