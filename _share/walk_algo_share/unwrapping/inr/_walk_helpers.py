"""Small standalone helpers `walk_emulsion.py` needs, extracted verbatim
(unmodified logic) from three unrelated larger modules in the main codebase
so this package doesn't have to ship those modules just for one function
each:

  - detect_seam_angle  <- geodesic_unwrap.py's `_detect_seam_angle`
                           (that file is an abandoned earlier unwrapping
                           approach; this one function is the only thing
                           walk_emulsion.py actually uses from it)
  - bilinear_sample     <- sanity_winding_strip.py's `bilinear_sample`
                           (that file belongs to a different, separate
                           INR-based unwrapping method)
  - load_volume_chunk   <- unwrap_data_3d.py's `load_volume_chunk`, with
                           its unrelated torch-dependent code dropped (the
                           original module imports torch at the top for
                           other, unused classes; this version only needs
                           numpy + h5py)
"""

import numpy as np


def detect_seam_angle(film_mask, center_yx, r_min, r_max):
    """Detect the spiral opening angle by radial ray scanning.

    At most angles, a radial ray crosses all windings (film-air-film-...).
    At the spiral opening angle, one air gap is missing (two windings merge),
    so the ray crosses fewer air gaps. Returns the angle with the fewest
    film->air transitions.
    """
    cy, cx = center_yx
    H, W = film_mask.shape
    n_angles = 720
    angles = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    gap_counts = np.zeros(n_angles, dtype=np.int32)

    r_vals = np.arange(int(r_min), int(r_max) + 1)
    for i, a in enumerate(angles):
        ray_y = np.round(cy + r_vals * np.sin(a)).astype(int)
        ray_x = np.round(cx + r_vals * np.cos(a)).astype(int)
        valid = (ray_y >= 0) & (ray_y < H) & (ray_x >= 0) & (ray_x < W)
        vals = film_mask[ray_y[valid], ray_x[valid]].astype(np.int8)
        gap_counts[i] = int(np.sum(np.diff(vals) == -1))

    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(gap_counts.astype(np.float64), size=20, mode="wrap")
    return angles[np.argmin(smooth)]


def bilinear_sample(img, xs, ys):
    """Bilinear-sample img (H, W) at float coords xs, ys (1-D arrays).
    Out-of-bounds samples return 0.0."""
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


def load_volume_chunk(volume_path, probs_path, max_slices=None):
    """Load one 3D volume chunk and its segmentation.

    volume_path: HDF5 with dataset "volume" (raw CT, any z,y,x shape).
    probs_path:  HDF5 with dataset "exported_data" (per-voxel class softmax,
                 shape z,y,x,n_classes) -- argmax'd here into integer labels
                 (0=background, 1=film base, 2=emulsion).
    If `max_slices` is given, only the first `max_slices` z-slices are read.
    """
    import h5py

    with h5py.File(volume_path, "r") as f:
        ds = f["volume"]
        n_read = ds.shape[0] if max_slices is None else min(max_slices, ds.shape[0])
        volume = ds[:n_read].astype(np.float32)
    vmin, vmax = volume.min(), volume.max()
    volume = (volume - vmin) / (vmax - vmin + 1e-8)

    with h5py.File(probs_path, "r") as f:
        ds = f["exported_data"]
        n_read = ds.shape[0] if max_slices is None else min(max_slices, ds.shape[0])
        probs = ds[:n_read].astype(np.float32)
    seg = np.argmax(probs, axis=-1).astype(np.int64)

    return volume, seg
