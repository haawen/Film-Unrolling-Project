"""
Detect the center of the film spool from a segmentation mask.

The spool center is the centroid of the large air hole in the middle of the roll.
We distinguish it from the air gaps between film layers by finding the largest
connected component of air that is near the image center.
"""

import numpy as np
from scipy import ndimage


def find_spool_center(seg: np.ndarray) -> tuple[float, float]:
    """Find the spool center from a 2D segmentation mask.

    Args:
        seg: 2D uint8 array with classes 0=air, 1=film, 2=silver/emulsion.

    Returns:
        (cy, cx) center coordinates in pixel space.
    """
    air_mask = seg == 0
    labeled, n_components = ndimage.label(air_mask)

    if n_components == 0:
        # Fallback: centroid of film pixels
        return _film_centroid(seg)

    # Find the largest air component that overlaps with the center region
    img_cy, img_cx = seg.shape[0] / 2, seg.shape[1] / 2
    center_radius = min(seg.shape) * 0.3  # look within 30% of image size

    best_label = None
    best_size = 0

    component_sizes = ndimage.sum(air_mask, labeled, range(1, n_components + 1))

    for i in range(1, n_components + 1):
        comp_mask = labeled == i
        ys, xs = np.where(comp_mask)
        centroid_y, centroid_x = ys.mean(), xs.mean()
        dist_to_center = np.sqrt((centroid_y - img_cy) ** 2 + (centroid_x - img_cx) ** 2)

        # Must be near image center and reasonably large
        if dist_to_center < center_radius and component_sizes[i - 1] > best_size:
            best_size = component_sizes[i - 1]
            best_label = i

    if best_label is not None:
        hole_mask = labeled == best_label
        ys, xs = np.where(hole_mask)
        return float(ys.mean()), float(xs.mean())

    # Fallback: use centroid of all film pixels
    return _film_centroid(seg)


def _film_centroid(seg: np.ndarray) -> tuple[float, float]:
    """Centroid of all film+silver pixels as fallback."""
    film_mask = seg >= 1
    ys, xs = np.where(film_mask)
    if len(ys) == 0:
        return seg.shape[0] / 2, seg.shape[1] / 2
    return float(ys.mean()), float(xs.mean())


def find_spool_center_stack(seg_stack: np.ndarray) -> np.ndarray:
    """Find spool center for each z-slice in a 3D segmentation volume.

    Args:
        seg_stack: 3D array (Z, H, W) with segmentation labels.

    Returns:
        Array of shape (Z, 2) with (cy, cx) per slice.
    """
    centers = np.zeros((seg_stack.shape[0], 2), dtype=np.float64)
    for z in range(seg_stack.shape[0]):
        centers[z] = find_spool_center(seg_stack[z])
    return centers
