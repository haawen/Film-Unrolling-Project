"""
Data preparation for learned unwrapping (Step 1b).

Extracts film pixels from segmentation, computes initial (u, v) parameterization
using polar coordinates and layer detection (reusing v3 logic), and provides
GPU-resident sampling for training.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch


def compute_polar(seg, center):
    """Convert film pixel positions to polar coordinates relative to spiral center.

    Args:
        seg: (H, W) segmentation mask, 0=air, 1=film, 2=emulsion
        center: (cy, cx) spiral center in pixel coords

    Returns:
        film_yx: (N, 2) pixel positions of film pixels (row, col)
        polar_r: (N,) radius for each film pixel
        polar_theta: (N,) angle in [0, 2π) for each film pixel
    """
    cy, cx = center
    film_mask = seg > 0  # film base (1) or emulsion (2)
    ys, xs = np.where(film_mask)

    dy = ys.astype(np.float64) - cy
    dx = xs.astype(np.float64) - cx

    polar_r = np.sqrt(dy ** 2 + dx ** 2)
    polar_theta = np.arctan2(dy, dx) % (2 * np.pi)  # [0, 2π)

    film_yx = np.stack([ys, xs], axis=-1)
    return film_yx, polar_r, polar_theta


def assign_layers_polar(seg, center, n_angles=1440, min_thickness=3, sigma=15):
    """Detect and number film layers using radial scanning in polar space.

    At each discrete angle, scans radially outward, identifies contiguous
    film segments separated by air gaps, and numbers them inside-out.
    Gaussian smoothing across angles ensures continuity.

    Returns:
        n_layers: number of detected layers
        layer_radii: (n_layers, n_angles) smoothed center radius per layer per angle
        layer_r_start: (n_layers, n_angles) inner edge radius
        layer_r_end: (n_layers, n_angles) outer edge radius
        angles: (n_angles,) angle values
    """
    cy, cx = center
    H, W = seg.shape
    max_radius = int(np.sqrt(max(cy, H - cy) ** 2 + max(cx, W - cx) ** 2))
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

    # Scan radially at each angle
    layers_per_angle = []
    for i, theta in enumerate(angles):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        layers = []
        in_layer = False
        start = 0

        for r in range(max_radius):
            y = int(round(cy + r * sin_t))
            x = int(round(cx + r * cos_t))
            is_film = (0 <= y < H and 0 <= x < W and seg[y, x] > 0)

            if is_film and not in_layer:
                in_layer = True
                start = r
            elif not is_film and in_layer:
                in_layer = False
                if r - start >= min_thickness:
                    layers.append({
                        "r_start": float(start),
                        "r_end": float(r - 1),
                        "r_center": (start + r - 1) / 2.0,
                    })

        if in_layer and max_radius - start >= min_thickness:
            layers.append({
                "r_start": float(start),
                "r_end": float(max_radius - 1),
                "r_center": (start + max_radius - 1) / 2.0,
            })

        layers_per_angle.append(layers)

    # Determine number of layers (modal count)
    counts = [len(l) for l in layers_per_angle if len(l) > 0]
    if not counts:
        raise ValueError("No film layers detected")
    hist, edges = np.histogram(counts, bins=range(min(counts), max(counts) + 2))
    n_layers = int(edges[np.argmax(hist)])

    # Build radius matrices and smooth across angles
    radii = np.full((n_layers, n_angles), np.nan)
    r_starts = np.full((n_layers, n_angles), np.nan)
    r_ends = np.full((n_layers, n_angles), np.nan)

    for i in range(n_angles):
        sorted_layers = sorted(layers_per_angle[i], key=lambda l: l["r_center"])
        n = min(len(sorted_layers), n_layers)
        for k in range(n):
            radii[k, i] = sorted_layers[k]["r_center"]
            r_starts[k, i] = sorted_layers[k]["r_start"]
            r_ends[k, i] = sorted_layers[k]["r_end"]

    # Gaussian smooth across angles for continuity
    smoothed_radii = np.full_like(radii, np.nan)
    smoothed_starts = np.full_like(r_starts, np.nan)
    smoothed_ends = np.full_like(r_ends, np.nan)

    for k in range(n_layers):
        for arr, out in [(radii, smoothed_radii), (r_starts, smoothed_starts),
                         (r_ends, smoothed_ends)]:
            r = arr[k].copy()
            valid = ~np.isnan(r)
            if valid.sum() < sigma * 3:
                out[k] = r
                continue
            nans = np.isnan(r)
            if nans.any() and not nans.all():
                valid_idx = np.flatnonzero(~nans)
                r[nans] = np.interp(np.flatnonzero(nans), valid_idx, r[valid_idx])
            out[k] = gaussian_filter1d(r, sigma=sigma, mode="wrap")

    return n_layers, smoothed_radii, smoothed_starts, smoothed_ends, angles


def compute_initial_uv(film_yx, polar_r, polar_theta, seg, center,
                       n_layers, layer_radii, layer_starts, layer_ends, angles):
    """Compute initial (u, v) parameterization for each film pixel.

    u = normalized arc-length position along the strip (layer_index + theta/2π)
    v = normalized radial position within the layer [0, 1]

    Args:
        film_yx: (N, 2) pixel positions
        polar_r, polar_theta: (N,) polar coordinates
        seg: (H, W) segmentation
        center: (cy, cx)
        n_layers: int
        layer_radii: (n_layers, n_angles) smoothed center radii
        layer_starts: (n_layers, n_angles) inner edge radii
        layer_ends: (n_layers, n_angles) outer edge radii
        angles: (n_angles,) angle values

    Returns:
        u: (N,) position along strip, range ~ [0, n_layers]
        v: (N,) position across strip, range ~ [0, 1]
        valid: (N,) bool mask — True if pixel was successfully assigned to a layer
    """
    N = len(film_yx)
    u = np.zeros(N, dtype=np.float64)
    v = np.zeros(N, dtype=np.float64)
    valid = np.zeros(N, dtype=bool)

    n_angles = len(angles)
    d_angle = angles[1] - angles[0]

    for i in range(N):
        r = polar_r[i]
        theta = polar_theta[i]

        # Find nearest angle bin
        angle_idx = int(theta / d_angle) % n_angles

        # Find which layer this pixel belongs to (nearest center radius)
        best_layer = -1
        best_dist = np.inf
        for k in range(n_layers):
            r_center = layer_radii[k, angle_idx]
            r_start = layer_starts[k, angle_idx]
            r_end = layer_ends[k, angle_idx]
            if np.isnan(r_center):
                continue
            # Check if within the layer bounds (with some tolerance)
            tolerance = max(3.0, (r_end - r_start) * 0.3)
            if r_start - tolerance <= r <= r_end + tolerance:
                dist = abs(r - r_center)
                if dist < best_dist:
                    best_dist = dist
                    best_layer = k

        if best_layer < 0:
            continue

        # u: position along the strip
        u[i] = best_layer + theta / (2 * np.pi)

        # v: normalized position within layer [0, 1]
        r_s = layer_starts[best_layer, angle_idx]
        r_e = layer_ends[best_layer, angle_idx]
        if not np.isnan(r_s) and not np.isnan(r_e) and r_e > r_s:
            v[i] = np.clip((r - r_s) / (r_e - r_s), 0.0, 1.0)
        else:
            v[i] = 0.5

        valid[i] = True

    return u, v, valid


def compute_initial_uv_fast(film_yx, polar_r, polar_theta,
                            n_layers, layer_radii, layer_starts, layer_ends,
                            angles):
    """Vectorized version of compute_initial_uv — much faster for millions of pixels.

    Returns:
        u: (N,) position along strip
        v: (N,) position across strip
        valid: (N,) bool mask
    """
    N = len(film_yx)
    n_angles = len(angles)
    d_angle = angles[1] - angles[0]

    # Find nearest angle bin for each pixel
    angle_idx = (polar_theta / d_angle).astype(np.int64) % n_angles

    # For each pixel, find nearest layer
    u = np.zeros(N, dtype=np.float64)
    v = np.zeros(N, dtype=np.float64)
    valid = np.zeros(N, dtype=bool)

    # Gather layer info at each pixel's angle
    # layer_radii is (n_layers, n_angles), index by angle_idx → (n_layers, N)
    centers_at_pixel = layer_radii[:, angle_idx]    # (n_layers, N)
    starts_at_pixel = layer_starts[:, angle_idx]    # (n_layers, N)
    ends_at_pixel = layer_ends[:, angle_idx]        # (n_layers, N)

    # Distance from each pixel's r to each layer's center
    r_expanded = polar_r[np.newaxis, :]  # (1, N)
    dist = np.abs(centers_at_pixel - r_expanded)  # (n_layers, N)
    dist[np.isnan(centers_at_pixel)] = np.inf

    # Check within-layer bounds
    tolerance = np.maximum(3.0, (ends_at_pixel - starts_at_pixel) * 0.3)
    in_bounds = (r_expanded >= starts_at_pixel - tolerance) & \
                (r_expanded <= ends_at_pixel + tolerance)
    dist[~in_bounds] = np.inf

    # Best layer per pixel
    best_layer = np.argmin(dist, axis=0)  # (N,)
    best_dist = dist[best_layer, np.arange(N)]
    valid = np.isfinite(best_dist)

    # Compute u and v for valid pixels
    theta_frac = polar_theta / (2 * np.pi)
    u = best_layer.astype(np.float64) + theta_frac

    # v: normalized position within layer
    r_s = starts_at_pixel[best_layer, np.arange(N)]
    r_e = ends_at_pixel[best_layer, np.arange(N)]
    layer_width = r_e - r_s
    layer_width[layer_width <= 0] = 1.0  # avoid div by zero
    v = np.clip((polar_r - r_s) / layer_width, 0.0, 1.0)

    # Zero out invalid
    u[~valid] = 0.0
    v[~valid] = 0.0

    return u, v, valid


class UnwrapDataset:
    """Holds all data needed for unwrap training, GPU-resident.

    Attributes:
        coords: (N, 2) normalized pixel coordinates in [-1, 1], film pixels only
        u_target: (N,) initial u parameterization
        v_target: (N,) initial v parameterization
        intensities: (N,) pixel intensities [0, 1]
        seg_labels: (N,) segmentation class per pixel (1 or 2)
        film_yx: (N, 2) original pixel positions (for strip generation)
        n_layers: number of detected layers
        image_shape: (H, W)
    """

    def __init__(self, image, seg, center, device="cuda"):
        H, W = image.shape
        self.image_shape = (H, W)

        print("  Computing polar coordinates...")
        film_yx, polar_r, polar_theta = compute_polar(seg, center)
        n_film = len(film_yx)
        print(f"  Film pixels: {n_film:,} ({100 * n_film / (H * W):.1f}% of image)")

        print("  Detecting layers via radial scanning...")
        n_layers, layer_radii, layer_starts, layer_ends, angles = \
            assign_layers_polar(seg, center)
        self.n_layers = n_layers
        print(f"  Detected {n_layers} layers")

        print("  Computing initial (u, v) parameterization...")
        u, v, valid = compute_initial_uv_fast(
            film_yx, polar_r, polar_theta,
            n_layers, layer_radii, layer_starts, layer_ends, angles
        )
        n_valid = valid.sum()
        print(f"  Valid parameterization: {n_valid:,} / {n_film:,} "
              f"({100 * n_valid / n_film:.1f}%)")

        # Keep only valid pixels
        film_yx = film_yx[valid]
        u = u[valid]
        v = v[valid]

        # Normalized coordinates in [-1, 1]
        coords_y = 2.0 * film_yx[:, 0] / (H - 1) - 1.0
        coords_x = 2.0 * film_yx[:, 1] / (W - 1) - 1.0
        coords = np.stack([coords_x, coords_y], axis=-1).astype(np.float32)

        # Intensities
        intensities = image[film_yx[:, 0], film_yx[:, 1]].astype(np.float32)

        # Seg labels
        seg_labels = seg[film_yx[:, 0], film_yx[:, 1]].astype(np.int64)

        # Store raw u, v ranges for denormalization at eval time
        # u is in [0, n_layers], v is in [0, 1]
        self.u_min = float(u.min())
        self.u_max = float(u.max())
        self.v_min = float(v.min())
        self.v_max = float(v.max())
        print(f"  Raw u range: [{self.u_min:.2f}, {self.u_max:.2f}]")
        print(f"  Raw v range: [{self.v_min:.2f}, {self.v_max:.2f}]")

        # Standardize targets: zero mean, unit variance
        # This lets the network output any real value, no sigmoid needed
        u_f = u.astype(np.float32)
        v_f = v.astype(np.float32)
        self.u_mean = float(u_f.mean())
        self.u_std = float(u_f.std())
        self.v_mean = float(v_f.mean())
        self.v_std = float(v_f.std())

        u_norm = (u_f - self.u_mean) / (self.u_std + 1e-8)
        v_norm = (v_f - self.v_mean) / (self.v_std + 1e-8)
        print(f"  u stats: mean={self.u_mean:.4f}, std={self.u_std:.4f}")
        print(f"  v stats: mean={self.v_mean:.4f}, std={self.v_std:.4f}")

        # Move to GPU
        self.coords = torch.from_numpy(coords).to(device)
        self.u_target = torch.from_numpy(u_norm).to(device)
        self.v_target = torch.from_numpy(v_norm).to(device)
        self.intensities = torch.from_numpy(intensities).to(device)
        self.seg_labels = torch.from_numpy(seg_labels).to(device)
        self.film_yx = torch.from_numpy(film_yx.astype(np.int64)).to(device)
        self.n = len(coords)

        # Store raw u for strip generation
        self.u_raw = torch.from_numpy(u_f).to(device)

    def sample(self, batch_size):
        """Sample a random batch of film pixels."""
        idx = torch.randint(0, self.n, (batch_size,), device=self.coords.device)
        return {
            "coords": self.coords[idx],
            "u_target": self.u_target[idx],
            "v_target": self.v_target[idx],
            "intensities": self.intensities[idx],
            "seg_labels": self.seg_labels[idx],
        }
