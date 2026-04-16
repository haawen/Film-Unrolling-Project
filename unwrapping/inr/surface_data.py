"""
Data module for Vesuvius-style inverse-mapping INR unwrapping.

The INR learns a correction on top of a fixed analytical spiral:
    f(u, z) = analytical_xy(u) + INR_residual(u, z)

This module provides:
  - A GPU-resident mask volume (Z, H, W) with classes {film base, emulsion}
    collapsed to a single binary channel — used by the attachment loss.
  - A GPU-resident image volume (Z, H, W) — used by eval strip sampling.
  - The analytical per-winding spiral f_analytical(u) → (x_norm, y_norm).
  - A uniform (u, z) sampler.

All heavy detection (center, winding boundaries, seam) runs ONCE on a single
mid-stack reference slice at init time — the segmentation is reliable enough
that per-slice redetection would add noise, not accuracy.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.center_detection import find_spool_center
from unwrapping.inr.geodesic_unwrap import _detect_seam_angle
from unwrapping.inr.unwrap_data_3d import discover_volumes, load_volume_chunk


def detect_winding_boundaries_normalized(film_mask, center_yx, sigma=2.0,
                                          min_spacing=15, out_dir=None):
    """Detect winding boundaries from the radial histogram, normalized by 2πr.

    The raw radial histogram is biased: outer radii have more pixels simply
    because their circumference is larger. Dividing by 2πr gives the fraction
    of each circle occupied by film — valleys in this normalized signal have
    consistent depth at all radii, so a single prominence threshold works.

    Returns: (boundaries, n_layers)
        boundaries: array [r_min, v1, ..., v_n, r_max] with valleys at air gaps
        n_layers: number of windings (len(valleys) + 1)
    """
    cy, cx = center_yx
    ys, xs = np.where(film_mask)
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)

    r_int = np.floor(r).astype(np.int32)
    hist = np.bincount(r_int, minlength=r_int.max() + 1).astype(np.float64)

    # Normalize by circumference: fraction of circle that is film at each radius
    r_arr = np.arange(len(hist), dtype=np.float64)
    circumference = 2.0 * np.pi * np.maximum(r_arr, 1.0)
    hist_norm = hist / circumference

    # Only consider the radial range where film actually exists
    r_lo, r_hi = int(r.min()), int(r.max())
    hist_norm[:r_lo] = 0.0
    hist_norm[r_hi + 1:] = 0.0

    smooth = gaussian_filter1d(hist_norm, sigma=sigma)

    # In the film region, find valleys (air gaps between windings).
    # Restrict search to [r_lo+5, r_hi-5] to avoid edge artifacts.
    search = smooth.copy()
    search[:r_lo + 5] = search[r_lo + 5]
    search[r_hi - 5:] = search[r_hi - 5]

    # Prominence relative to the peak of the normalized signal in the film band
    peak_val = smooth[r_lo:r_hi + 1].max()
    valleys, props = find_peaks(
        -search, distance=min_spacing, prominence=peak_val * 0.03,
    )

    # Keep only valleys that fall within the film radial range
    valleys = valleys[(valleys >= r_lo) & (valleys <= r_hi)]
    n_layers = len(valleys) + 1

    boundaries = np.concatenate([[r.min()], valleys.astype(np.float64), [r.max()]])

    # Diagnostic plot
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fig, axes = plt.subplots(2, 1, figsize=(16, 8))

        axes[0].plot(hist, color="steelblue", lw=0.5)
        axes[0].set_title("Raw radial histogram (biased by circumference)")
        axes[0].set_xlabel("Radius (px)")
        axes[0].set_ylabel("Pixel count")
        axes[0].set_xlim(r_lo - 20, r_hi + 20)

        axes[1].plot(smooth, color="steelblue", lw=0.8, label="smoothed")
        axes[1].plot(hist_norm, color="lightgray", lw=0.3, alpha=0.5,
                     label="raw normalized")
        for v in valleys:
            axes[1].axvline(v, color="red", lw=0.6, alpha=0.7)
        axes[1].set_title(
            f"Normalized by 2πr — {len(valleys)} valleys → {n_layers} windings"
        )
        axes[1].set_xlabel("Radius (px)")
        axes[1].set_ylabel("Film fraction")
        axes[1].set_xlim(r_lo - 20, r_hi + 20)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "radial_histogram.png"), dpi=200)
        plt.close()

    return boundaries, n_layers


class SurfaceDataset:
    """Inverse-mapping dataset: samples (u, z), provides analytical spiral + mask.

    Attributes:
        mask_volume: (Z, H, W) float32 GPU tensor, 1.0 where seg > 0 (film ∪ emulsion).
        image_volume: (Z, H, W) float32 GPU tensor, normalized intensities.
        Z, H, W: volume dimensions.
        cx, cy: spool center in pixel coords.
        boundaries: (n_layers + 1,) GPU tensor of radii (inner → outer).
        theta_seam: scalar, seam angle in radians.
        n_layers: int, detected winding count.
        device: torch device.
    """

    def __init__(self, data_dir, max_slices=None, device="cuda", diag_dir=None):
        self.device = torch.device(device)

        pairs = discover_volumes(data_dir)
        if not pairs:
            raise ValueError(f"No volume files found in {data_dir}")

        print(f"  Loading {len(pairs)} volume chunks...")

        vols, segs, z_indices = [], [], []
        slices_done = 0
        for vol_path, probs_path, z_start, z_end in pairs:
            volume, seg = load_volume_chunk(vol_path, probs_path)
            D = volume.shape[0]
            for local_z in range(D):
                vols.append(volume[local_z])
                segs.append(seg[local_z])
                z_indices.append(z_start + local_z)
                slices_done += 1
                if max_slices and slices_done >= max_slices:
                    break
            del volume, seg
            if max_slices and slices_done >= max_slices:
                break

        image_stack = np.stack(vols, axis=0).astype(np.float32)   # (Z, H, W)
        seg_stack = np.stack(segs, axis=0).astype(np.int64)       # (Z, H, W)
        self.z_indices = np.array(z_indices, dtype=np.int64)
        self.Z, self.H, self.W = image_stack.shape

        # Reference slice (mid-stack) for center/boundaries/seam detection.
        ref_idx = self.Z // 2
        ref_seg = seg_stack[ref_idx]
        ref_film_mask = ref_seg > 0

        cy, cx = find_spool_center(ref_seg)
        self.cy = float(cy)
        self.cx = float(cx)
        print(f"  Spool center: ({self.cy:.1f}, {self.cx:.1f})")

        boundaries_np, n_layers = detect_winding_boundaries_normalized(
            ref_film_mask, (cy, cx), out_dir=diag_dir,
        )
        self.n_layers = int(n_layers)
        assert len(boundaries_np) == n_layers + 1, (
            f"Boundary count mismatch: {len(boundaries_np)} vs n_layers+1={n_layers+1}"
        )
        print(f"  Detected {self.n_layers} windings, "
              f"r=[{boundaries_np[0]:.1f}, {boundaries_np[-1]:.1f}]")

        ys_f, xs_f = np.where(ref_film_mask)
        r_f = np.sqrt((ys_f - cy) ** 2 + (xs_f - cx) ** 2)
        theta_seam = _detect_seam_angle(ref_film_mask, (cy, cx), r_f.min(), r_f.max())
        self.theta_seam = float(theta_seam)
        print(f"  Seam angle: {np.degrees(self.theta_seam):.1f}°")

        # GPU tensors — stored as (1, 1, Z, H, W) for 3D grid_sample
        self.mask_volume = torch.from_numpy(
            (seg_stack > 0).astype(np.float32)
        ).to(self.device).view(1, 1, self.Z, self.H, self.W)
        self.image_volume = torch.from_numpy(image_stack).to(
            self.device
        ).view(1, 1, self.Z, self.H, self.W)
        self.boundaries = torch.from_numpy(boundaries_np.astype(np.float32)).to(self.device)

        # Precompute per-winding delta for vectorized analytical lookup
        # segment k spans u ∈ [k, k+1), r goes from boundaries[k] → boundaries[k+1]
        self.boundary_deltas = self.boundaries[1:] - self.boundaries[:-1]   # (n_layers,)

        vol_gb = (self.mask_volume.element_size() * self.mask_volume.nelement()
                  + self.image_volume.element_size() * self.image_volume.nelement()) / 1e9
        print(f"  GPU memory (mask+image): {vol_gb:.2f} GB")

    def sample(self, batch_size):
        """Sample uniform (u, z) pairs.

        Returns dict with:
            uv_norm: (B, 2) in [-1, 1] for INR input
            u_raw:   (B,)   float in [0, n_layers)
            z_idx:   (B,)   int64 in [0, Z)
        """
        u_raw = torch.rand(batch_size, device=self.device) * self.n_layers
        z_idx = torch.randint(0, self.Z, (batch_size,), device=self.device)

        u_norm = u_raw / self.n_layers * 2.0 - 1.0
        if self.Z > 1:
            z_norm = z_idx.float() / (self.Z - 1) * 2.0 - 1.0
        else:
            z_norm = torch.zeros(batch_size, device=self.device)

        uv_norm = torch.stack([u_norm, z_norm], dim=1)
        return {"uv_norm": uv_norm, "u_raw": u_raw, "z_idx": z_idx}

    def analytical_xy_norm(self, u_raw):
        """Per-winding piecewise-linear Archimedean spiral.

        For u in [k, k+1):
            r(u) = boundaries[k] + (u - k) * (boundaries[k+1] - boundaries[k])
            theta(u) = 2π * u + theta_seam
            x(u) = cx + r * cos(theta)
            y(u) = cy + r * sin(theta)

        Returns (x_norm, y_norm) in [-1, 1].
        """
        k = torch.clamp(u_raw.long(), 0, self.n_layers - 1)
        frac = u_raw - k.float()
        r_base = self.boundaries[k] + frac * self.boundary_deltas[k]

        theta = 2.0 * torch.pi * u_raw + self.theta_seam
        x = self.cx + r_base * torch.cos(theta)
        y = self.cy + r_base * torch.sin(theta)

        x_norm = 2.0 * x / (self.W - 1) - 1.0
        y_norm = 2.0 * y / (self.H - 1) - 1.0
        return torch.stack([x_norm, y_norm], dim=1)

    def _z_norm(self, z_idx):
        """Convert integer z indices to [-1, 1] range for grid_sample."""
        if self.Z > 1:
            return z_idx.float() / (self.Z - 1) * 2.0 - 1.0
        return torch.zeros(z_idx.shape[0], device=self.device)

    def _build_grid_3d(self, xy_norm, z_idx):
        """Build a (1, B, 1, 1, 3) grid for 3D grid_sample.

        grid_sample 3D convention:
            grid[..., 0] → W (x), grid[..., 1] → H (y), grid[..., 2] → D (z)
        """
        B = xy_norm.shape[0]
        z_n = self._z_norm(z_idx)
        grid = torch.stack([xy_norm[:, 0], xy_norm[:, 1], z_n], dim=1)  # (B, 3)
        return grid.view(1, B, 1, 1, 3)

    def sample_mask(self, xy_norm, z_idx):
        """Bilinear-sample the binary film mask at predicted positions.

        Uses 3D grid_sample on the (1, 1, Z, H, W) volume — no per-sample
        slice expansion, constant memory regardless of batch size.

        Args:
            xy_norm: (B, 2) in [-1, 1]
            z_idx:   (B,)   int64 slice indices

        Returns:
            (B,) float in [0, 1].
        """
        grid = self._build_grid_3d(xy_norm, z_idx)
        sampled = torch.nn.functional.grid_sample(
            self.mask_volume, grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        )
        return sampled.view(-1)

    def sample_image(self, xy_norm, z_idx):
        """Bilinear-sample the intensity volume — used by eval for strip building."""
        grid = self._build_grid_3d(xy_norm, z_idx)
        sampled = torch.nn.functional.grid_sample(
            self.image_volume, grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        )
        return sampled.view(-1)
