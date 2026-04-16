"""
Self-supervised data module for 3D unwrapping.

Loads film voxels with geometric metadata (radius, boundary flags) but
NO pre-computed UV labels. The model discovers the mapping from geometric
constraints alone: eikonal regularity, boundary anchoring, radial ordering.

Data loading: ~0.15s/slice → 500 slices in ~75s.
"""

import glob
import os
import re

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from unwrapping.center_detection import find_spool_center
from unwrapping.inr.unwrap_data_3d import discover_volumes, load_volume_chunk


def detect_n_layers(seg_2d, center, sigma=3.0):
    """Estimate number of spiral layers from radial histogram of film pixels.

    Counts peaks in the smoothed radial histogram. Fast: ~0.01s.
    """
    cy, cx = center
    ys, xs = np.where(seg_2d > 0)
    r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)

    r_int = np.floor(r).astype(np.int32)
    hist = np.bincount(r_int, minlength=r_int.max() + 1).astype(np.float32)
    smooth = gaussian_filter1d(hist, sigma=sigma)

    valleys, _ = find_peaks(-smooth, distance=8, prominence=smooth.max() * 0.08)
    return len(valleys) + 1


def detect_boundaries(seg_2d, center, n_angles=720):
    """Detect inner/outer film boundaries by radial scanning (vectorized).

    At each angle, the first film pixel hit (smallest r) is inner boundary,
    the last film pixel hit (largest r) is outer boundary.

    Returns boolean masks: inner_mask, outer_mask (same shape as seg_2d).
    """
    cy, cx = center
    H, W = seg_2d.shape
    r_max = int(np.sqrt(max(cy, H - cy) ** 2 + max(cx, W - cx) ** 2)) + 1

    inner_mask = np.zeros((H, W), dtype=bool)
    outer_mask = np.zeros((H, W), dtype=bool)

    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    sin_a = np.sin(angles)
    cos_a = np.cos(angles)
    radii = np.arange(1, r_max)

    for ai in range(n_angles):
        ys = np.round(cy + radii * sin_a[ai]).astype(np.int32)
        xs = np.round(cx + radii * cos_a[ai]).astype(np.int32)

        valid = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        # Find first out-of-bounds to truncate
        if not valid.all():
            first_oob = np.argmin(valid)
            if first_oob == 0:
                continue
            ys = ys[:first_oob]
            xs = xs[:first_oob]

        film_hits = seg_2d[ys, xs] > 0
        if not film_hits.any():
            continue

        hit_indices = np.where(film_hits)[0]
        inner_mask[ys[hit_indices[0]], xs[hit_indices[0]]] = True
        outer_mask[ys[hit_indices[-1]], xs[hit_indices[-1]]] = True

    return inner_mask, outer_mask


class SelfSupDataset3D:
    """Self-supervised dataset for 3D unwrap training.

    Stores film voxel coordinates with geometric metadata but NO UV labels.
    The model learns the mapping from eikonal, boundary, and ordering losses.

    Attributes:
        coords: (N, 3) normalized voxel coords in [-1, 1]
        intensities: (N,) voxel absorption values
        radius: (N,) pixel distance from spool center
        is_inner: (N,) bool, inner film boundary
        is_outer: (N,) bool, outer film boundary
        z_global: (N,) global z-index
        n_layers_est: estimated number of spiral layers
        n: total number of film voxels
    """

    def __init__(self, data_dir, device="cuda", max_slices=None):
        pairs = discover_volumes(data_dir)
        if not pairs:
            raise ValueError(f"No volume files found in {data_dir}")

        total_slices = sum(z_end - z_start + 1 for _, _, z_start, z_end in pairs)
        print(f"  Found {len(pairs)} volume chunks, {total_slices} total slices")

        z_min_global = min(z_start for _, _, z_start, _ in pairs)
        z_max_global = max(z_end for _, _, _, z_end in pairs)
        self.z_min_global = z_min_global
        self.z_max_global = z_max_global

        all_coords = []
        all_intensities = []
        all_radius = []
        all_inner = []
        all_outer = []
        all_z_global = []

        n_layers_est = None
        center = None  # Compute once, reuse for all slices
        slices_done = 0

        for chunk_idx, (vol_path, probs_path, z_start, z_end) in enumerate(pairs):
            print(f"  Chunk {chunk_idx + 1}/{len(pairs)}: z=[{z_start}-{z_end}]")
            volume, seg = load_volume_chunk(vol_path, probs_path)
            D, H, W = volume.shape

            if chunk_idx == 0:
                self.image_shape = (H, W)

            for local_z in range(D):
                global_z = z_start + local_z
                image_2d = volume[local_z]
                seg_2d = seg[local_z]

                # Compute center once from first slice, reuse for all
                if center is None:
                    center = find_spool_center(seg_2d)
                    print(f"    Spool center: ({center[0]:.0f}, {center[1]:.0f})")
                    # Normalized center for tangent-alignment loss (v3)
                    self.center_norm = (
                        2.0 * center[1] / (W - 1) - 1.0,  # cx → x_norm
                        2.0 * center[0] / (H - 1) - 1.0,   # cy → y_norm
                    )
                cy, cx = center

                # Estimate layer count once
                if n_layers_est is None:
                    n_layers_est = detect_n_layers(seg_2d, center)
                    print(f"    Estimated layers: {n_layers_est} (from z={global_z})")

                # Detect inner/outer boundaries
                inner_mask, outer_mask = detect_boundaries(seg_2d, center)

                # Film pixel coordinates
                film_mask = seg_2d > 0
                ys, xs = np.where(film_mask)
                n_film = len(ys)

                # Radius from center
                r = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2).astype(np.float32)

                # Boundary flags for this slice's film pixels
                is_inner = inner_mask[ys, xs]
                is_outer = outer_mask[ys, xs]

                # Normalized coords
                coords_x = (2.0 * xs / (W - 1) - 1.0).astype(np.float32)
                coords_y = (2.0 * ys / (H - 1) - 1.0).astype(np.float32)
                if z_max_global > z_min_global:
                    z_norm = np.float32(
                        2.0 * (global_z - z_min_global)
                        / (z_max_global - z_min_global) - 1.0
                    )
                else:
                    z_norm = np.float32(0.0)
                coords_z = np.full(n_film, z_norm, dtype=np.float32)
                coords = np.stack([coords_x, coords_y, coords_z], axis=-1)

                intensities = image_2d[ys, xs].astype(np.float32)

                all_coords.append(coords)
                all_intensities.append(intensities)
                all_radius.append(r)
                all_inner.append(is_inner)
                all_outer.append(is_outer)
                all_z_global.extend([global_z] * n_film)

                slices_done += 1
                if local_z % 10 == 0 or local_z == D - 1:
                    n_inner = is_inner.sum()
                    n_outer = is_outer.sum()
                    print(f"    z={global_z}: {n_film:,} film, "
                          f"{n_inner} inner, {n_outer} outer")

                if max_slices and slices_done >= max_slices:
                    break

            del volume, seg
            if max_slices and slices_done >= max_slices:
                print(f"  Reached max_slices={max_slices}, stopping")
                break

        print(f"  Concatenating {slices_done} slices...")
        all_coords = np.concatenate(all_coords, axis=0)
        all_intensities = np.concatenate(all_intensities, axis=0)
        all_radius = np.concatenate(all_radius, axis=0)
        all_inner = np.concatenate(all_inner, axis=0)
        all_outer = np.concatenate(all_outer, axis=0)
        all_z_global = np.array(all_z_global, dtype=np.int64)

        self.n = len(all_coords)
        self.n_layers_est = n_layers_est
        self.device = device
        print(f"  Total film voxels: {self.n:,} across {slices_done} slices")
        print(f"  Inner boundary: {all_inner.sum():,} pixels "
              f"({100*all_inner.mean():.2f}%)")
        print(f"  Outer boundary: {all_outer.sum():,} pixels "
              f"({100*all_outer.mean():.2f}%)")

        # Build separate index arrays for boundary oversampling
        self._inner_idx = np.where(all_inner)[0]
        self._outer_idx = np.where(all_outer)[0]

        # Keep numpy arrays for pair sampling (shares memory with tensors)
        self._radius_np = all_radius

        # Convert to tensors — torch.from_numpy shares memory, no copy
        self.coords = torch.from_numpy(all_coords)
        self.intensities = torch.from_numpy(all_intensities)
        self.radius = torch.from_numpy(all_radius)
        self.is_inner = torch.from_numpy(all_inner.astype(np.uint8))
        self.is_outer = torch.from_numpy(all_outer.astype(np.uint8))
        self.z_global = torch.from_numpy(all_z_global)

        mem_gb = (all_coords.nbytes + all_intensities.nbytes + all_radius.nbytes
                  + all_inner.nbytes + all_outer.nbytes + all_z_global.nbytes) / 1e9
        print(f"  CPU memory: {mem_gb:.1f} GB")

    def sample(self, batch_size, boundary_frac=0.1):
        """Sample a random batch with boundary oversampling.

        10% of the batch is guaranteed to be boundary pixels (5% inner, 5% outer).
        """
        n_boundary = int(batch_size * boundary_frac)
        n_inner = n_boundary // 2
        n_outer = n_boundary - n_inner
        n_random = batch_size - n_boundary

        # Random film pixels
        idx_random = torch.randint(0, self.n, (n_random,))

        # Oversampled boundary pixels
        if len(self._inner_idx) > 0 and n_inner > 0:
            idx_inner = self._inner_idx[
                np.random.randint(0, len(self._inner_idx), n_inner)
            ]
            idx_inner = torch.from_numpy(idx_inner)
        else:
            idx_inner = torch.zeros(0, dtype=torch.long)

        if len(self._outer_idx) > 0 and n_outer > 0:
            idx_outer = self._outer_idx[
                np.random.randint(0, len(self._outer_idx), n_outer)
            ]
            idx_outer = torch.from_numpy(idx_outer)
        else:
            idx_outer = torch.zeros(0, dtype=torch.long)

        idx = torch.cat([idx_random, idx_inner, idx_outer])

        return {
            "coords": self.coords[idx].to(self.device, non_blocking=True),
            "intensities": self.intensities[idx].to(self.device, non_blocking=True),
            "radius": self.radius[idx].to(self.device, non_blocking=True),
            "is_inner": self.is_inner[idx].to(self.device, non_blocking=True),
            "is_outer": self.is_outer[idx].to(self.device, non_blocking=True),
            "z_global": self.z_global[idx].to(self.device, non_blocking=True),
        }

    def sample_pairs(self, n_pairs):
        """Sample ordered pairs where r_a < r_b for radial ordering loss.

        Returns two batches (a, b) where a has smaller radius.
        """
        # Sample 2*n_pairs random indices, pair them, sort by radius
        idx = np.random.randint(0, self.n, (n_pairs, 2))
        r0 = self._radius_np[idx[:, 0]]
        r1 = self._radius_np[idx[:, 1]]

        # Ensure a is inner, b is outer
        swap = r0 > r1
        idx_a = np.where(swap, idx[:, 1], idx[:, 0])
        idx_b = np.where(swap, idx[:, 0], idx[:, 1])

        idx_a = torch.from_numpy(idx_a)
        idx_b = torch.from_numpy(idx_b)

        coords_a = self.coords[idx_a].to(self.device, non_blocking=True)
        coords_b = self.coords[idx_b].to(self.device, non_blocking=True)

        return coords_a, coords_b

    def sample_angular_pairs(self, n_pairs, radius_tol=5.0):
        """Sample pairs at similar radius (same winding) for angular diversity.

        For u=f(r), these pairs have u_a ≈ u_b → loss fires on ALL pairs.
        For correct spiral, u_a ≠ u_b → loss is naturally small.
        """
        # Sample reference points
        idx_a = np.random.randint(0, self.n, (n_pairs,))
        r_a = self._radius_np[idx_a]

        # For each, find point at similar radius from random candidates
        n_cand = 10
        idx_cand = np.random.randint(0, self.n, (n_pairs, n_cand))
        r_cand = self._radius_np[idx_cand]  # (n_pairs, n_cand)

        r_diff = np.abs(r_cand - r_a[:, None])
        best = r_diff.argmin(axis=1)
        idx_b = idx_cand[np.arange(n_pairs), best]

        # Keep only pairs within tolerance
        close = np.abs(self._radius_np[idx_a] - self._radius_np[idx_b]) < radius_tol
        idx_a = idx_a[close]
        idx_b = idx_b[close]

        coords_a = self.coords[torch.from_numpy(idx_a)].to(self.device, non_blocking=True)
        coords_b = self.coords[torch.from_numpy(idx_b)].to(self.device, non_blocking=True)

        return coords_a, coords_b
