"""
Data preparation for 3D learned unwrapping (Step 1b — 3D).

Loads multiple 3D volume chunks, computes per-slice polar parameterization
as supervision targets, and provides CPU-resident dataset with GPU batch
transfer for training.

Each film voxel gets:
  - coords: (x, y, z) normalized to [-1, 1]
  - u_target: position along the unrolled film (layer + theta/2pi)
  - v_target: position within layer thickness [0, 1]
  - intensity: voxel absorption value
"""

import glob
import os
import re

import numpy as np
import torch

from unwrapping.inr.unwrap_data import (
    compute_polar, assign_layers_polar, compute_initial_uv_fast
)
from unwrapping.center_detection import find_spool_center


def discover_volumes(data_dir):
    """Find all volume + probability file pairs, sorted by z-range."""
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
    pairs.sort(key=lambda x: x[2])
    return pairs


def load_volume_chunk(volume_path, probs_path):
    """Load one 3D volume chunk and its segmentation."""
    import h5py

    with h5py.File(volume_path, "r") as f:
        volume = f["volume"][:].astype(np.float32)
    vmin, vmax = volume.min(), volume.max()
    volume = (volume - vmin) / (vmax - vmin + 1e-8)

    with h5py.File(probs_path, "r") as f:
        probs = f["exported_data"][:].astype(np.float32)
    seg = np.argmax(probs, axis=-1).astype(np.int64)

    return volume, seg


class UnwrapDataset3D:
    """Dataset for 3D unwrap training. Data on CPU, batches transferred to GPU.

    Processes all volume chunks, computes per-slice polar parameterization,
    and stores all film voxel data for random batch sampling.

    Attributes:
        coords: (N, 3) normalized voxel coords (x, y, z) in [-1, 1]
        u_target: (N,) standardized u target
        v_target: (N,) standardized v target
        intensities: (N,) voxel intensities
        z_global: (N,) global z-index for each voxel
        n_layers: number of detected spiral layers
        n: total number of film voxels
    """

    def __init__(self, data_dir, device="cuda", max_slices=None):
        pairs = discover_volumes(data_dir)
        if not pairs:
            raise ValueError(f"No volume files found in {data_dir}")

        total_slices = sum(z_end - z_start + 1 for _, _, z_start, z_end in pairs)
        print(f"  Found {len(pairs)} volume chunks, {total_slices} total slices")

        # Collect z-range for coordinate normalization
        z_min_global = min(z_start for _, _, z_start, _ in pairs)
        z_max_global = max(z_end for _, _, _, z_end in pairs)
        self.z_min_global = z_min_global
        self.z_max_global = z_max_global

        # Process all chunks
        all_coords = []
        all_u = []
        all_v = []
        all_intensities = []
        all_z_global = []
        n_layers = None
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

                center = find_spool_center(seg_2d)
                film_yx, polar_r, polar_theta = compute_polar(seg_2d, center)

                if n_layers is None:
                    nl, layer_radii, layer_starts, layer_ends, angles = \
                        assign_layers_polar(seg_2d, center)
                    n_layers = nl
                    # Reuse first slice's layer structure for consistency
                    ref_layers = (n_layers, layer_radii, layer_starts, layer_ends, angles)
                    print(f"    Reference layers: {n_layers} (from z={global_z})")

                u, v, valid = compute_initial_uv_fast(
                    film_yx, polar_r, polar_theta,
                    *ref_layers
                )

                film_yx_v = film_yx[valid]
                u_v = u[valid]
                v_v = v[valid]

                # Normalized coordinates: x, y in [-1, 1], z in [-1, 1]
                coords_x = 2.0 * film_yx_v[:, 1] / (W - 1) - 1.0
                coords_y = 2.0 * film_yx_v[:, 0] / (H - 1) - 1.0
                if z_max_global > z_min_global:
                    coords_z = 2.0 * (global_z - z_min_global) / (z_max_global - z_min_global) - 1.0
                else:
                    coords_z = 0.0
                coords_z_arr = np.full(len(film_yx_v), coords_z, dtype=np.float32)
                coords = np.stack([coords_x, coords_y, coords_z_arr], axis=-1).astype(np.float32)

                intensities = image_2d[film_yx_v[:, 0], film_yx_v[:, 1]].astype(np.float32)

                all_coords.append(coords)
                all_u.append(u_v.astype(np.float32))
                all_v.append(v_v.astype(np.float32))
                all_intensities.append(intensities)
                all_z_global.extend([global_z] * len(film_yx_v))

                slices_done += 1
                n_valid = len(film_yx_v)
                n_film = (seg_2d > 0).sum()
                if local_z % 5 == 0 or local_z == D - 1:
                    print(f"    z={global_z}: {n_valid:,}/{n_film:,} valid "
                          f"({100*n_valid/max(n_film,1):.0f}%)")

                if max_slices and slices_done >= max_slices:
                    break

            del volume, seg
            if max_slices and slices_done >= max_slices:
                print(f"  Reached max_slices={max_slices}, stopping")
                break

        # Concatenate all
        print(f"  Concatenating {slices_done} slices...")
        all_coords = np.concatenate(all_coords, axis=0)
        all_u = np.concatenate(all_u, axis=0)
        all_v = np.concatenate(all_v, axis=0)
        all_intensities = np.concatenate(all_intensities, axis=0)
        all_z_global = np.array(all_z_global, dtype=np.int64)

        self.n = len(all_coords)
        self.n_layers = n_layers
        self.device = device
        print(f"  Total film voxels: {self.n:,} across {slices_done} slices")

        # Store raw ranges for denormalization
        self.u_min = float(all_u.min())
        self.u_max = float(all_u.max())
        self.v_min = float(all_v.min())
        self.v_max = float(all_v.max())
        print(f"  Raw u range: [{self.u_min:.2f}, {self.u_max:.2f}]")
        print(f"  Raw v range: [{self.v_min:.2f}, {self.v_max:.2f}]")

        # Standardize targets
        self.u_mean = float(all_u.mean())
        self.u_std = float(all_u.std())
        self.v_mean = float(all_v.mean())
        self.v_std = float(all_v.std())
        print(f"  u stats: mean={self.u_mean:.4f}, std={self.u_std:.4f}")
        print(f"  v stats: mean={self.v_mean:.4f}, std={self.v_std:.4f}")

        u_norm = (all_u - self.u_mean) / (self.u_std + 1e-8)
        v_norm = (all_v - self.v_mean) / (self.v_std + 1e-8)

        # Store on CPU as pinned memory for fast GPU transfer
        self.coords = torch.from_numpy(all_coords).pin_memory()
        self.u_target = torch.from_numpy(u_norm).pin_memory()
        self.v_target = torch.from_numpy(v_norm).pin_memory()
        self.intensities = torch.from_numpy(all_intensities).pin_memory()
        self.z_global = torch.from_numpy(all_z_global).pin_memory()

        mem_gb = (all_coords.nbytes + all_u.nbytes + all_v.nbytes +
                  all_intensities.nbytes + all_z_global.nbytes) / 1e9
        print(f"  CPU memory: {mem_gb:.1f} GB")

    def sample(self, batch_size):
        """Sample a random batch and transfer to GPU."""
        idx = torch.randint(0, self.n, (batch_size,))
        return {
            "coords": self.coords[idx].to(self.device, non_blocking=True),
            "u_target": self.u_target[idx].to(self.device, non_blocking=True),
            "v_target": self.v_target[idx].to(self.device, non_blocking=True),
            "intensities": self.intensities[idx].to(self.device, non_blocking=True),
            "z_global": self.z_global[idx].to(self.device, non_blocking=True),
        }
