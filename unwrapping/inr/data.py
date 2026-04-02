"""
Data loading utilities for INR training on CT reconstruction data.

Supports:
- 2D HDF5 slices (key='image', uint16, 3063x3062)
- 3D HDF5 volumes (key='volume', uint16, 3063x3062x20)
- Optional segmentation from probability maps (key='exported_data')

Coordinates are normalized to [-1, 1]. Intensities normalized to [0, 1].
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def load_2d_slice(image_path: str, probs_path: str = None):
    """Load a single 2D CT slice and optional segmentation."""
    with h5py.File(image_path, "r") as f:
        image = f["image"][:].astype(np.float32)

    # Normalize intensity to [0, 1]
    imin, imax = image.min(), image.max()
    image = (image - imin) / (imax - imin + 1e-8)

    seg = None
    if probs_path is not None:
        with h5py.File(probs_path, "r") as f:
            probs = f["exported_data"][:].astype(np.float32)
        seg = np.argmax(probs, axis=-1).astype(np.int64)  # (H, W), values 0/1/2

    return image, seg


def load_3d_volume(volume_path: str, probs_path: str = None):
    """Load a 3D CT volume and optional segmentation."""
    with h5py.File(volume_path, "r") as f:
        volume = f["volume"][:].astype(np.float32)

    # Normalize intensity to [0, 1]
    vmin, vmax = volume.min(), volume.max()
    volume = (volume - vmin) / (vmax - vmin + 1e-8)

    seg = None
    if probs_path is not None:
        with h5py.File(probs_path, "r") as f:
            probs = f["exported_data"][:].astype(np.float32)
        seg = np.argmax(probs, axis=-1).astype(np.int64)

    return volume, seg


class CoordinateDataset2D(Dataset):
    """
    Dataset that yields (coordinate, intensity[, seg_label]) pairs from a 2D image.

    Coordinates are normalized to [-1, 1].
    """

    def __init__(self, image: np.ndarray, seg: np.ndarray = None):
        H, W = image.shape
        self.H, self.W = H, W
        self.n_pixels = H * W

        # Build flat coordinate array: (N, 2) with values in [-1, 1]
        ys = np.linspace(-1, 1, H, dtype=np.float32)
        xs = np.linspace(-1, 1, W, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        self.coords = np.stack([xx.ravel(), yy.ravel()], axis=-1)  # (N, 2) x, y

        self.intensities = image.ravel().astype(np.float32)  # (N,)
        self.seg = seg.ravel().astype(np.int64) if seg is not None else None

    def __len__(self):
        return self.n_pixels

    def __getitem__(self, idx):
        coord = torch.from_numpy(self.coords[idx])
        intensity = torch.tensor(self.intensities[idx], dtype=torch.float32)
        if self.seg is not None:
            label = torch.tensor(self.seg[idx], dtype=torch.long)
            return coord, intensity, label
        return coord, intensity


class CoordinateDataset3D(Dataset):
    """
    Dataset that yields (coordinate, intensity[, seg_label]) pairs from a 3D volume.

    Coordinates are normalized to [-1, 1]. Z is normalized independently
    to account for anisotropic resolution.
    """

    def __init__(self, volume: np.ndarray, seg: np.ndarray = None):
        # volume shape: (D, H, W) or (H, W, D) — we expect (D, H, W)
        if volume.ndim != 3:
            raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

        D, H, W = volume.shape
        self.D, self.H, self.W = D, H, W
        self.n_voxels = D * H * W

        # Build flat coordinate array: (N, 3) with values in [-1, 1]
        zs = np.linspace(-1, 1, D, dtype=np.float32)
        ys = np.linspace(-1, 1, H, dtype=np.float32)
        xs = np.linspace(-1, 1, W, dtype=np.float32)
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
        self.coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)  # (N, 3)

        self.intensities = volume.ravel().astype(np.float32)
        self.seg = seg.ravel().astype(np.int64) if seg is not None else None

    def __len__(self):
        return self.n_voxels

    def __getitem__(self, idx):
        coord = torch.from_numpy(self.coords[idx])
        intensity = torch.tensor(self.intensities[idx], dtype=torch.float32)
        if self.seg is not None:
            label = torch.tensor(self.seg[idx], dtype=torch.long)
            return coord, intensity, label
        return coord, intensity


class RandomCoordinateSampler:
    """
    Efficiently samples random coordinate-value pairs without DataLoader overhead.

    For large images (9M+ pixels), this is faster than DataLoader with shuffle=True
    because it avoids full-dataset permutation each epoch.
    """

    def __init__(self, dataset, batch_size: int = 2**18, device: str = "cuda"):
        self.device = device
        self.batch_size = batch_size
        self.n = len(dataset)
        self.has_seg = dataset.seg is not None

        # Move everything to GPU at once
        self.coords = torch.from_numpy(dataset.coords).to(device)
        self.intensities = torch.from_numpy(dataset.intensities).to(device)
        if self.has_seg:
            self.seg = torch.from_numpy(dataset.seg).to(device)

    def sample(self):
        """Return a random batch of (coords, intensities[, seg_labels])."""
        idx = torch.randint(0, self.n, (self.batch_size,), device=self.device)
        coords = self.coords[idx]
        intensities = self.intensities[idx]
        if self.has_seg:
            return coords, intensities, self.seg[idx]
        return coords, intensities
