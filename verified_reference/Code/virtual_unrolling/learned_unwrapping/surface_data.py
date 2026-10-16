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

import glob
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import (
    gaussian_filter1d, binary_erosion, distance_transform_edt,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from virtual_unrolling.common.spool_center import find_spool_center


# ---- verbatim copies of the helpers formerly imported from geodesic_unwrap / unwrap_data_3d


def _detect_seam_angle(film_mask, center_yx, r_min, r_max):
    """Detect the spiral opening angle by radial ray scanning.

    At most angles, a radial ray crosses all windings (film-air-film-...).
    At the spiral opening angle, one air gap is missing (two windings merge),
    so the ray crosses fewer air gaps.  We return the angle with the fewest
    film→air transitions.
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
        # Count film→air transitions (end of a winding)
        gap_counts[i] = int(np.sum(np.diff(vals) == -1))

    # Smooth to avoid noise, then pick the minimum
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(gap_counts.astype(np.float64), size=20,
                              mode="wrap")
    return angles[np.argmin(smooth)]


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


def load_volume_chunk(volume_path, probs_path, max_slices=None):
    """Load one 3D volume chunk and its segmentation.

    If `max_slices` is provided, only the first `max_slices` z-slices are
    read from the HDF5 — avoids OOM on large (e.g. 256-slice 4K) volumes.
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


def _circular_fill_and_smooth(per_angle, sigma_bins=8.0):
    """Fill NaNs in a (n_layers, n_rays) array via circular interpolation along
    the ray axis, then smooth each row with a wrap-around Gaussian.

    Used by the raycast detector to produce a clean per-angle centerline for
    each winding. NaNs occur on rays that didn't hit the k-th emulsion run
    (e.g. near the spiral seam where the innermost winding is missing).
    """
    out = np.array(per_angle, dtype=np.float64, copy=True)
    n_layers, n_rays = out.shape
    for k in range(n_layers):
        row = out[k]
        valid = ~np.isnan(row)
        if not valid.any():
            out[k] = 0.0
            continue
        if valid.all():
            continue
        # Circular linear interpolation: extend valid samples by one period.
        idx = np.arange(n_rays)
        v_idx = idx[valid]
        v_val = row[valid]
        # Pad with wrap-around so np.interp handles the seam.
        v_idx_pad = np.concatenate([v_idx - n_rays, v_idx, v_idx + n_rays])
        v_val_pad = np.concatenate([v_val, v_val, v_val])
        out[k] = np.interp(idx, v_idx_pad, v_val_pad)
    # Wrap-around Gaussian smooth: tile the row 3× and slice the middle.
    if sigma_bins > 0:
        smoothed = np.empty_like(out)
        for k in range(n_layers):
            tiled = np.concatenate([out[k], out[k], out[k]])
            sm = gaussian_filter1d(tiled, sigma=sigma_bins, mode="nearest")
            smoothed[k] = sm[n_rays:2 * n_rays]
        out = smoothed
    return out


def detect_winding_boundaries_raycast(seg_2d, center_yx, n_rays=720, min_run=2,
                                       out_dir=None, return_per_angle=False,
                                       per_angle_smooth_sigma=8.0):
    """Multi-angle ray-cast winding detector.

    For each ray from `center_yx`, traverse outward and count connected runs of
    emulsion class (=2). Each run = one winding's emulsion sub-band. Robust
    to pinch points and touching windings (which break histogram-based
    detectors): a pinch only affects local angles, and touching windings still
    have separate emulsion runs along any radial ray.

    Returns: (boundaries, n_layers) in the same format as
    `detect_winding_boundaries_normalized`:
        boundaries: [r_min, gap_1, ..., gap_{n-1}, r_max]
        n_layers:   number of windings (mode of per-ray emulsion-run counts)

    Per-winding emulsion centerlines are converted to inter-winding gap radii
    via the midpoint between consecutive centerlines, then padded with the
    outermost film-pixel radius bounds so the analytical-base derivation
    (which uses interior valleys to estimate layer_spacing) keeps working.
    """
    cy, cx = center_yx
    H, W = seg_2d.shape
    r_max_global = int(min(cy, cx, H - cy, W - cx)) - 2
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    radii = np.arange(r_max_global)

    # Vectorized: build (n_rays, r_max) sample array
    samples = np.empty((n_rays, r_max_global), dtype=np.uint8)
    for k, theta in enumerate(angles):
        ys = np.clip((cy + radii * np.sin(theta)).astype(np.int32), 0, H - 1)
        xs = np.clip((cx + radii * np.cos(theta)).astype(np.int32), 0, W - 1)
        samples[k] = seg_2d[ys, xs]

    # Per-ray emulsion-run counting
    is_emul = (samples == 2).astype(np.int32)
    per_ray_counts = np.zeros(n_rays, dtype=np.int32)
    per_ray_radii = []
    for k in range(n_rays):
        runs = []
        in_run = False
        start = 0
        v = is_emul[k]
        for i in range(len(v)):
            if v[i] and not in_run:
                start = i
                in_run = True
            elif not v[i] and in_run:
                if i - start >= min_run:
                    runs.append((start + i - 1) / 2.0)
                in_run = False
        if in_run and len(v) - start >= min_run:
            runs.append((start + len(v) - 1) / 2.0)
        per_ray_counts[k] = len(runs)
        per_ray_radii.append(runs)

    # Mode of per-ray counts → expected number of windings
    counts_unique, freqs = np.unique(per_ray_counts, return_counts=True)
    n_layers = int(counts_unique[np.argmax(freqs)])

    # Median radius of the k-th emulsion run across rays that have ≥ k+1 runs
    centerlines = np.full(n_layers, np.nan)
    # Per-angle centerlines (n_layers, n_rays): NaN where ray didn't hit run k.
    per_angle = np.full((n_layers, n_rays), np.nan, dtype=np.float64)
    for k in range(n_layers):
        rs = [r[k] for r in per_ray_radii if len(r) > k]
        if rs:
            centerlines[k] = float(np.median(rs))
        for ray_idx, runs in enumerate(per_ray_radii):
            if len(runs) > k:
                per_angle[k, ray_idx] = runs[k]

    # Convert centerlines to inter-winding gap radii (same format as histogram)
    # Outermost bounds come from actual film-pixel radial extent.
    film = seg_2d > 0
    if not film.any():
        raise ValueError("No film pixels in reference slice.")
    ys_f, xs_f = np.where(film)
    r_film = np.sqrt((ys_f - cy) ** 2 + (xs_f - cx) ** 2)
    r_min, r_max = float(r_film.min()), float(r_film.max())

    if n_layers >= 2:
        gaps = (centerlines[:-1] + centerlines[1:]) / 2.0
        boundaries = np.concatenate([[r_min], gaps, [r_max]])
    else:
        boundaries = np.array([r_min, r_max], dtype=np.float64)

    # Cleaned per-angle centerlines (NaN-filled + circular-Gaussian smoothed).
    per_angle_clean = _circular_fill_and_smooth(
        per_angle, sigma_bins=per_angle_smooth_sigma,
    )

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        # Cross-section overlay: data-driven per-angle curves (track non-circular).
        axes[0].imshow(seg_2d, cmap="gray")
        axes[0].plot(cx, cy, "r+", markersize=12)
        ang_plot = angles  # used by per-angle curves
        for k in range(n_layers):
            xs_curve = cx + per_angle_clean[k] * np.cos(ang_plot)
            ys_curve = cy + per_angle_clean[k] * np.sin(ang_plot)
            xs_curve = np.append(xs_curve, xs_curve[0])
            ys_curve = np.append(ys_curve, ys_curve[0])
            axes[0].plot(xs_curve, ys_curve, "-", color="cyan",
                         linewidth=0.4, alpha=0.7)
        axes[0].set_title(f"Raycast detector → {n_layers} windings (per-angle curves)")
        axes[0].set_xlim(0, W); axes[0].set_ylim(H, 0)
        # Per-ray count distribution
        axes[1].bar(counts_unique, freqs, width=0.8)
        axes[1].axvline(n_layers, color="r", linestyle="--",
                        label=f"mode = {n_layers}")
        axes[1].set_xlabel("emulsion runs per ray")
        axes[1].set_ylabel("number of rays")
        axes[1].set_title(
            f"Per-ray distribution ({freqs.max()}/{n_rays} rays at mode)"
        )
        axes[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "raycast_detector.png"), dpi=150)
        plt.close()

    if return_per_angle:
        return boundaries, n_layers, per_angle_clean
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

    def __init__(self, data_dir, device="cuda", diag_dir=None,
                 attachment="film", centerline_erode=1):
        """
        Args:
            attachment: which binary mask drives the attachment loss.
                "film"       — seg > 0 (class 1 + 2), 17px slab, original behavior
                "emulsion"   — seg == 2, thin band (~6px synthetic, ~3px real)
                "centerline" — emulsion eroded by `centerline_erode` iterations;
                               falls back to emulsion per-slice if erosion empties
            centerline_erode: erosion iterations for "centerline" mode (default 1)

        Windings are detected with the raycast detector on the mid-stack slice.
        """
        assert attachment in ("film", "emulsion", "centerline"), attachment
        self.attachment_mode = attachment
        self.centerline_erode = centerline_erode
        self.device = torch.device(device)

        pairs = discover_volumes(data_dir)
        if not pairs:
            raise ValueError(f"No volume files found in {data_dir}")

        print(f"  Loading {len(pairs)} volume chunks...")

        vols, segs, z_indices = [], [], []
        for vol_path, probs_path, z_start, z_end in pairs:
            volume, seg = load_volume_chunk(vol_path, probs_path)
            for local_z in range(volume.shape[0]):
                vols.append(volume[local_z])
                segs.append(seg[local_z])
                z_indices.append(z_start + local_z)
            del volume, seg

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

        boundaries_np, n_layers, _ = detect_winding_boundaries_raycast(
            ref_seg, (cy, cx), out_dir=diag_dir,
            return_per_angle=True,
        )
        print(f"  Raycast detector: {n_layers} windings")
        self.n_layers = int(n_layers)
        assert len(boundaries_np) == n_layers + 1, (
            f"Boundary count mismatch: {len(boundaries_np)} vs n_layers+1={n_layers+1}"
        )
        print(f"  r=[{boundaries_np[0]:.1f}, {boundaries_np[-1]:.1f}]")

        ys_f, xs_f = np.where(ref_film_mask)
        r_f = np.sqrt((ys_f - cy) ** 2 + (xs_f - cx) ** 2)
        theta_seam = _detect_seam_angle(ref_film_mask, (cy, cx), r_f.min(), r_f.max())
        self.theta_seam = float(theta_seam)
        print(f"  Seam angle: {np.degrees(self.theta_seam):.1f}°")

        # Build the attachment mask per `self.attachment_mode`.
        if attachment == "film":
            mask_np = (seg_stack > 0).astype(np.float32)
        elif attachment == "emulsion":
            mask_np = (seg_stack == 2).astype(np.float32)
        else:  # centerline
            emul_np = (seg_stack == 2)
            mask_np = np.zeros_like(emul_np, dtype=np.float32)
            n_fallback = 0
            for z in range(self.Z):
                eroded = binary_erosion(emul_np[z], iterations=centerline_erode)
                if eroded.sum() < 0.1 * emul_np[z].sum():
                    mask_np[z] = emul_np[z].astype(np.float32)
                    n_fallback += 1
                else:
                    mask_np[z] = eroded.astype(np.float32)
            if n_fallback:
                print(f"  Centerline: {n_fallback}/{self.Z} slices fell back "
                      f"to emulsion (erosion with n={centerline_erode} emptied mask)")
        frac = mask_np.mean()
        print(f"  Attachment mode: '{attachment}', mask fill fraction={frac:.4f}")

        # GPU tensors — stored as (1, 1, Z, H, W) for 3D grid_sample
        self.mask_volume = torch.from_numpy(mask_np).to(
            self.device
        ).view(1, 1, self.Z, self.H, self.W)
        self.image_volume = torch.from_numpy(image_stack).to(
            self.device
        ).view(1, 1, self.Z, self.H, self.W)

        # Signed distance to the attachment target, normalized by self.dist_scale
        # so sampled values land in [0, 1] (1 = far, 0 = on target). Using a
        # smooth distance field instead of (1 - mask)² gives a non-zero gradient
        # everywhere, which is the missing signal when the target is thin
        # (centerline, emulsion band) and binary attachment stalls off-target.
        self.dist_scale = 50.0  # pixels; distances past this saturate at 1.
        dist_np = np.empty_like(mask_np, dtype=np.float32)
        for z in range(self.Z):
            tm = mask_np[z] > 0.5
            if not tm.any():
                dist_np[z] = 1.0
                continue
            d = distance_transform_edt(~tm)
            dist_np[z] = np.clip(d / self.dist_scale, 0.0, 1.0).astype(np.float32)
        self.dist_volume = torch.from_numpy(dist_np).to(
            self.device
        ).view(1, 1, self.Z, self.H, self.W)

        # Keep the raw seg on CPU — used for supervised mode emulsion sampling.
        self.seg_stack = seg_stack
        self.boundaries = torch.from_numpy(boundaries_np.astype(np.float32)).to(self.device)

        # Precompute per-winding delta for vectorized analytical lookup
        # segment k spans u ∈ [k, k+1), r goes from boundaries[k] → boundaries[k+1]
        self.boundary_deltas = self.boundaries[1:] - self.boundaries[:-1]   # (n_layers,)
        # Signed radial offset from film centerline to attachment-target centerline.
        # 0 = land on film centerline (default). For synthetic GT this is set in
        # load_synthetic_gt to land directly on the emulsion centerline, so the
        # INR residual only needs to fix small geometric perturbations rather
        # than re-learn a 15px direction-varying offset for every winding.
        self.emulsion_offset_signed = 0.0
        # Arc-length parameterization: u_starts[k], u_ends[k] give the u range
        # assigned to winding k such that u advances proportionally to arc length.
        # This matches the synthetic generator's GT u_map (arc/total_arc * n_windings).
        self._compute_arc_parameters()

        vol_gb = (self.mask_volume.element_size() * self.mask_volume.nelement()
                  + self.image_volume.element_size() * self.image_volume.nelement()
                  + self.dist_volume.element_size() * self.dist_volume.nelement()) / 1e9
        print(f"  GPU memory (mask+image+dist): {vol_gb:.2f} GB, "
              f"dist_scale={self.dist_scale}px")

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


    def _compute_arc_parameters(self):
        """Precompute exact continuous-Archimedean parameters.

        Matches the synthetic generator exactly (see Code/virtual_unrolling/synthetic/generate.py):
            arc(θ_total) = r_inner·θ_total + 0.5·a·θ_total²   (∫ r dθ')
            u_norm       = arc / total_arc  ∈ [0, 1]

        We derive (r_inner, a, total_arc) from the boundary array:
            layer_spacing ≈ mean interior valley spacing
            r_inner       = centerline radius of innermost winding ≈ b[1] - ls/2
            a             = layer_spacing / (2π)
            total_arc     = r_inner·(2π·N) + 0.5·a·(2π·N)²
        """
        b = self.boundaries  # (n_layers + 1,)
        n = self.n_layers
        if n >= 2:
            # Use interior valleys (b[1]..b[n-1]) — most reliable estimate of spacing.
            ls = (b[n - 1] - b[1]) / max(1, (n - 2))
            r_inner = b[1] - 0.5 * ls
        else:
            ls = (b[1] - b[0])
            r_inner = b[0]
        a = ls / (2.0 * math.pi)
        theta_max = 2.0 * math.pi * n
        total_arc = r_inner * theta_max + 0.5 * a * theta_max ** 2
        self.layer_spacing = float(ls.item() if torch.is_tensor(ls) else ls)
        self.r_inner_centerline = float(r_inner.item() if torch.is_tensor(r_inner) else r_inner)
        self.spiral_advance = float(a.item() if torch.is_tensor(a) else a)
        self.total_arc = float(total_arc.item() if torch.is_tensor(total_arc) else total_arc)


    def analytical_xy_norm(self, u_raw):
        """Continuous Archimedean inverse: u → (x, y) on the spiral centerline.

        Given u_raw ∈ [0, n_layers), compute u_norm = u_raw / n_layers, then
        solve the quadratic  0.5·a·θ² + r_inner·θ − u_norm·total_arc = 0  for
        θ_total, from which r(θ) = r_inner + a·θ and (x, y) follow.

        Returns (x_norm, y_norm) in [-1, 1].
        """
        u_clamped = torch.clamp(u_raw, 0.0, self.n_layers - 1e-6)
        u_norm = u_clamped / self.n_layers
        a = self.spiral_advance
        r0 = self.r_inner_centerline
        total = self.total_arc
        disc = r0 * r0 + 2.0 * a * u_norm * total
        theta_total = (torch.sqrt(disc) - r0) / a
        theta = theta_total + self.theta_seam
        r = r0 + a * theta_total - self.emulsion_offset_signed
        x = self.cx + r * torch.cos(theta)
        y = self.cy + r * torch.sin(theta)
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


    def sample_distance(self, xy_norm, z_idx):
        """Bilinear-sample the normalized distance-to-target volume.

        Uses padding_mode='border' so pixels projected off the image get the
        saturated-far value (1.0) rather than 0.0 — otherwise the INR could
        trivially minimize attachment by pointing outside the image.
        """
        grid = self._build_grid_3d(xy_norm, z_idx)
        sampled = torch.nn.functional.grid_sample(
            self.dist_volume, grid, mode="bilinear",
            padding_mode="border", align_corners=True,
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

    # ─── Synthetic ground-truth supervision ────────────────────────────

    def load_synthetic_gt(self, gt_npz_path, max_slices=None, geometry_only=False):
        """Load ground_truth.npz from a synthetic preset.

        The GT `u_map` is (H_gt, W_gt), float32, NaN off-film, [0, 1] on-film
        (normalized by total spiral arc length).  Stored once; re-used across
        slices because synthetic geometry is z-invariant.

        Populates self.gt_emul_indices: (N, 3) int64 array of (z, y, x) for
        every emulsion pixel across the loaded z-stack, and self.gt_u_raw:
        (N,) float32 targets scaled to [0, n_layers).

        Args:
            geometry_only: if True, only override analytical geometry (n_layers,
                boundaries, center, seam) — skip building the supervised pool.
                Used by eval to match the geometry the model was trained under.
        """
        gt = np.load(gt_npz_path)
        u_map_2d = gt["u_map"]          # (H, W), NaN off-film
        seg_2d = gt["seg"]              # (H, W), {0,1,2}
        H_gt, W_gt = u_map_2d.shape
        assert H_gt == self.H and W_gt == self.W, (
            f"GT shape {u_map_2d.shape} mismatches dataset {(self.H, self.W)}"
        )
        self.gt_n_windings = int(gt["n_windings"])
        self.gt_strip = gt["strip"]     # (n_z_gt, strip_width)

        # Unconditionally snap geometry to GT when a synthetic GT is loaded.
        # Detection is fragile (center ~6px off, seam ~4° off on clean_4k), and
        # the exact Archimedean analytical amplifies those errors into
        # hundreds-of-pixels residuals the INR can't absorb smoothly.
        # Round 1 holds geometry fixed to GT; Round 2 tests detection-only runs.
        ls = float(gt["film_thickness"]) + float(gt["air_gap"])
        half_t = 0.5 * float(gt["film_thickness"])
        # r_inner_override is present for Mickey-matched presets; default
        # falls back to layer_spacing*3 to match the historical generator.
        if "r_inner_override" in gt.files:
            r_inner_center = float(gt["r_inner_override"])
        else:
            r_inner_center = ls * 3.0
        r_min = r_inner_center - half_t
        r_max = r_inner_center + (self.gt_n_windings - 1) * ls + half_t
        valleys = np.array(
            [r_inner_center + (k + 0.5) * ls
             for k in range(self.gt_n_windings - 1)],
            dtype=np.float32,
        )
        boundaries_np = np.concatenate([[r_min], valleys, [r_max]])
        print(f"  GT override: n_layers {self.n_layers} → "
              f"{self.gt_n_windings} (GT), r=[{r_min:.1f}, {r_max:.1f}], "
              f"center→({self.W/2.0:.1f},{self.H/2.0:.1f}), seam→0°")
        self.n_layers = int(self.gt_n_windings)
        self.boundaries = torch.from_numpy(
            boundaries_np.astype(np.float32)
        ).to(self.device)
        self.boundary_deltas = self.boundaries[1:] - self.boundaries[:-1]
        self.cy = float(self.H) / 2.0
        self.cx = float(self.W) / 2.0
        self.theta_seam = 0.0
        # Land the analytical on the *emulsion* centerline, not the film centerline.
        # Without this the residual must encode a direction-varying ~15px radial
        # offset per winding — which the INR can only approximate as "average
        # direction", failing badly on inner windings where dθ/du is large.
        emul_thickness = float(gt["film_thickness"]) * float(gt["emulsion_fraction"])
        offset_mag = half_t - 0.5 * emul_thickness
        side = str(gt["emulsion_side"])
        if side == "inner":
            self.emulsion_offset_signed = float(offset_mag)
        elif side == "outer":
            self.emulsion_offset_signed = -float(offset_mag)
        else:
            self.emulsion_offset_signed = 0.0
        print(f"  Emulsion offset baked into analytical: "
              f"{self.emulsion_offset_signed:+.2f}px (side={side}, "
              f"emul_thickness={emul_thickness:.2f}px)")
        self._compute_arc_parameters()

        if geometry_only:
            return

        # Emulsion-pixel sampler — z replicated across all loaded slices.
        emul_mask = (seg_2d == 2)
        ys_e, xs_e = np.where(emul_mask)
        u_at_pix = u_map_2d[ys_e, xs_e]
        keep = ~np.isnan(u_at_pix)
        ys_e, xs_e, u_at_pix = ys_e[keep], xs_e[keep], u_at_pix[keep]

        # Scale GT u ∈ [0, 1]  →  u_raw ∈ [0, n_layers) (detected).
        # Supervision is internally consistent with the INR's own scale;
        # whether detection == GT n_windings is orthogonal (Round 2 topic).
        u_raw = (u_at_pix * self.n_layers).astype(np.float32)
        u_raw = np.clip(u_raw, 0.0, self.n_layers - 1e-4)

        # Replicate across every loaded z-slice (emulsion geometry is z-invariant
        # in the current synthetic generator).
        n_per_z = len(ys_e)
        n_total = n_per_z * self.Z
        idx_z = np.repeat(np.arange(self.Z, dtype=np.int64), n_per_z)
        idx_y = np.tile(ys_e.astype(np.int64), self.Z)
        idx_x = np.tile(xs_e.astype(np.int64), self.Z)
        u_raw_all = np.tile(u_raw, self.Z)

        print(f"  GT supervised pool: {n_total:,} emulsion samples "
              f"({n_per_z:,}/slice × {self.Z} slices), n_windings_gt={self.gt_n_windings}")

        self.gt_zyx = torch.from_numpy(
            np.stack([idx_z, idx_y, idx_x], axis=1)
        ).to(self.device)
        self.gt_u_raw = torch.from_numpy(u_raw_all).to(self.device)
        # Precompute normalized (x, y) targets in [-1, 1] for MSE against INR.
        xy_norm = np.stack([
            2.0 * idx_x / (self.W - 1) - 1.0,
            2.0 * idx_y / (self.H - 1) - 1.0,
        ], axis=1).astype(np.float32)
        self.gt_xy_norm = torch.from_numpy(xy_norm).to(self.device)

    def sample_supervised(self, batch_size):
        """Sample (u_raw, z, xy_target) for supervised MSE on GT u_map.

        Returns dict:
            uv_norm:   (B, 2) in [-1, 1]
            u_raw:     (B,)   float in [0, n_layers)
            z_idx:     (B,)   int64 in [0, Z)
            xy_target: (B, 2) ground-truth (x, y) in [-1, 1]
        """
        assert hasattr(self, "gt_zyx"), (
            "load_synthetic_gt() must be called before sample_supervised()"
        )
        n = self.gt_zyx.shape[0]
        idx = torch.randint(0, n, (batch_size,), device=self.device)
        zyx = self.gt_zyx[idx]
        u_raw = self.gt_u_raw[idx]
        xy_target = self.gt_xy_norm[idx]
        z_idx = zyx[:, 0]

        u_norm = u_raw / self.n_layers * 2.0 - 1.0
        if self.Z > 1:
            z_norm = z_idx.float() / (self.Z - 1) * 2.0 - 1.0
        else:
            z_norm = torch.zeros(batch_size, device=self.device)
        uv_norm = torch.stack([u_norm, z_norm], dim=1)

        return {
            "uv_norm": uv_norm,
            "u_raw": u_raw,
            "z_idx": z_idx,
            "xy_target": xy_target,
        }
