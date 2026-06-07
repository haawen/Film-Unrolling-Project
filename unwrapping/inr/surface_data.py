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

import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import (
    gaussian_filter1d, binary_erosion, distance_transform_edt, label as cc_label,
)
from scipy.signal import find_peaks

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.center_detection import find_spool_center
from unwrapping.inr.geodesic_unwrap import _detect_seam_angle
from unwrapping.inr.unwrap_data_3d import discover_volumes, load_volume_chunk


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

    def __init__(self, data_dir, max_slices=None, device="cuda", diag_dir=None,
                 attachment="film", centerline_erode=1,
                 winding_detector="raycast", data_driven_base=False):
        """
        Args:
            attachment: which binary mask drives the attachment loss.
                "film"       — seg > 0 (class 1 + 2), 17px slab, original behavior
                "emulsion"   — seg == 2, thin band (~6px synthetic, ~3px real)
                "centerline" — emulsion eroded by `centerline_erode` iterations;
                               falls back to emulsion per-slice if erosion empties
            centerline_erode: erosion iterations for "centerline" mode (default 1)
            winding_detector: "raycast" (default, robust to pinch points and
                touching windings) or "histogram" (legacy, may miss windings
                when air gaps disappear).
            data_driven_base: if True, the analytical base uses per-angle
                centerline radii from raycast detection (Track 1) instead of
                concentric circles. Each winding becomes its own non-circular
                closed curve, absorbing eccentricity and a fraction of jitter
                into the base. Requires winding_detector="raycast".
        """
        assert attachment in ("film", "emulsion", "centerline"), attachment
        assert winding_detector in ("raycast", "histogram"), winding_detector
        if data_driven_base and winding_detector != "raycast":
            raise ValueError("data_driven_base=True requires winding_detector='raycast'")
        self.attachment_mode = attachment
        self.centerline_erode = centerline_erode
        self.winding_detector = winding_detector
        self.data_driven_base = data_driven_base
        self.per_angle_centerlines = None  # set below if data-driven
        self.device = torch.device(device)

        pairs = discover_volumes(data_dir)
        if not pairs:
            raise ValueError(f"No volume files found in {data_dir}")

        print(f"  Loading {len(pairs)} volume chunks...")

        vols, segs, z_indices = [], [], []
        slices_done = 0
        for vol_path, probs_path, z_start, z_end in pairs:
            # Partial HDF5 read when max_slices is small: avoids OOM on
            # large hi-res presets (e.g. 256×4K reads ~50 GB unconstrained).
            remaining = None if max_slices is None else max(0, max_slices - slices_done)
            volume, seg = load_volume_chunk(vol_path, probs_path, max_slices=remaining)
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

        if self.winding_detector == "raycast":
            result = detect_winding_boundaries_raycast(
                ref_seg, (cy, cx), out_dir=diag_dir,
                return_per_angle=True,
            )
            boundaries_np, n_layers, per_angle_clean = result
            print(f"  Raycast detector: {n_layers} windings")
            # Always persist per-angle centerlines for Z1 (MAPS) soft
            # supervision — they're cheap to keep, only used if --w-maps > 0.
            # data_driven_base controls whether the analytical *base* looks
            # them up (closed in R9 Track 1); MAPS uses them as soft targets.
            self._per_angle_np = per_angle_clean.astype(np.float32)
            if self.data_driven_base:
                print(f"  Data-driven base: per-angle centerlines "
                      f"shape={per_angle_clean.shape}, "
                      f"r range=[{per_angle_clean.min():.1f}, "
                      f"{per_angle_clean.max():.1f}]")
        else:
            boundaries_np, n_layers = detect_winding_boundaries_normalized(
                ref_film_mask, (cy, cx), out_dir=diag_dir,
            )
            print(f"  Histogram detector: {n_layers} windings")
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

        # Track 1: GPU tensor of per-angle centerlines, used by the data-driven
        # analytical base. Indexed [winding_k, ray_idx] in pixel radii.
        if self.data_driven_base and hasattr(self, "_per_angle_np"):
            self.per_angle_centerlines = torch.from_numpy(
                self._per_angle_np
            ).to(self.device)
            self.n_rays = int(self.per_angle_centerlines.shape[1])

        # Precompute per-winding delta for vectorized analytical lookup
        # segment k spans u ∈ [k, k+1), r goes from boundaries[k] → boundaries[k+1]
        self.boundary_deltas = self.boundaries[1:] - self.boundaries[:-1]   # (n_layers,)
        # Signed radial offset from film centerline to attachment-target centerline.
        # 0 = land on film centerline (default). For synthetic GT this is set in
        # load_synthetic_gt to land directly on the emulsion centerline, so the
        # INR residual only needs to fix small geometric perturbations rather
        # than re-learn a 15px direction-varying offset for every winding.
        self.emulsion_offset_signed = 0.0
        # Learnable eccentricity correction (R2b): r += amp * cos(theta - phase).
        # Init to zero, kept non-learnable until `enable_learnable_eccentricity`
        # is called by the trainer. Directly targets the ecc term the synthetic
        # generator adds (`r + ecc * cos(theta - pi/4)`).
        self.ecc_amp = torch.zeros((), device=self.device, requires_grad=False)
        self.ecc_phase = torch.zeros((), device=self.device, requires_grad=False)
        self._ecc_learnable = False
        # Per-winding (R3 lever 2): one (amp, phase) per detected winding,
        # off by default; turned on via `enable_per_winding_eccentricity`.
        self.pw_amp = torch.zeros((1,), device=self.device, requires_grad=False)
        self.pw_phase = torch.zeros((1,), device=self.device, requires_grad=False)
        self._pw_ecc_learnable = False
        # A4 (per-winding angular phase): one θ-offset per winding, added to
        # `theta` inside `analytical_xy_norm`. Off by default; enable via
        # `enable_per_winding_theta_phase`. Targets global+per-winding angular
        # misalignment that the INR residual learns slowly through σ=20 Fourier
        # features. Linear interp between adjacent winding scalars; one DoF
        # per winding (~28 scalars on HQ presets).
        self.theta_offset = torch.zeros((1,), device=self.device, requires_grad=False)
        self._theta_offset_learnable = False
        # CC-based winding-label volume — built lazily on first call to
        # `build_winding_label_volume`. Used for the topology-aware winding
        # penalty (R3 lever 3-alt): each emulsion pixel carries its true
        # winding number (CC on eroded film, ordered by mean radius), then
        # propagated to all pixels via nearest-emulsion EDT.
        self.winding_label_volume = None
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

    def sample_z_pairs(self, batch_size):
        """Sample (u, z) and (u, z+1) for z-coherence: same u, adjacent z.

        Returns dict with:
            uv_a, uv_b: (B, 2) in [-1, 1] — same u, z and z+1
            u_raw:      (B,)   float in [0, n_layers)
        Caller computes pred_a = base(u) + INR(uv_a), pred_b similarly,
        then penalizes (pred_a - pred_b)². Requires Z >= 2.
        """
        if self.Z < 2:
            return None
        u_raw = torch.rand(batch_size, device=self.device) * self.n_layers
        z_a = torch.randint(0, self.Z - 1, (batch_size,), device=self.device)
        z_b = z_a + 1

        u_norm = u_raw / self.n_layers * 2.0 - 1.0
        z_a_norm = z_a.float() / (self.Z - 1) * 2.0 - 1.0
        z_b_norm = z_b.float() / (self.Z - 1) * 2.0 - 1.0
        uv_a = torch.stack([u_norm, z_a_norm], dim=1)
        uv_b = torch.stack([u_norm, z_b_norm], dim=1)
        return {"uv_a": uv_a, "uv_b": uv_b, "u_raw": u_raw}

    def enable_learnable_eccentricity(self):
        """Turn on the per-theta radial correction (amp·cos(theta-phase)).

        Returns the tensors that must be added to the trainer optimizer.
        Prior: caller is expected to L2-regularize `ecc_amp`.
        """
        self.ecc_amp = torch.zeros((), device=self.device, requires_grad=True)
        self.ecc_phase = torch.zeros((), device=self.device, requires_grad=True)
        self._ecc_learnable = True
        return [self.ecc_amp, self.ecc_phase]

    def enable_per_winding_eccentricity(self):
        """Per-winding radial correction: r_k += amp_k·cos(theta - phase_k).

        Targets the synthetic generator's per-winding sinusoidal jitter
        (sum of 3 components with per-winding phase shifts) — a single global
        (amp, phase) cannot match it. n_layers × 2 scalar params; expected
        amp magnitudes ≤ 5px (synthetic jitter amplitude).
        """
        self.pw_amp = torch.zeros(
            (self.n_layers,), device=self.device, requires_grad=True
        )
        self.pw_phase = torch.zeros(
            (self.n_layers,), device=self.device, requires_grad=True
        )
        self._pw_ecc_learnable = True
        return [self.pw_amp, self.pw_phase]

    def sample_maps_labels(self, batch_size):
        """Z1 — sample soft pseudo-supervision labels from the raycast
        detector's per-angle centerlines.

        Per-angle centerlines `_per_angle_np` are computed once on the
        reference slice (n_layers, n_rays) and give the emulsion-centerline
        radius at each (winding k, ray idx i). Each (k, i) pair maps to:
            theta_total = 2π·k + 2π·(i / n_rays)
            arc(θ_total) = r_inner·θ_total + 0.5·a·θ_total²
            u_raw       = arc / total_arc * n_layers
            (x, y)       = (cx + r·cos(theta_seam + 2π·(i/n_rays)),
                            cy + r·sin(theta_seam + 2π·(i/n_rays)))

        Synthetic geometry is z-invariant — the same (xy, u) labels apply at
        every z. Soft (low-weight) MSE supervision against these labels gives
        the model an angular signal at every (u, z), which pure SS lacks.
        Differs from R9 Track 1 (data-driven base, closed) in that the seam
        identity-switch is averaged across many labels in expectation rather
        than baked as a hard constraint.

        Returns: dict with uv_norm (B,2), u_raw (B,), z_idx (B,), xy_target (B,2).
        """
        if not hasattr(self, "_per_angle_np"):
            return None
        pa = self._per_angle_np  # (n_layers, n_rays)
        n_rays = pa.shape[1]
        n_layers = self.n_layers

        # Random (k, i, z) indices.
        k = torch.randint(0, n_layers, (batch_size,), device=self.device)
        i = torch.randint(0, n_rays, (batch_size,), device=self.device)
        z_idx = torch.randint(0, self.Z, (batch_size,), device=self.device)

        # Lookup centerline radius.
        pa_t = torch.from_numpy(pa).to(self.device) if not hasattr(self, "_pa_t") \
               else self._pa_t
        if not hasattr(self, "_pa_t"):
            self._pa_t = pa_t
        r_px = pa_t[k, i]  # (B,) — pixel radius

        # Angles.
        ray_frac = i.float() / n_rays
        theta_within = 2.0 * math.pi * ray_frac
        theta = self.theta_seam + theta_within
        theta_total = 2.0 * math.pi * k.float() + theta_within

        # Pixel (x, y).
        x_px = self.cx + r_px * torch.cos(theta)
        y_px = self.cy + r_px * torch.sin(theta)
        x_norm = 2.0 * x_px / (self.W - 1) - 1.0
        y_norm = 2.0 * y_px / (self.H - 1) - 1.0
        xy_target = torch.stack([x_norm, y_norm], dim=1)

        # u from arc-length.
        a = self.spiral_advance
        r0 = self.r_inner_centerline
        total = self.total_arc
        arc = r0 * theta_total + 0.5 * a * theta_total * theta_total
        u_raw = (arc / total * n_layers).clamp(0.0, n_layers - 1e-6)

        # Build uv_norm for INR.
        u_norm = u_raw / n_layers * 2.0 - 1.0
        if self.Z > 1:
            z_norm = z_idx.float() / (self.Z - 1) * 2.0 - 1.0
        else:
            z_norm = torch.zeros(batch_size, device=self.device)
        uv_norm = torch.stack([u_norm, z_norm], dim=1)

        return {"uv_norm": uv_norm, "u_raw": u_raw,
                "z_idx": z_idx, "xy_target": xy_target}

    def enable_per_winding_theta_phase(self):
        """A4: per-winding angular phase offset θ_offset[k] added to `theta`.

        Linear interp between adjacent winding scalars when u falls between
        integer winding indices. n_layers scalar params, init zero. Targets
        the residual median-shift that the INR fails to absorb quickly via
        global Fourier features. Expected magnitudes |θ_offset| < 0.05 rad
        (corresponds to a few px at outer windings); L2-regularize via
        --w-theta-phase-prior.
        """
        self.theta_offset = torch.zeros(
            (self.n_layers,), device=self.device, requires_grad=True
        )
        self._theta_offset_learnable = True
        return [self.theta_offset]

    def _compute_arc_parameters(self):
        """Precompute exact continuous-Archimedean parameters.

        Matches the synthetic generator exactly (see unwrapping/synthetic/generate.py):
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

    def _lookup_per_angle_radius(self, u_clamped, theta):
        """Bilinear lookup of per-angle centerline radii (Track 1).

        Args:
            u_clamped: (B,) in [0, n_layers - eps); fractional winding index.
            theta:     (B,) angle in radians (already includes theta_seam).

        Returns: (B,) radius in pixel units.

        Linear interpolation between adjacent windings (k, k+1), circular
        linear interpolation across `n_rays` angle bins.
        """
        pa = self.per_angle_centerlines  # (n_layers, n_rays) float32
        n_rays = self.n_rays
        # Angle → fractional bin index (0..n_rays).
        theta_mod = torch.remainder(theta, 2.0 * math.pi)
        ang_f = theta_mod / (2.0 * math.pi) * n_rays
        ang_lo = torch.floor(ang_f).long() % n_rays
        ang_hi = (ang_lo + 1) % n_rays
        ang_frac = ang_f - torch.floor(ang_f)

        k_lo = u_clamped.long().clamp(0, self.n_layers - 1)
        k_hi = (k_lo + 1).clamp(0, self.n_layers - 1)
        k_frac = u_clamped - k_lo.float()

        # Gather four corner radii.
        r00 = pa[k_lo, ang_lo]
        r01 = pa[k_lo, ang_hi]
        r10 = pa[k_hi, ang_lo]
        r11 = pa[k_hi, ang_hi]
        r0 = r00 * (1.0 - ang_frac) + r01 * ang_frac
        r1 = r10 * (1.0 - ang_frac) + r11 * ang_frac
        return r0 * (1.0 - k_frac) + r1 * k_frac

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
        if self._theta_offset_learnable:
            # A4: per-winding angular phase, linearly interpolated in u.
            k_lo = u_clamped.long().clamp(0, self.n_layers - 1)
            k_hi = (k_lo + 1).clamp(0, self.n_layers - 1)
            k_frac = u_clamped - k_lo.float()
            theta = theta + (
                self.theta_offset[k_lo] * (1.0 - k_frac)
                + self.theta_offset[k_hi] * k_frac
            )
        if self.per_angle_centerlines is not None:
            # Data-driven base (Track 1): per-winding non-circular curves.
            # Bilinear lookup in (k_real, theta_bin) with circular wrap on theta.
            # NOTE: the raycast detector returns emulsion-run centerline radii,
            # so the table is ALREADY on the emulsion. Do NOT apply
            # emulsion_offset_signed here (which is what the Archimedean base
            # uses to convert film-centerline → emulsion-centerline).
            r = self._lookup_per_angle_radius(u_clamped, theta)
        else:
            # Concentric Archimedean fallback (legacy).
            r = r0 + a * theta_total - self.emulsion_offset_signed
        if self._ecc_learnable:
            r = r + self.ecc_amp * torch.cos(theta - self.ecc_phase)
        if self._pw_ecc_learnable:
            # Index by floor(u_clamped) → which winding each sample lives in.
            k = u_clamped.long().clamp(0, self.n_layers - 1)
            r = r + self.pw_amp[k] * torch.cos(theta - self.pw_phase[k])
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

    def build_winding_label_volume(self, erode_iters=2):
        """Per-pixel winding-number map via CC on the eroded film mask.

        Each connected component (after a small erosion that breaks pinch
        points) is one winding. Components are ordered by mean radius and
        labeled 0..n_layers-1. The map is then propagated to ALL pixels via
        nearest-emulsion EDT so off-film queries get a sensible winding number
        too. Stored as a (1,1,Z,H,W) float volume on GPU; bilinear sampling
        gives a smooth gradient across air gaps (winding number ramps from
        k to k+1 across the gap), which is the disambiguation signal that
        plain mask attachment lacks.

        If CC finds more components than detected windings, the extras (small,
        spurious) are merged into the nearest larger component by radius.
        If CC finds fewer (pinch points survived erosion), pixels in the
        merged component still get a single winding label — the penalty
        will still distinguish "mostly correct winding" from "wildly wrong".
        """
        n_z, H, W = self.Z, self.H, self.W
        label_vol = np.zeros((n_z, H, W), dtype=np.float32)
        n_targets = self.n_layers
        for z in range(n_z):
            film = (self.seg_stack[z] > 0)
            if not film.any():
                continue
            eroded = binary_erosion(film, iterations=erode_iters)
            if not eroded.any():
                eroded = film
            cc, n_cc = cc_label(eroded)
            if n_cc == 0:
                continue
            # Order components by mean radius, keep top n_targets.
            comps = []
            for c in range(1, n_cc + 1):
                ys, xs = np.where(cc == c)
                if len(ys) < 8:
                    continue
                r = float(np.sqrt(
                    (ys - self.cy) ** 2 + (xs - self.cx) ** 2
                ).mean())
                comps.append((r, c, len(ys)))
            comps.sort(key=lambda t: t[0])
            # If we have more comps than expected windings, drop the smallest
            # by pixel count.
            if len(comps) > n_targets:
                comps_sorted_size = sorted(comps, key=lambda t: -t[2])
                keep_ids = set(t[1] for t in comps_sorted_size[:n_targets])
                comps = [c for c in comps if c[1] in keep_ids]
                comps.sort(key=lambda t: t[0])
            # Assign winding label by radial order.
            label_2d = np.full((H, W), -1, dtype=np.int32)
            for new_idx, (_, c, _) in enumerate(comps):
                label_2d[cc == c] = new_idx
            # Propagate -1 pixels to nearest labeled pixel via EDT.
            valid = label_2d >= 0
            if valid.any():
                _, nn_idx = distance_transform_edt(~valid, return_indices=True)
                label_2d = label_2d[nn_idx[0], nn_idx[1]]
            label_vol[z] = label_2d.astype(np.float32)
        # Cap at [0, n_layers-1] for safety.
        label_vol = np.clip(label_vol, 0, n_targets - 1)
        max_per_z = label_vol.reshape(n_z, -1).max(axis=1)
        print(f"  Winding-label volume built. erode={erode_iters}, "
              f"max winding/slice: min={int(max_per_z.min())}, "
              f"max={int(max_per_z.max())}, expected={n_targets-1}")
        self.winding_label_volume = torch.from_numpy(label_vol).to(
            self.device
        ).view(1, 1, n_z, H, W)

    def sample_winding_label(self, xy_norm, z_idx):
        """Bilinear-sample the per-pixel winding-label volume.

        Returns continuous winding numbers (gradient through grid_sample).
        Caller should compare to expected_winding=floor(u_raw) and penalize
        the squared difference.
        """
        assert self.winding_label_volume is not None, (
            "build_winding_label_volume() must be called first"
        )
        grid = self._build_grid_3d(xy_norm, z_idx)
        sampled = torch.nn.functional.grid_sample(
            self.winding_label_volume, grid, mode="bilinear",
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

        # Track 1: when data-driven base is active, per-angle centerlines were
        # detected from the actual (eccentric/jittered) segmentation. Snapping
        # geometry to a perfect GT spiral would invalidate them. Skip the
        # override; supervised pool (gt_zyx, gt_u_raw) is still built below.
        if self.data_driven_base:
            print(f"  GT loaded but geometry override SKIPPED "
                  f"(data_driven_base=True, n_layers={self.n_layers}, "
                  f"GT n={int(gt['n_windings'])})")
            # Per-angle table is already on emulsion → no offset to apply.
            self.emulsion_offset_signed = 0.0
            if geometry_only:
                return
            # Build supervised pool against detected n_layers.
            # Critical: GT u=0 corresponds to theta=0 in the synthetic
            # generator's frame, but the analytical's u=0 is at
            # theta=self.theta_seam. Shift GT u by theta_seam/(2π) windings
            # so MSE-supervised pred_xy lands at the right angle.
            emul_mask = (seg_2d == 2)
            ys_e, xs_e = np.where(emul_mask)
            u_at_pix = u_map_2d[ys_e, xs_e]
            keep = ~np.isnan(u_at_pix)
            ys_e, xs_e, u_at_pix = ys_e[keep], xs_e[keep], u_at_pix[keep]
            u_shift = self.theta_seam / (2.0 * math.pi)
            u_raw = (u_at_pix * self.n_layers - u_shift).astype(np.float32)
            u_raw = np.mod(u_raw, self.n_layers)
            u_raw = np.clip(u_raw, 0.0, self.n_layers - 1e-4)
            print(f"  GT u shifted by -{u_shift:.4f} windings "
                  f"(theta_seam={math.degrees(self.theta_seam):.1f}°)")
            n_per_z = len(ys_e)
            n_total = n_per_z * self.Z
            idx_z = np.repeat(np.arange(self.Z, dtype=np.int64), n_per_z)
            idx_y = np.tile(ys_e.astype(np.int64), self.Z)
            idx_x = np.tile(xs_e.astype(np.int64), self.Z)
            u_raw_all = np.tile(u_raw, self.Z)
            print(f"  GT supervised pool: {n_total:,} samples "
                  f"({n_per_z:,}/slice × {self.Z} slices)")
            self.gt_zyx = torch.from_numpy(
                np.stack([idx_z, idx_y, idx_x], axis=1)
            ).to(self.device)
            self.gt_u_raw = torch.from_numpy(u_raw_all).to(self.device)
            xy_norm = np.stack([
                2.0 * idx_x / (self.W - 1) - 1.0,
                2.0 * idx_y / (self.H - 1) - 1.0,
            ], axis=1).astype(np.float32)
            self.gt_xy_norm = torch.from_numpy(xy_norm).to(self.device)
            return

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
