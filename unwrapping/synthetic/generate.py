"""
Synthetic Spiral Film Roll Dataset Generator
=============================================
Creates synthetic CT-like cross-sections of wound film rolls by rolling
a 2D strip image into an Archimedean spiral.  Produces HDF5 files compatible
with the existing unwrapping pipeline, plus ground truth for evaluation.

The spiral is modelled as concentric ring-like windings (matching real CT
cross-sections).  Each winding carries a thin emulsion layer whose intensity
comes from the source strip, and a uniform film-base layer.

Usage:
    python -m unwrapping.synthetic.generate --preset quick_test
    python -m unwrapping.synthetic.generate --preset clean_4k --output-dir results/synthetic
"""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SpiralParams:
    """Parameters defining the spiral geometry and rendering."""

    image_size: int = 512           # output image H=W (square)
    n_windings: int = 15            # number of spiral windings
    film_thickness: float = 12.0    # total radial thickness of film (px)
    emulsion_fraction: float = 0.18 # fraction of film that is emulsion
    air_gap: float = 4.0            # radial air gap between windings (px)
    n_z_slices: int = 5             # depth slices in the volume
    pattern: str = "frames"         # strip pattern type

    # emulsion placement
    emulsion_side: str = "inner"    # "inner" or "outer" radial edge

    # imperfection controls
    noise_std: float = 0.0          # additive Gaussian noise sigma
    eccentricity: float = 0.0       # spiral center offset (px)
    radial_jitter: float = 0.0      # per-winding radial perturbation (px)

    # rendering intensities  (model real CT absorption)
    #   air         ~ 0.02  (very low absorption)
    #   film base   ~ 0.12  (dark plastic / cellulose)
    #   emulsion @0 ~ 0.12  (unexposed = same as base, no silver)
    #   emulsion @1 ~ 0.85  (fully exposed = dense silver grains, bright in CT)
    background_intensity: float = 0.02
    film_base_intensity: float = 0.12
    emulsion_max_intensity: float = 0.85

    center: Optional[Tuple[float, float]] = None  # (cy, cx); None = image center

    # ── derived ──────────────────────────────────────────────────────────
    @property
    def layer_spacing(self) -> float:
        return self.film_thickness + self.air_gap

    @property
    def emulsion_thickness(self) -> float:
        return self.film_thickness * self.emulsion_fraction

    @property
    def r_inner(self) -> float:
        """Innermost winding centerline radius (leaves room for spool core)."""
        return self.layer_spacing * 3

    @property
    def r_outer(self) -> float:
        return self.r_inner + self.layer_spacing * (self.n_windings - 1)

    @property
    def spiral_advance(self) -> float:
        """Radial advance per radian: a in  r = r0 + a*theta."""
        return self.layer_spacing / (2 * np.pi)

    def get_center(self) -> Tuple[float, float]:
        if self.center is None:
            c = self.image_size / 2.0
            return (c, c)
        return self.center

    def compute_strip_length(self) -> int:
        """Approximate total spiral arc-length in pixels."""
        a = self.spiral_advance
        theta_max = 2 * np.pi * self.n_windings
        return int(self.r_inner * theta_max + 0.5 * a * theta_max ** 2)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in [
            "image_size", "n_windings", "film_thickness", "emulsion_fraction",
            "air_gap", "n_z_slices", "pattern", "emulsion_side",
            "noise_std", "eccentricity", "radial_jitter",
            "background_intensity", "film_base_intensity",
            "emulsion_max_intensity",
        ]}


# ═══════════════════════════════════════════════════════════════════════════
# Presets
# ═══════════════════════════════════════════════════════════════════════════

PRESETS: Dict[str, SpiralParams] = {
    # Phase 1 — quick verification (small, clean)
    "quick_test": SpiralParams(
        image_size=512, n_windings=15, film_thickness=12, air_gap=4,
        n_z_slices=5, pattern="frames",
    ),
    # Phase 2 — 4K variants
    "clean_4k": SpiralParams(
        image_size=4096, n_windings=28, film_thickness=36, air_gap=12,
        n_z_slices=20, pattern="frames",
    ),
    "noisy_4k": SpiralParams(
        image_size=4096, n_windings=28, film_thickness=36, air_gap=12,
        n_z_slices=20, noise_std=0.05, pattern="frames",
    ),
    "tight_4k": SpiralParams(
        image_size=4096, n_windings=28, film_thickness=36, air_gap=3,
        n_z_slices=20, pattern="frames",
    ),
    "imperfect_4k": SpiralParams(
        image_size=4096, n_windings=28, film_thickness=36, air_gap=12,
        n_z_slices=20, noise_std=0.03, eccentricity=10.0, radial_jitter=5.0,
        pattern="frames",
    ),
    # Phase 2 — outer-edge emulsion variants
    "clean_4k_outer": SpiralParams(
        image_size=4096, n_windings=28, film_thickness=36, air_gap=12,
        n_z_slices=20, pattern="frames", emulsion_side="outer",
    ),
    "imperfect_4k_outer": SpiralParams(
        image_size=4096, n_windings=28, film_thickness=36, air_gap=12,
        n_z_slices=20, noise_std=0.03, eccentricity=10.0, radial_jitter=5.0,
        pattern="frames", emulsion_side="outer",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Strip Pattern Generators
# ═══════════════════════════════════════════════════════════════════════════

def generate_strip(width: int, height: int, pattern: str = "frames") -> np.ndarray:
    """Generate a 2D test-pattern strip  (height, width)  in [0, 1]."""
    fn = {
        "gradient": _gradient_strip,
        "bars": _bars_strip,
        "frames": _frames_strip,
        "composite": _composite_strip,
    }
    if pattern not in fn:
        raise ValueError(f"Unknown pattern '{pattern}'. Choose from {list(fn)}")
    return fn[pattern](width, height)


def _gradient_strip(w: int, h: int) -> np.ndarray:
    """Smooth horizontal gradient — value == normalised position."""
    return np.tile(
        np.linspace(0.1, 0.9, w, dtype=np.float32)[np.newaxis, :], (h, 1)
    )


def _bars_strip(w: int, h: int) -> np.ndarray:
    """Alternating vertical bars."""
    bar_w = max(w // 40, 10)
    x = np.arange(w)
    row = ((x // bar_w) % 2).astype(np.float32) * 0.6 + 0.2
    return np.tile(row[np.newaxis, :], (h, 1))


def _frames_strip(w: int, h: int) -> np.ndarray:
    """Simulated movie frames with per-frame unique patterns."""
    strip = np.full((h, w), 0.15, dtype=np.float32)
    frame_w = max(200, h * 4 // 3)  # wide enough to be recognisable
    n_frames = w // frame_w + 1

    yy_full, xx_full = np.mgrid[0:h, 0:w]

    for i in range(n_frames):
        x0 = i * frame_w
        x1 = min(x0 + frame_w, w)
        if x0 >= w:
            break
        lx = xx_full[:, x0:x1] - x0
        ly = yy_full[:, x0:x1]
        fw = x1 - x0

        # unique diagonal stripes
        angle = i * 0.4 + 0.2
        period = max(h // 3, 8)
        wave = np.sin(
            2 * np.pi * (lx * np.cos(angle) + ly * np.sin(angle)) / period
        )
        base = 0.3 + 0.4 * ((i * 7) % 11) / 10.0
        content = base + 0.15 * wave

        # circle in each frame (simulates a character / object)
        cx_f, cy_f = fw // 2, h // 2
        r_circle = min(fw, h) * 0.22
        dist = np.sqrt((lx - cx_f) ** 2 + (ly - cy_f) ** 2)
        content = np.where(dist < r_circle, content + 0.15, content)

        # thin frame border
        bw = max(1, h // 20)
        content[:bw, :] = 0.10
        content[-bw:, :] = 0.10
        content[:, :min(bw, fw)] = 0.10

        strip[:, x0:x1] = np.clip(content, 0.05, 0.95)

    # perforation holes along top / bottom
    perf_sp = frame_w // 2
    perf_sz = max(h // 8, 2)
    for px in range(perf_sz, w, perf_sp):
        p0 = max(px - perf_sz // 2, 0)
        p1 = min(px + perf_sz // 2, w)
        strip[:perf_sz, p0:p1] = 0.05
        strip[-perf_sz:, p0:p1] = 0.05

    return strip


def _composite_strip(w: int, h: int) -> np.ndarray:
    """Frames blended with a positional gradient."""
    return np.clip(
        0.7 * _frames_strip(w, h) + 0.3 * _gradient_strip(w, h), 0, 1
    )


# ═══════════════════════════════════════════════════════════════════════════
# Spiral Geometry  (computed once, reused for every z-slice)
# ═══════════════════════════════════════════════════════════════════════════

def compute_spiral_geometry(params: SpiralParams, strip_width: int) -> dict:
    """Pre-compute all geometry arrays shared across z-slices.

    Returns a dict with boolean masks, segmentation template, u-map template,
    and interpolation indices/weights into the source strip.
    """
    H = W = params.image_size
    cy, cx = params.get_center()
    a = params.spiral_advance

    # coordinate grids (float32 to save memory at 4K)
    yy, xx = np.mgrid[0:H, 0:W]
    yy = yy.astype(np.float32)
    xx = xx.astype(np.float32)

    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    theta = np.arctan2(yy - cy, xx - cx) % (2 * np.pi)

    # ── eccentricity: shift effective radius (≈ off-centre spool) ─────
    if params.eccentricity > 0:
        r = r + params.eccentricity * np.cos(theta - np.pi / 4)

    # ── winding assignment ────────────────────────────────────────────
    k_cont = (r - params.r_inner - a * theta) / params.layer_spacing
    k = np.rint(k_cont).astype(np.int32)

    # centerline radius for nearest winding
    r_center = params.r_inner + a * (theta + 2 * np.pi * k)

    # ── per-winding radial jitter ─────────────────────────────────────
    if params.radial_jitter > 0:
        jitter = params.radial_jitter * (
            0.5 * np.sin(theta * 2.0 + k * 0.7)
            + 0.3 * np.sin(theta * 5.0 + k * 1.3)
            + 0.2 * np.sin(theta * 11.0 + k * 2.1)
        )
        r_center = r_center + jitter

    # signed radial distance from winding centerline
    dr = r - r_center
    half_t = params.film_thickness / 2.0

    # masks
    on_film = (np.abs(dr) <= half_t) & (k >= 0) & (k < params.n_windings)
    emul_t = params.emulsion_thickness
    if params.emulsion_side == "outer":
        # emulsion on the outer radial edge (larger r)
        on_emulsion = on_film & (dr <= half_t) & (dr > half_t - emul_t)
    else:
        # emulsion on the inner radial edge (smaller r)
        on_emulsion = on_film & (dr >= -half_t) & (dr < -half_t + emul_t)
    on_base = on_film & ~on_emulsion

    # segmentation template (same for every z)
    seg = np.zeros((H, W), dtype=np.uint8)
    seg[on_base] = 1
    seg[on_emulsion] = 2

    # ── arc-length → strip u-coordinate ───────────────────────────────
    theta_total = theta + 2 * np.pi * k.astype(np.float32)
    arc = params.r_inner * theta_total + 0.5 * a * theta_total ** 2
    theta_max = 2 * np.pi * params.n_windings
    total_arc = params.r_inner * theta_max + 0.5 * a * theta_max ** 2
    u_norm = np.clip(arc / total_arc, 0, 1)  # normalised [0,1]

    u_map = np.full((H, W), np.nan, dtype=np.float32)
    u_map[on_film] = u_norm[on_film]

    # interpolation indices into strip
    u_pixel = u_norm * (strip_width - 1)
    u_pixel = np.clip(u_pixel, 0, strip_width - 1)
    u_floor = np.floor(u_pixel).astype(np.int32)
    u_ceil = np.minimum(u_floor + 1, strip_width - 1)
    u_frac = (u_pixel - u_floor).astype(np.float32)

    # free large intermediates
    del yy, xx, r, theta, k_cont, k, r_center, dr, theta_total, arc, u_norm, u_pixel

    return dict(
        seg=seg, u_map=u_map,
        on_film=on_film, on_base=on_base, on_emulsion=on_emulsion,
        u_floor=u_floor, u_ceil=u_ceil, u_frac=u_frac,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Slice Rendering
# ═══════════════════════════════════════════════════════════════════════════

def render_slice(
    strip_row: np.ndarray,
    geo: dict,
    params: SpiralParams,
) -> np.ndarray:
    """Render one z-slice from a single strip row + pre-computed geometry.

    Returns (H, W) float32 image in [0, 1].
    """
    H = W = params.image_size
    base = params.film_base_intensity
    image = np.full((H, W), params.background_intensity, dtype=np.float32)
    image[geo["on_base"]] = base

    # bilinear interpolation of strip content [0,1]
    strip_val = (
        strip_row[geo["u_floor"]] * (1.0 - geo["u_frac"])
        + strip_row[geo["u_ceil"]] * geo["u_frac"]
    )
    # map strip content to CT absorption:
    #   strip=0 (black/unexposed) -> same as film base (dark, no silver)
    #   strip=1 (white/exposed)   -> emulsion_max (bright, dense silver)
    emul_intensity = base + strip_val * (params.emulsion_max_intensity - base)
    image[geo["on_emulsion"]] = emul_intensity[geo["on_emulsion"]]

    if params.noise_std > 0:
        noise = np.random.normal(0, params.noise_std, (H, W)).astype(np.float32)
        image = np.clip(image + noise, 0, 1)

    return image


# ═══════════════════════════════════════════════════════════════════════════
# Save helpers
# ═══════════════════════════════════════════════════════════════════════════

def _seg_to_onehot(seg: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Convert integer seg-map to one-hot (…, n_classes) float32."""
    oh = np.zeros((*seg.shape, n_classes), dtype=np.float32)
    for c in range(n_classes):
        oh[..., c] = (seg == c).astype(np.float32)
    return oh


def save_dataset(
    strip: np.ndarray,
    geo: dict,
    params: SpiralParams,
    output_dir: Path,
):
    """Render every z-slice and write HDF5 files + ground truth."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_z = params.n_z_slices
    H = W = params.image_size
    seg = geo["seg"]
    seg_oh_2d = _seg_to_onehot(seg)  # reuse for every slice

    # ── 3D volume files (written slice-by-slice to save memory) ───────
    vol_path = output_dir / f"volume_0000-{n_z - 1:04d}.h5"
    vol_prob_path = output_dir / f"volume_0000-{n_z - 1:04d}_Probabilities.h5"

    f_vol = h5py.File(vol_path, "w")
    f_prob = h5py.File(vol_prob_path, "w")
    dset_vol = f_vol.create_dataset("volume", shape=(n_z, H, W), dtype=np.uint16)
    dset_prob = f_prob.create_dataset(
        "exported_data", shape=(n_z, H, W, 3), dtype=np.float32,
        compression="gzip", compression_opts=1,
    )

    first_image = None

    for z in range(n_z):
        image = render_slice(strip[z], geo, params)
        img_u16 = (np.clip(image, 0, 1) * 65535).astype(np.uint16)

        # 2D slice files
        with h5py.File(output_dir / f"slice_{z:04d}.h5", "w") as f:
            f.create_dataset("image", data=img_u16)
        with h5py.File(output_dir / f"slice_{z:04d}_Probabilities.h5", "w") as f:
            f.create_dataset("exported_data", data=seg_oh_2d)

        # 3D volume
        dset_vol[z] = img_u16
        dset_prob[z] = seg_oh_2d

        if z == 0:
            first_image = image.copy()
        if (z + 1) % 5 == 0 or z == n_z - 1:
            print(f"    rendered slice {z + 1}/{n_z}")

    f_vol.close()
    f_prob.close()

    # ── ground truth ──────────────────────────────────────────────────
    gt_path = output_dir / "ground_truth.npz"
    np.savez_compressed(
        gt_path,
        strip=strip,
        u_map=geo["u_map"],
        seg=seg,
        **params.to_dict(),
    )

    # ── params JSON (human-readable) ─────────────────────────────────
    with open(output_dir / "params.json", "w") as f:
        json.dump(params.to_dict(), f, indent=2)

    print(f"  Saved {n_z} 2D slices  +  1 3D volume  ->  {output_dir}")
    return first_image


# ═══════════════════════════════════════════════════════════════════════════
# Verification
# ═══════════════════════════════════════════════════════════════════════════

def verify_roundtrip(
    image: np.ndarray,
    seg: np.ndarray,
    u_map: np.ndarray,
    strip_row: np.ndarray,
    params: SpiralParams,
    output_dir: Path,
):
    """Unroll using ground-truth u-map, compare to original strip row.

    Only uses emulsion pixels (they carry the varying film content).
    Inverts the CT intensity model (base + content*(max-base)) to recover
    the original strip content before comparison.
    Returns RMSE of the recovered strip vs original.
    """
    output_dir = Path(output_dir)
    emul_mask = seg == 2
    u_vals = u_map[emul_mask]
    raw_intensities = image[emul_mask]

    # invert CT intensity model: content = (intensity - base) / (max - base)
    base = params.film_base_intensity
    span = params.emulsion_max_intensity - base
    content = (raw_intensities.astype(np.float64) - base) / span

    n_bins = len(strip_row)
    recovered = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int64)

    u_bins = np.clip((u_vals * (n_bins - 1)).astype(np.int64), 0, n_bins - 1)
    np.add.at(recovered, u_bins, content)
    np.add.at(counts, u_bins, 1)

    valid = counts > 0
    recovered[valid] /= counts[valid]

    # metrics (compare recovered content to original strip content)
    both = valid & (strip_row > 0)
    if both.sum() == 0:
        print("  roundtrip: no overlapping pixels!")
        return np.inf

    diff = recovered[both] - strip_row[both].astype(np.float64)
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_err = float(np.max(np.abs(diff)))
    coverage = valid.sum() / n_bins

    print(f"  roundtrip  RMSE={rmse:.5f}  max_err={max_err:.5f}  "
          f"coverage={coverage:.1%}  ({valid.sum()}/{n_bins} bins)")

    # ── plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(14, 6), sharex=True)

    # show a small window for visibility
    show_n = min(2000, n_bins)
    xs = np.arange(show_n)
    axes[0].plot(xs, strip_row[:show_n], "k-", lw=0.5, label="original")
    axes[0].set_ylabel("content")
    axes[0].set_title("Original strip content (first 2000 px)")
    axes[0].legend(fontsize=8)

    axes[1].plot(xs, recovered[:show_n], "b-", lw=0.5, label="recovered")
    axes[1].set_ylabel("content")
    axes[1].set_title(f"Recovered strip content (RMSE={rmse:.5f})")
    axes[1].legend(fontsize=8)

    err_plot = recovered[:show_n] - strip_row[:show_n].astype(np.float64)
    axes[2].plot(xs, err_plot, "r-", lw=0.5)
    axes[2].set_ylabel("error")
    axes[2].set_xlabel("strip position (px)")
    axes[2].set_title("Difference")

    plt.tight_layout()
    plt.savefig(output_dir / "roundtrip_verify.png", dpi=120, bbox_inches="tight")
    plt.close()
    return rmse


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic Plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_diagnostics(
    image: np.ndarray,
    seg: np.ndarray,
    u_map: np.ndarray,
    strip: np.ndarray,
    params: SpiralParams,
    output_dir: Path,
):
    """Save overview and zoomed diagnostic figures."""
    output_dir = Path(output_dir)

    # ── overview ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(strip, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[0, 0].set_title(f"Strip  {strip.shape[0]}x{strip.shape[1]}")
    axes[0, 0].set_xlabel("arc-length (px)")
    axes[0, 0].set_ylabel("z-slice")

    axes[0, 1].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Rolled image (slice 0)")

    seg_rgb = np.zeros((*seg.shape, 3), dtype=np.float32)
    seg_rgb[seg == 0] = [0.10, 0.10, 0.20]
    seg_rgb[seg == 1] = [0.20, 0.70, 0.20]
    seg_rgb[seg == 2] = [0.90, 0.30, 0.10]
    axes[0, 2].imshow(seg_rgb)
    axes[0, 2].set_title("Segmentation  (air / base / emulsion)")

    im = axes[1, 0].imshow(u_map, cmap="turbo", vmin=0, vmax=1)
    axes[1, 0].set_title("Ground-truth u-map")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)

    cy = int(params.get_center()[0])
    axes[1, 1].plot(image[cy, :], "k-", lw=0.4)
    axes[1, 1].set_title(f"Radial profile  (y={cy})")
    axes[1, 1].set_xlabel("x (px)")
    axes[1, 1].set_ylabel("intensity")

    u_valid = u_map[~np.isnan(u_map)]
    axes[1, 2].hist(u_valid, bins=100, color="steelblue", edgecolor="none")
    axes[1, 2].set_title("u-map histogram")
    axes[1, 2].set_xlabel("u")

    plt.suptitle(
        f"Synthetic -- {params.pattern}  |  {params.n_windings} windings  |  "
        f"{params.image_size}x{params.image_size}  |  emulsion={params.emulsion_side}  |  "
        f"noise={params.noise_std}  ecc={params.eccentricity}  jit={params.radial_jitter}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── zoomed view (3-4 windings) ────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    zoom_r = int(params.layer_spacing * 4)
    y0 = max(cy - zoom_r, 0)
    y1 = min(cy + zoom_r, params.image_size)
    x0 = cy  # start from centre going right
    x1 = min(cy + zoom_r * 2, params.image_size)

    axes[0].imshow(image[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Zoomed: intensity")
    axes[1].imshow(seg_rgb[y0:y1, x0:x1])
    axes[1].set_title("Zoomed: segmentation")
    im = axes[2].imshow(u_map[y0:y1, x0:x1], cmap="turbo")
    axes[2].set_title("Zoomed: u-map")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.suptitle("Zoomed view  (3-4 windings)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_dir / "diagnostics_zoom.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── segmentation class counts ─────────────────────────────────────
    n_total = params.image_size ** 2
    n_air = int((seg == 0).sum())
    n_base = int((seg == 1).sum())
    n_emul = int((seg == 2).sum())
    print(f"  seg classes -- air: {n_air/n_total:.1%}  "
          f"base: {n_base/n_total:.1%}  emulsion: {n_emul/n_total:.1%}")


def plot_strip(strip: np.ndarray, params: SpiralParams, output_dir: Path):
    """Save the original unrolled strip with zoomed crops."""
    output_dir = Path(output_dir)
    h, w = strip.shape
    n_crops = 4
    crop_w = min(800, w // 4)

    fig_h = max(3, 1.5 * (1 + n_crops))
    fig, axes = plt.subplots(1 + n_crops, 1, figsize=(16, fig_h))

    # full strip (compressed)
    axes[0].imshow(strip, cmap="gray", aspect="auto", vmin=0, vmax=1)
    axes[0].set_title(f"Full strip  ({h} x {w} px)")
    axes[0].set_xlabel("arc-length (px)")
    axes[0].set_ylabel("z")

    # zoomed crops at evenly spaced positions
    rng = np.random.RandomState(42)
    positions = np.linspace(0, w - crop_w, n_crops + 2)[1:-1].astype(int)
    # add some randomness
    jitter = rng.randint(-crop_w // 4, crop_w // 4, size=n_crops)
    positions = np.clip(positions + jitter, 0, w - crop_w)

    for i, x0 in enumerate(positions):
        axes[i + 1].imshow(
            strip[:, x0 : x0 + crop_w], cmap="gray", aspect="auto", vmin=0, vmax=1
        )
        pct = x0 / w * 100
        axes[i + 1].set_title(f"Crop @ x={x0}  ({pct:.0f}% along strip)")
        axes[i + 1].set_ylabel("z")

    plt.suptitle(
        f"Original film strip  --  {params.pattern} pattern", fontsize=13
    )
    plt.tight_layout()
    plt.savefig(output_dir / "strip.png", dpi=150, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

def generate_dataset(params: SpiralParams, output_dir: str) -> dict:
    """End-to-end: generate strip, compute geometry, render, save, verify."""
    output_dir = Path(output_dir)
    strip_w = params.compute_strip_length()

    print(f"\n{'=' * 60}")
    print(f"Generating synthetic dataset")
    print(f"  preset params : {params.to_dict()}")
    print(f"  strip size    : {params.n_z_slices} x {strip_w}")
    print(f"  output        : {output_dir}")
    print(f"{'=' * 60}")

    t0 = time.time()
    strip = generate_strip(strip_w, params.n_z_slices, params.pattern)
    print(f"  strip generated     {time.time() - t0:.1f}s")

    t1 = time.time()
    geo = compute_spiral_geometry(params, strip_w)
    print(f"  geometry computed   {time.time() - t1:.1f}s")

    t2 = time.time()
    first_image = save_dataset(strip, geo, params, output_dir)
    print(f"  rendering + save   {time.time() - t2:.1f}s")

    plot_diagnostics(first_image, geo["seg"], geo["u_map"], strip, params, output_dir)
    plot_strip(strip, params, output_dir)
    rmse = verify_roundtrip(
        first_image, geo["seg"], geo["u_map"], strip[0], params, output_dir
    )

    total = time.time() - t0
    print(f"  total time         {total:.1f}s")
    print(f"{'=' * 60}\n")

    return {"rmse": rmse, "output_dir": str(output_dir), "strip_shape": strip.shape}


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic spiral film-roll datasets"
    )
    parser.add_argument(
        "--preset", default="quick_test", choices=list(PRESETS),
        help="Parameter preset (default: quick_test)",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--pattern", default=None,
        choices=["gradient", "bars", "frames", "composite"],
        help="Override strip pattern",
    )
    parser.add_argument(
        "--all-phase2", action="store_true",
        help="Generate all Phase 2 (4K) presets",
    )
    args = parser.parse_args()

    if args.all_phase2:
        for name, preset in PRESETS.items():
            if "4k" in name:
                out = args.output_dir or "unwrapping/synthetic/results"
                generate_dataset(preset, f"{out}/{name}")
    else:
        params = PRESETS[args.preset]
        if args.pattern:
            params.pattern = args.pattern
        out = args.output_dir or f"unwrapping/synthetic/results/{args.preset}"
        generate_dataset(params, out)


if __name__ == "__main__":
    main()
