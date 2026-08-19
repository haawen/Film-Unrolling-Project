"""Unroll the whole Mickey scroll over the CONTINUOUS z-axis (raw CT, 1184 slices).

Segmentation exists only for the 25 gapped 20-slice subvolumes (`01_Mickey_3d`),
but the full raw CT is a contiguous z-stack of single-slice HDF5s
(`01_Mickey_hdf/01_Mickey_Stitch_Stitch_Export_{z:04d}.h5`, key `image`,
z 0832-2015). The roll is ~z-invariant with a slow spool-center drift, so:

  1. Fit center + spline r(phi) on each of the 25 segmented chunks' mid slices
     (z anchors), reusing the gapped pipeline's `fit_chunk`.
  2. Interpolate (center, r(phi) on a fixed phi grid) across z to EVERY raw-CT
     slice (scipy interp1d; clamp outside the anchor span).
  3. Render the raw CT along the interpolated curve for all 1184 slices, with a
     FIXED seam / arc-range / n_cols taken from the reference-chunk manual marks
     (same convention as `unroll_whole_scroll.py`), so rows stay content-aligned.

Per-slice robust (1-99 pct) normalization replaces the gapped run's per-chunk
min-max norm, which removes the inter-chunk intensity banding seen before.

Fit-overlay PNGs at several z (incl. interpolated gap slices) are written BEFORE
the long render loop so the fit can be sanity-checked early / on timeout.

Usage:
  python -m unwrapping.inr.unroll_continuous \
      --ct-dir 01_Mickey_hdf --seg-dir 01_Mickey_3d \
      --out-dir unwrapping/inr/results/whole_scroll_continuous \
      --start-xy 2145,1600 --end-xy 1200,178 \
      --smooth-per-pt 0.35 --n-rays 2880 --seam-exclude-deg 8 [--overlay-only]
"""

import argparse
import glob
import math
import os
import re
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.inr.geodesic_unwrap import _detect_seam_angle
from unwrapping.inr.unwrap_data_3d import discover_volumes, load_volume_chunk
from unwrapping.center_detection import find_spool_center
from unwrapping.inr.sanity_winding_strip import bilinear_sample
from unwrapping.inr.spline_base_unwrap import (
    arclength_table, cols_from_arc, xy_to_phi, snap_to_emulsion, parse_xy,
)
from unwrapping.inr.unroll_whole_scroll import fit_chunk, save_overview, save_zoom


class InterpSpline:
    """r(phi) by linear interpolation of R over a fixed phi grid.

    Mimics the small slice of the UnivariateSpline API that arclength_table
    needs: __call__(phi) and derivative() -> callable.
    """

    def __init__(self, phi_grid, R):
        self.phi_grid = phi_grid
        self.R = R
        self._dR = np.gradient(R, phi_grid)

    def __call__(self, phi):
        return np.interp(phi, self.phi_grid, self.R)

    def derivative(self):
        pg, dR = self.phi_grid, self._dR
        return lambda phi: np.interp(phi, pg, dR)


def discover_ct_slices(ct_dir):
    """Map raw-CT z-index -> file path from 01_Mickey_Stitch..._{z:04d}.h5."""
    z_to_path = {}
    for p in glob.glob(os.path.join(ct_dir, "*.h5")):
        m = re.search(r"_(\d+)\.h5$", os.path.basename(p))
        if m:
            z_to_path[int(m.group(1))] = p
    return z_to_path


def load_ct(path):
    with h5py.File(path, "r") as f:
        return np.asarray(f["image"], dtype=np.float32)


def load_seg_mid(probs_path):
    """Read ONLY the mid z-slice's segmentation from a chunk's probability map.

    Avoids loading the full 20-slice volume + 2.2 GB probability map per chunk
    (load_volume_chunk) just to fit on one slice. Returns (seg_2d, mid_index).
    """
    with h5py.File(probs_path, "r") as f:
        ds = f["exported_data"]
        mid = ds.shape[0] // 2
        probs_mid = ds[mid].astype(np.float32)            # (H, W, 3)
    return np.argmax(probs_mid, axis=-1).astype(np.int64), mid


def load_seg_slices(probs_path, frac=1.0):
    """Read `frac` of a chunk's z-slices' segmentation, evenly spaced in z.

    frac=1.0 -> all 20 slices of the chunk; 0.5 -> ~10, etc. The 2.2 GB prob map
    is opened once and only the selected slices are decompressed. Returns a list of
    (seg_2d, slice_index) so the caller can walk many z-slices per chunk (dense
    anchors) instead of just the mid one. Used for the 50/75/100%-of-slices renders.
    """
    with h5py.File(probs_path, "r") as f:
        ds = f["exported_data"]
        Z = ds.shape[0]
        n = max(1, int(round(frac * Z)))
        idxs = np.unique(np.linspace(0, Z - 1, n).round().astype(int))
        # decompress the whole (z-chunked) prob map ONCE — per-slice ds[i] reads
        # re-decompress the entire 2.2 GB map each time (~20x slower per chunk).
        arr = ds[:]                                       # (Z, H, W, 3)
    return [(np.argmax(arr[int(i)], axis=-1).astype(np.int64), int(i)) for i in idxs]


def norm_slice(img):
    """Robust per-slice normalization to [0,1] (1-99 pct) -> kills z-banding."""
    lo, hi = np.percentile(img, [1, 99])
    return np.clip((img - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def save_overlay(img, cx, cy, xs, ys, z, is_anchor, out_dir, ds=3):
    """Draw the (interpolated) spiral curve on a downscaled raw-CT slice."""
    small = norm_slice(img)[::ds, ::ds]
    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    ax.imshow(small, cmap="gray")
    ax.plot(xs / ds, ys / ds, "-", color="red", lw=0.7, alpha=0.6,
            solid_capstyle="round")
    ax.plot(cx / ds, cy / ds, "c+", markersize=12)
    tag = "ANCHOR (segmented)" if is_anchor else "interpolated gap slice"
    ax.set_title(f"z={z}  [{tag}] — spline r(phi) fit on raw CT")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"fit_overlay_z{z:04d}.png"), dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-dir", required=True, help="Continuous raw-CT h5 dir.")
    ap.add_argument("--seg-dir", required=True, help="Segmented 20-slice chunks.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start-xy", required=True)
    ap.add_argument("--end-xy", required=True)
    ap.add_argument("--smooth-per-pt", type=float, default=0.35)
    ap.add_argument("--n-rays", type=int, default=2880)
    ap.add_argument("--seam-exclude-deg", type=float, default=8.0)
    ap.add_argument("--no-snap", action="store_true")
    ap.add_argument("--n-overlays", type=int, default=8,
                    help="Number of fit-overlay slices across z (incl. anchors).")
    ap.add_argument("--overlay-only", action="store_true",
                    help="Write overlays + exit (fast fit sanity check).")
    ap.add_argument("--z-step", type=int, default=1,
                    help="Render every z-step'th slice (1 = all 1184).")
    ap.add_argument("--inspect", action="store_true",
                    help="Render a few SMALL fixed arc windows (close-up of the "
                         "frames, 1:1) over the whole z-axis, instead of the full "
                         "200k-col strip. Each window uses only a short local arc "
                         "so it avoids the whole-roll spline extrapolation error.")
    ap.add_argument("--n-windows", type=int, default=4,
                    help="Inspect: number of arc windows inner->outer.")
    ap.add_argument("--window-arc-px", type=int, default=2000,
                    help="Inspect: arc length (px) per window, sampled 1:1.")
    ap.add_argument("--use-cached-anchors", action="store_true",
                    help="Load anchors.npz from out-dir (skip the slow 25-chunk "
                         "probability-map fit) if present. Anchors are the same "
                         "every run; this makes render-only reruns fast.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    anchors_path = os.path.join(args.out_dir, "anchors.npz")

    if args.use_cached_anchors and os.path.exists(anchors_path):
        d = np.load(anchors_path)
        z_anchor = d["z_anchor"]; R_anchors = d["R_anchors"]
        cx_a = list(d["cx_a"]); cy_a = list(d["cy_a"])
        phi_grid = d["phi_grid"]; phimaxs = list(d["phimaxs"]); nls = list(d["nls"])
        seam = float(d["seam"])
        phi_lo = float(d["phi_lo"]); phi_hi = float(d["phi_hi"])
        n_cols = int(d["n_cols"]); s_total0 = float(n_cols)
        print(f"Loaded cached anchors ({len(z_anchor)}) from {anchors_path}; "
              f"seam={math.degrees(seam):.1f} arc=[{phi_lo:.2f},{phi_hi:.2f}] "
              f"n_cols={n_cols}")
        _fit_anchors = False
    else:
        _fit_anchors = True

    # ── anchor fits on the 25 segmented chunks ──
    if _fit_anchors:
        _fit_anchor_section(args, anchors_path)
        d = np.load(anchors_path)
        z_anchor = d["z_anchor"]; R_anchors = d["R_anchors"]
        cx_a = list(d["cx_a"]); cy_a = list(d["cy_a"])
        phi_grid = d["phi_grid"]; phimaxs = list(d["phimaxs"]); nls = list(d["nls"])
        seam = float(d["seam"])
        phi_lo = float(d["phi_lo"]); phi_hi = float(d["phi_hi"])
        n_cols = int(d["n_cols"]); s_total0 = float(n_cols)

    fR = interp1d(z_anchor, R_anchors, axis=0, bounds_error=False,
                  fill_value=(R_anchors[0], R_anchors[-1]))
    fcx = interp1d(z_anchor, cx_a, bounds_error=False, fill_value=(cx_a[0], cx_a[-1]))
    fcy = interp1d(z_anchor, cy_a, bounds_error=False, fill_value=(cy_a[0], cy_a[-1]))
    _run_render(args, fR, fcx, fcy, phi_grid, phi_lo, phi_hi, n_cols, s_total0,
                seam, z_anchor)


def _fit_anchor_section(args, anchors_path):
    """Fit center + spline r(phi) on the 25 segmented chunks; write anchors.npz."""
    pairs = discover_volumes(args.seg_dir)
    print(f"{len(pairs)} segmented chunks (anchors), z {pairs[0][2]}-{pairs[-1][3]}")

    # Fixed seam from the reference (first) chunk (mid slice only).
    seg_ref, ref = load_seg_mid(pairs[0][1])
    cy0, cx0 = find_spool_center(seg_ref)
    film0 = seg_ref > 0
    ys0, xs0 = np.where(film0)
    rf = np.hypot(ys0 - cy0, xs0 - cx0)
    seam = _detect_seam_angle(film0, (cy0, cx0), rf.min(), rf.max())
    print(f"  fixed seam = {math.degrees(seam):.1f} deg")

    z_anchor, cx_a, cy_a, spls, phimaxs, nls = [], [], [], [], [], []
    for (vp, pp, z0, z1) in pairs:
        seg_mid, mid = load_seg_mid(pp)
        cx, cy, spl, phimax, nl = fit_chunk(
            seg_mid, args.n_rays, seam, args.seam_exclude_deg, args.smooth_per_pt)
        z_anchor.append(z0 + mid)
        cx_a.append(cx); cy_a.append(cy); spls.append(spl)
        phimaxs.append(phimax); nls.append(nl)
    z_anchor = np.array(z_anchor, dtype=float)
    print(f"  anchor z: {z_anchor.astype(int).tolist()}")
    print(f"  windings per anchor: {nls}")

    # Common phi grid; sample every anchor spline on it for z-interpolation.
    # CLAMP phi to each anchor's own data extent (phimax) before evaluating:
    # the 33-winding anchors have no segmentation out to the 34th turn, and
    # UnivariateSpline extrapolation there blows the outermost turn off-image.
    # Holding the last fitted radius (flat) is safe; the z-interpolation then
    # blends it with the neighboring 34-winding anchors that do cover the turn.
    phi_grid = np.linspace(0.0, float(max(phimaxs)), 12000)
    R_anchors = np.array([s(np.minimum(phi_grid, pm))
                          for s, pm in zip(spls, phimaxs)])     # (n_anchor, n_grid)

    # ── fixed arc-range + n_cols from the reference-chunk manual marks ──
    sx, sy = parse_xy(args.start_xy); ex, ey = parse_xy(args.end_xy)
    if not args.no_snap:
        sx, sy = snap_to_emulsion(sx, sy, seg_ref)
        ex, ey = snap_to_emulsion(ex, ey, seg_ref)
    nl0 = nls[0]
    phi_lo = xy_to_phi(sx, sy, cx_a[0], cy_a[0], seam, spls[0], nl0)
    phi_hi = xy_to_phi(ex, ey, cx_a[0], cy_a[0], seam, spls[0], nl0)
    if phi_lo > phi_hi:
        phi_lo, phi_hi = phi_hi, phi_lo
    _, _, s_total0 = arclength_table(spls[0], phi_hi, phi_min=phi_lo)
    n_cols = int(round(s_total0))
    print(f"  arc range phi=[{phi_lo:.2f},{phi_hi:.2f}] "
          f"({(phi_hi - phi_lo) / (2 * math.pi):.2f} turns), n_cols={n_cols} (1:1)")
    np.savez_compressed(
        os.path.join(args.out_dir, "anchors.npz"),
        z_anchor=z_anchor, R_anchors=R_anchors, cx_a=cx_a, cy_a=cy_a,
        phi_grid=phi_grid, phimaxs=phimaxs, nls=nls, seam=seam,
        phi_lo=phi_lo, phi_hi=phi_hi, n_cols=n_cols)


def _run_render(args, fR, fcx, fcy, phi_grid, phi_lo, phi_hi, n_cols, s_total0,
                seam, z_anchor):
    """Overlays + (inspect windows | full strip) using the interpolated fit."""

    def fit_for_z(z):
        """Interpolated (cx, cy, spline, arclength table) for one z."""
        cx, cy = float(fcx(z)), float(fcy(z))
        spl = InterpSpline(phi_grid, fR(z))
        phid, sd, st = arclength_table(spl, phi_hi, phi_min=phi_lo)
        return cx, cy, spl, phid, sd, st

    def cols_for_z(z, col_s=None):
        """Interpolated (cx, cy, col_phi, col_r) for one z.

        col_s = arc-length sample positions (px); default = full 1:1 strip.
        """
        cx, cy, spl, phid, sd, st = fit_for_z(z)
        if col_s is None:
            col_s = np.linspace(0.0, st, n_cols)
        col_phi, col_r = cols_from_arc(spl, phid, sd, col_s)
        return cx, cy, col_phi, col_r

    z_to_path = discover_ct_slices(args.ct_dir)
    z_list = sorted(z_to_path)[:: args.z_step]
    print(f"{len(z_to_path)} raw-CT slices (z {z_list[0]}-{z_list[-1]}); "
          f"rendering {len(z_list)} (step {args.z_step})")

    # ── fit-overlay PNGs first (cheap sanity check before the render loop) ──
    anchor_zset = set(int(z) for z in z_anchor)
    ov_z = [int(round(v)) for v in np.linspace(z_list[0], z_list[-1], args.n_overlays)]
    ov_z = sorted(set(ov_z) | {z_list[0], z_list[-1]})
    for z in ov_z:
        z = min(z_to_path, key=lambda zz: abs(zz - z))
        cx, cy, col_phi, col_r = cols_for_z(z)
        theta = seam + col_phi
        xs = cx + col_r * np.cos(theta); ys = cy + col_r * np.sin(theta)
        save_overlay(load_ct(z_to_path[z]), cx, cy, xs, ys, z,
                     z in anchor_zset, args.out_dir)
    print(f"  wrote {len(ov_z)} fit overlays: {ov_z}")
    if args.overlay_only:
        print("Done (overlay-only).")
        return

    # ── inspect: small fixed arc windows over the WHOLE z-axis (close-up) ──
    if args.inspect:
        Wpx = args.window_arc_px
        # Window centres evenly spaced inner->outer (fixed arc-length px, so the
        # same physical film region is sampled at every z -> rows stay aligned).
        centres = np.linspace(0.12, 0.88, args.n_windows) * s_total0
        win_s0 = [float(np.clip(c - Wpx / 2.0, 0.0, max(0.0, s_total0 - Wpx)))
                  for c in centres]
        wins = [np.empty((len(z_list), Wpx), dtype=np.float32) for _ in win_s0]
        win_wnd = [None] * len(win_s0)              # approx winding number per window
        print(f"  inspect: {len(win_s0)} windows x {Wpx}px @ 1:1 over {len(z_list)} z")
        for i, z in enumerate(z_list):
            img = norm_slice(load_ct(z_to_path[z]))
            cx, cy, spl, phid, sd, st = fit_for_z(z)
            for w, s0 in enumerate(win_s0):
                col_s = s0 + np.arange(Wpx)
                col_phi, col_r = cols_from_arc(spl, phid, sd, col_s)
                theta = seam + col_phi
                wins[w][i] = bilinear_sample(img, cx + col_r * np.cos(theta),
                                             cy + col_r * np.sin(theta))
                if win_wnd[w] is None:
                    win_wnd[w] = int(col_phi[len(col_phi) // 2] / (2 * math.pi))
            if i % 100 == 0 or i == len(z_list) - 1:
                print(f"  [{i + 1}/{len(z_list)}] z{z}", flush=True)
        z_index = np.array(z_list)
        np.save(os.path.join(args.out_dir, "z_index.npy"), z_index)
        for w, (s0, strip) in enumerate(zip(win_s0, wins)):
            np.save(os.path.join(args.out_dir, f"inspect_w{w}.npy"), strip)
            lo, hi = np.percentile(strip, [1, 99])
            disp = np.clip((strip - lo) / (hi - lo + 1e-6), 0, 1)
            from PIL import Image as _Im
            _Im.fromarray((disp * 255).astype(np.uint8)).save(
                os.path.join(args.out_dir, f"inspect_w{w}_raw.png"))
            fig, ax = plt.subplots(1, 1, figsize=(14, max(4, len(z_list) / 90)))
            ax.imshow(disp, cmap="gray", aspect="auto", interpolation="nearest")
            ax.set_title(f"Inspect window {w} (~winding {win_wnd[w]}, "
                         f"arc {int(s0)}-{int(s0) + Wpx}px, {len(z_list)} z-slices, 1:1)")
            ax.set_xlabel("arc length (px)  →"); ax.set_ylabel("z slice")
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"inspect_w{w}.png"), dpi=130)
            plt.close()
            print(f"  window {w}: {strip.shape} arc[{int(s0)},{int(s0)+Wpx}] "
                  f"~winding {win_wnd[w]}")
        print(f"Done (inspect). Outputs in {args.out_dir}")
        return

    # ── render every raw-CT slice along its interpolated curve ──
    strip = np.empty((len(z_list), n_cols), dtype=np.float32)
    for i, z in enumerate(z_list):
        img = norm_slice(load_ct(z_to_path[z]))
        cx, cy, col_phi, col_r = cols_for_z(z)
        theta = seam + col_phi
        strip[i] = bilinear_sample(img, cx + col_r * np.cos(theta),
                                   cy + col_r * np.sin(theta))
        if i % 100 == 0 or i == len(z_list) - 1:
            print(f"  [{i + 1}/{len(z_list)}] z{z} center=({cx:.0f},{cy:.0f})",
                  flush=True)

    z_index = np.array(z_list)
    print(f"FULL STRIP: {strip.shape}  ({strip.nbytes / 1e6:.0f} MB)")
    np.save(os.path.join(args.out_dir, "full_strip.npy"), strip)
    np.save(os.path.join(args.out_dir, "z_index.npy"), z_index)
    save_overview(strip, args.out_dir)
    for frac, tag in [(0.04, "start"), (0.33, "early"), (0.5, "mid"),
                      (0.66, "late"), (0.96, "end")]:
        save_zoom(strip, args.out_dir, frac, tag)
    print(f"Done. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
