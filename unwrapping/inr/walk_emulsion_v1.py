"""Emulsion-walk unwrapper (2026-06-30) — user's rethink.

Drops the polar r(theta)-about-a-center frame entirely (its failures: wrong
center/eccentricity + curves wandering between bands). Instead:

  1. SEED at one clean ray (or thin band): march outward, count class-2 emulsion
     RUNS that are at least `min-thick` px thick (ignores noise specks). The
     middle of each run = one seed point, one per winding. Count = #windings,
     straight from the data (no center-count, no infill).
  2. WALK each seed tangentially along ITS emulsion band, both directions, with a
     predictor-corrector: step along the local tangent, then snap onto the class-2
     centroid within a small perpendicular window (< half the layer spacing). Coast
     across dashes. Stop ~half a turn each way (meet at the seam).

Why this beats the polar methods on the two observed failures:
  * No center / no polar: the walk is local, so eccentricity is irrelevant.
  * No assignment: each band is followed as a curve, so curves can't wander into
    a neighbour (the perpendicular snap is capped below the spacing).
  * Touch-robust: even where film BASES touch, the class-2 emulsion sublayers stay
    separated by the base thickness, so snapping to class-2 only stays on-band.
  * Pinch-robust: walks ALONG the band, so the radial pinch connections that break
    geodesics (see algorithmic-unwrapping.md, geodesic abandoned) are ignored.

Prototype on ONE chunk (mid-slice fit, render strip over its z) before any roll.

Usage:
  python -m unwrapping.inr.walk_emulsion \
      --vol 01_Mickey_3d/volume_1595-1614.h5 \
      --probs 01_Mickey_3d/volume_1595-1614_Probabilities.h5 \
      --out-dir <out> [--seed-az-deg <abs deg>] [--min-thick 3] [--snap-max 8]
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.center_detection import find_spool_center
from unwrapping.inr.geodesic_unwrap import _detect_seam_angle
from unwrapping.inr.unwrap_data_3d import load_volume_chunk
from unwrapping.inr.sanity_winding_strip import bilinear_sample

TWO_PI = 2.0 * math.pi


def seed_runs(emul, cx, cy, az, r0, r1, min_thick, step=0.5):
    """Class-2 runs >= min_thick px along the ray at absolute angle `az`.

    Returns the middle radius of each run (one seed radius per winding)."""
    rr = np.arange(r0, r1, step)
    xs = cx + rr * math.cos(az); ys = cy + rr * math.sin(az)
    on = emul[np.clip(np.round(ys).astype(int), 0, emul.shape[0] - 1),
              np.clip(np.round(xs).astype(int), 0, emul.shape[1] - 1)] > 0.5
    seeds, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            if (rr[j - 1] - rr[i]) >= min_thick:
                seeds.append(0.5 * (rr[i] + rr[j - 1]))
            i = j
        else:
            i += 1
    return seeds


def band_seed_radii(emul, cx, cy, az, r0, r1, min_thick, half_deg=1.5, n_rays=7):
    """Robust seeds: take the ray in the +-half_deg band with the MOST runs."""
    best, bestn = [], -1
    for da in np.linspace(-math.radians(half_deg), math.radians(half_deg), n_rays):
        s = seed_runs(emul, cx, cy, az + da, r0, r1, min_thick)
        if len(s) > bestn:
            bestn, best = len(s), s
    return best


def best_seed_azimuth(emul, cx, cy, r0, r1, min_thick, n_scan=180):
    """Scan all azimuths; return the one whose ray has the MOST thick emulsion
    runs (the cleanest seed ray, where the most windings are separable). Robust to
    a dashed seed ray on noisy slices."""
    best_az, best, bestn = 0.0, [], -1
    for az in np.linspace(0, TWO_PI, n_scan, endpoint=False):
        s = seed_runs(emul, cx, cy, az, r0, r1, min_thick)
        if len(s) > bestn:
            bestn, best, best_az = len(s), s, az
    return best_az, best


def film_seed_azimuth(film, emul, cx, cy, r0, r1, min_thick, snap, n_scan=180):
    """Seed COUNT + radii from the CONTINUOUS film bands (class>=1), then pre-snap
    each onto the emulsion.

    Film bands are continuous (don't dash like the emulsion) so they give the true
    winding count ~34. Scan azimuths for the ray with the MOST film runs (where
    touching turns are air-separated). Each film-band center is then snapped to the
    nearest class-2 emulsion pixel along the ray (within +-snap) so the walk starts
    on the emulsion; if none there (dashed), keep the film center and let the walk's
    corrector snap on as it goes.
    """
    best_az, best, bestn = 0.0, [], -1
    for az in np.linspace(0, TWO_PI, n_scan, endpoint=False):
        s = seed_runs(film, cx, cy, az, r0, r1, min_thick)
        if len(s) > bestn:
            bestn, best, best_az = len(s), s, az
    # pre-snap each film-band center onto the emulsion along best_az
    snapped = []
    for rc in best:
        rr = np.arange(rc - snap, rc + snap, 0.5)
        xs = cx + rr * math.cos(best_az); ys = cy + rr * math.sin(best_az)
        on = emul[np.clip(np.round(ys).astype(int), 0, emul.shape[0] - 1),
                  np.clip(np.round(xs).astype(int), 0, emul.shape[1] - 1)] > 0.5
        snapped.append(float(rr[on].mean()) if on.any() else rc)
    return best_az, snapped


def walk(emul, p0, d0, cx, cy, step, snap_max, coast_max, stop_extent,
         smooth=0.5):
    """Predictor-corrector walk along the emulsion band from p0 in direction d0.

    Stops after cumulative azimuth change reaches stop_extent (or coasting too
    long / leaving the image). Returns the path as an (N,2) array of (x,y)."""
    H, W = emul.shape
    p = np.array(p0, float); d = np.array(d0, float); d /= np.linalg.norm(d)
    path = [p.copy()]
    a_prev = math.atan2(p[1] - cy, p[0] - cx); cum = 0.0
    ts = np.arange(-snap_max, snap_max + 1.0)
    coast = 0
    for _ in range(20000):
        p_pred = p + step * d
        n = np.array([-d[1], d[0]])                       # perpendicular
        xs = p_pred[0] + ts * n[0]; ys = p_pred[1] + ts * n[1]
        ix = np.clip(np.round(xs).astype(int), 0, W - 1)
        iy = np.clip(np.round(ys).astype(int), 0, H - 1)
        vals = emul[iy, ix]
        if vals.sum() > 0:
            t_snap = float((ts * vals).sum() / vals.sum())
            t_snap = float(np.clip(t_snap, -snap_max, snap_max))
            p_new = p_pred + t_snap * n
            coast = 0
        else:
            p_new = p_pred
            coast += 1
            if coast > coast_max:
                break
        step_vec = p_new - p
        if np.linalg.norm(step_vec) > 1e-6:
            d_new = step_vec / np.linalg.norm(step_vec)
            d = smooth * d + (1 - smooth) * d_new
            d /= np.linalg.norm(d)
        a = math.atan2(p_new[1] - cy, p_new[0] - cx)
        da = (a - a_prev + math.pi) % TWO_PI - math.pi
        cum += da; a_prev = a
        p = p_new; path.append(p.copy())
        if abs(cum) >= stop_extent:
            break
        if not (0 <= p[0] < W and 0 <= p[1] < H):
            break
    return np.array(path)


def smooth_path(path, smooth_px):
    """Denoise a walked path with a PARAMETRIC smoothing spline x(t), y(t) over
    arc length t — removes the per-step snapping jitter while keeping the band's
    large-scale shape. Center-free. Resamples to ~1 point per arc px."""
    from scipy.interpolate import UnivariateSpline
    if len(path) < 8:
        return path
    d = np.r_[0.0, np.cumsum(np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1])))]
    s = (smooth_px ** 2) * len(d)
    sx = UnivariateSpline(d, path[:, 0], k=3, s=s)
    sy = UnivariateSpline(d, path[:, 1], k=3, s=s)
    tn = np.linspace(0, d[-1], max(2, int(round(d[-1]))))
    return np.column_stack([sx(tn), sy(tn)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed-az-deg", type=float, default=None,
                    help="Absolute seed-ray angle (deg). Default = auto (cleanest).")
    ap.add_argument("--seed-mode", choices=["film", "emulsion"], default="film",
                    help="film = count+radii from continuous film bands (~34, "
                         "robust on noisy) pre-snapped to emulsion; emulsion = "
                         "class-2 runs directly (undercounts where dashed).")
    ap.add_argument("--film-min-thick", type=float, default=6.0,
                    help="Min class>=1 film-run thickness (px) for film seeding.")
    ap.add_argument("--min-thick", type=float, default=3.0,
                    help="Min class-2 run thickness (px) to seed a winding.")
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--snap-max", type=float, default=8.0,
                    help="Perpendicular snap half-window (px); < half layer spacing.")
    ap.add_argument("--coast-max", type=int, default=40,
                    help="Max consecutive dashed steps to coast (px = step*this).")
    ap.add_argument("--seam-exclude-deg", type=float, default=6.0)
    ap.add_argument("--smooth-px", type=float, default=0.6,
                    help="Parametric path-smoothing (target per-point RMS px); "
                         "removes per-step walk jitter. 0 disables. 0.6 hugs the "
                         "band closely + centered on emulsion (1.5 over-rounded).")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    vol, seg = load_volume_chunk(args.vol, args.probs)
    Z = vol.shape[0]; mid = Z // 2
    seg2 = seg[mid]; emul = (seg2 == 2).astype(np.float32)
    cy, cx = find_spool_center(seg2)
    film = seg2 > 0; ys, xs = np.where(film)
    rf = np.hypot(ys - cy, xs - cx); r0, r1 = rf.min() + 5, rf.max() - 5
    seam = _detect_seam_angle(film, (cy, cx), rf.min(), rf.max())
    if args.seed_az_deg is not None:
        seed_az = math.radians(args.seed_az_deg)
        seeds = band_seed_radii(emul, cx, cy, seed_az, r0, r1, args.min_thick)
    elif args.seed_mode == "film":
        seed_az, seeds = film_seed_azimuth(film.astype(np.float32), emul, cx, cy,
                                           r0, r1, args.film_min_thick, args.snap_max)
    else:
        seed_az, seeds = best_seed_azimuth(emul, cx, cy, r0, r1, args.min_thick)
    print(f"{Z} slices; center=({cx:.0f},{cy:.0f}); seam={math.degrees(seam):.1f}deg; "
          f"seed_az={math.degrees(seed_az):.1f}deg; r[{r0:.0f},{r1:.0f}]")
    print(f"  {len(seeds)} windings seeded (radii {seeds[0]:.0f}..{seeds[-1]:.0f})")

    stop_extent = math.pi - math.radians(args.seam_exclude_deg)
    tan_ccw = np.array([-math.sin(seed_az), math.cos(seed_az)])
    paths = []
    for rs in seeds:
        p0 = np.array([cx + rs * math.cos(seed_az), cy + rs * math.sin(seed_az)])
        fwd = walk(emul, p0, tan_ccw, cx, cy, args.step, args.snap_max,
                   args.coast_max, stop_extent)
        bwd = walk(emul, p0, -tan_ccw, cx, cy, args.step, args.snap_max,
                   args.coast_max, stop_extent)
        path = np.vstack([bwd[::-1], fwd[1:]])            # seam -> ... -> seam
        paths.append(smooth_path(path, args.smooth_px))

    # ── overlays: walked paths on CT and on seg ──
    disp = vol[mid]; lo, hi = np.percentile(disp, [1, 99])
    disp = np.clip((disp - lo) / (hi - lo + 1e-6), 0, 1)
    seg_rgb = np.zeros(seg2.shape + (3,), np.float32)
    seg_rgb[seg2 == 1] = (0.55, 0.55, 0.55); seg_rgb[seg2 == 2] = (1, 0.15, 0.15)
    cmap = plt.get_cmap("hsv")
    for bg, img, tag in [("ct", disp, "CT"), ("seg", seg_rgb, "SEG(emul=red)")]:
        fig, ax = plt.subplots(1, 1, figsize=(11, 11))
        ax.imshow(img, cmap="gray" if bg == "ct" else None)
        for k, pth in enumerate(paths):
            ax.plot(pth[:, 0], pth[:, 1], "-", color=cmap(k / max(1, len(paths))),
                    lw=0.7)
        ax.plot(cx, cy, "c+", ms=10)
        ax.set_title(f"{len(paths)} emulsion walks on {tag}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, f"walk_{bg}.png"),
                                        dpi=130); plt.close()

    # ── ZOOM: paths over emulsion pixels at full res (does the walk HUG?) ──
    ey, ex = np.where(seg2 == 2)
    th_e = np.arctan2(ey - cy, ex - cx)
    re = np.hypot(ey - cy, ex - cx)
    rmid = 0.5 * (r0 + r1)
    wmid = -math.pi / 2; wband = math.radians(8)
    m = (np.abs((th_e - wmid + math.pi) % TWO_PI - math.pi) < wband) \
        & (re > rmid - 130) & (re < rmid + 130)         # small mid-radius window
    if m.sum() > 50:
        x0, x1 = max(0, ex[m].min() - 12), ex[m].max() + 12
        y0, y1 = max(0, ey[m].min() - 12), ey[m].max() + 12
        fig, ax = plt.subplots(1, 1, figsize=(11, 11 * (y1 - y0) / max(1, x1 - x0)))
        ax.imshow(disp[y0:y1, x0:x1], cmap="gray")
        ax.scatter(ex[m] - x0, ey[m] - y0, s=3, c="cyan", alpha=0.35, label="emulsion px")
        for k, pth in enumerate(paths):
            ax.plot(pth[:, 0] - x0, pth[:, 1] - y0, "-", color="yellow", lw=0.9)
        ax.plot([], [], "-", color="yellow", lw=0.9, label="walk")
        ax.set_xlim(0, x1 - x0); ax.set_ylim(y1 - y0, 0); ax.legend(fontsize=9)
        ax.set_title("ZOOM: emulsion walks (yellow) vs emulsion px (cyan)")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "walk_zoom.png"),
                                        dpi=120); plt.close()

    # ── strip: sample CT along each winding path over all z, concat inner->outer ──
    strips = []
    for pth in paths:
        s = np.empty((Z, pth.shape[0]), np.float32)
        for z in range(Z):
            s[z] = bilinear_sample(vol[z], pth[:, 0], pth[:, 1])
        strips.append(s)
    minlen = min(s.shape[1] for s in strips)
    strip = np.concatenate([s[:, :minlen] for s in strips], axis=1)
    from PIL import Image as _Im
    loS, hiS = np.percentile(strip, [1, 99])
    sd = np.clip((strip - loS) / (hiS - loS + 1e-6), 0, 1)
    # downscaled full-strip overview (<=1800 wide) — global coherence across arc
    stepc = max(1, strip.shape[1] // 1800)
    ov = np.repeat(sd[:, ::stepc], 5, 0)
    _Im.fromarray((ov * 255).astype(np.uint8)).save(
        os.path.join(args.out_dir, "strip_overview.png"))
    # 1:1 content crop at mid-arc (a window of the actual unrolled emulsion)
    c0 = strip.shape[1] // 2
    crop = np.repeat(sd[:, c0:c0 + 1400], 8, 0)
    _Im.fromarray((crop * 255).astype(np.uint8)).save(
        os.path.join(args.out_dir, "strip_crop.png"))
    print(f"  path lens {[p.shape[0] for p in paths][:6]}... ; "
          f"strip {strip.shape}. Done -> {args.out_dir}")


if __name__ == "__main__":
    main()
