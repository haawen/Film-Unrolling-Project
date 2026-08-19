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


def seed_run_intervals(mask, cx, cy, az, r0, r1, min_thick, step=0.5):
    """(r_lo, r_hi) intervals of mask>0.5 runs >= min_thick px along the ray at
    absolute angle `az`."""
    rr = np.arange(r0, r1, step)
    xs = cx + rr * math.cos(az); ys = cy + rr * math.sin(az)
    on = mask[np.clip(np.round(ys).astype(int), 0, mask.shape[0] - 1),
              np.clip(np.round(xs).astype(int), 0, mask.shape[1] - 1)] > 0.5
    runs, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            if (rr[j - 1] - rr[i]) >= min_thick:
                runs.append((float(rr[i]), float(rr[j - 1])))
            i = j
        else:
            i += 1
    return runs


def seed_runs(emul, cx, cy, az, r0, r1, min_thick, step=0.5):
    """Class-2 runs >= min_thick px along the ray at absolute angle `az`.

    Returns the middle radius of each run (one seed radius per winding)."""
    return [0.5 * (a + b) for a, b in
            seed_run_intervals(emul, cx, cy, az, r0, r1, min_thick, step)]


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


def infill_spacing(bands):
    """Infill windings the single-ray detector missed: a missing winding shows up
    as a gap ~= k x the median single-winding spacing, so split such gaps evenly.
    Recovers dropped windings without merging neighbours (mirrors the winning
    poly-fit `filmband_seeds`). `bands` = sorted radii; returns sorted radii."""
    bands = np.sort(np.asarray(bands, float))
    if bands.size < 2:
        return list(bands)
    med = float(np.median(np.diff(bands)))                 # typical 1-winding spacing
    out = [bands[0]]
    for i in range(1, len(bands)):
        g = bands[i] - bands[i - 1]
        nfill = max(0, int(round(g / med)) - 1)            # missing windings in gap
        for j in range(1, nfill + 1):
            out.append(bands[i - 1] + g * j / (nfill + 1))
        out.append(bands[i])
    return out


def _snap_to_emul(emul, cx, cy, az, rc, snap):
    """Snap a film-band radius rc onto the NEAREST class-2 RUN centre along the ray
    at azimuth az (within +-snap); if none (dashed), keep rc. Snapping to the
    nearest run (not the mean of all in-window emulsion px) keeps two split seeds
    of a touching pair on their OWN emulsions instead of both averaging onto one."""
    runs = seed_run_intervals(emul, cx, cy, az, rc - snap, rc + snap, min_thick=1.0)
    if not runs:
        return rc
    centers = np.array([0.5 * (a + b) for a, b in runs])
    return float(centers[np.argmin(np.abs(centers - rc))])


def _split_merged_run(emul, cx, cy, az, r_lo, r_hi, med_thick):
    """Split a film run ~k x the median run thickness (k touching windings whose
    bases merged into one class>=1 run) into k seed radii.

    Prefer the interior class-2 evidence: even where the BASES touch, the k
    emulsion sublayers stay separated, so k emulsion runs inside the film run give
    the true split points. Fall back to an even split when the emulsion is dashed
    right there."""
    t = r_hi - r_lo
    k = max(1, int(round(t / max(1e-6, med_thick))))
    if k == 1:
        return [0.5 * (r_lo + r_hi)]
    er = seed_run_intervals(emul, cx, cy, az, r_lo - 1, r_hi + 1, min_thick=2.0)
    if len(er) >= 2:                       # emulsions resolve the windings directly
        return [0.5 * (a + b) for a, b in er]
    if t >= 1.7 * med_thick:               # blind even split needs strong evidence
        return [r_lo + (j + 0.5) * t / k for j in range(k)]
    return [0.5 * (r_lo + r_hi)]           # seg-thickened single run — don't split


def infill_spacing_local(bands, window=4, max_fill=3):
    """Spacing infill against the LOCAL gap median (window gaps each side,
    excluding the gap under test). Real spacing varies 17-28 px, so the global
    median mis-rounds borderline gaps (g=30 vs med=22 -> 0 infill = silent miss);
    the local median catches a missing winding in a locally-tight region.

    `max_fill` BOUNDS how many windings one gap may claim to hide. Without it a
    single spurious film run amplifies catastrophically: on the new scan one
    detection near the roll's hollow centre left a 775px gap to the innermost real
    winding, and at a ~23px local median that minted 33 fake seeds straight across
    the empty bore (observed: 37 real windings -> 71 seeds, tracks at r=18..759
    where the film starts at r=776). A genuine gap hides at most 2-3 windings
    (touching film), so a gap implying more is a VOID, not film: fill nothing.
    Returns (seeds, n_voids) so callers can report it.
    """
    bands = np.sort(np.asarray(bands, float))
    if bands.size < 3:
        return list(bands), 0
    gaps = np.diff(bands)
    out = [bands[0]]
    n_void = 0
    for i in range(len(gaps)):
        lo, hi = max(0, i - window), min(len(gaps), i + window + 1)
        others = np.r_[gaps[lo:i], gaps[i + 1:hi]]
        med = float(np.median(others)) if others.size else float(np.median(gaps))
        nfill = max(0, int(round(gaps[i] / max(1e-6, med))) - 1)
        if nfill > max_fill:
            n_void += 1                       # void (bore / outside the film)
        else:
            for j in range(1, nfill + 1):
                out.append(bands[i] + gaps[i] * j / (nfill + 1))
        out.append(bands[i + 1])
    return out, n_void


def film_seed_azimuth(film, emul, cx, cy, r0, r1, min_thick, snap, n_scan=180,
                      infill=True, consensus=True, verbose=True, max_fill=3):
    """Seed COUNT + radii from the CONTINUOUS film bands (class>=1), then pre-snap
    each onto the emulsion.

    Film bands are continuous (don't dash like the emulsion) so they carry the true
    winding count. But ANY single ray undercounts wherever two turns TOUCH at that
    azimuth (merged class>=1 run = 1 seed for 2 windings) — measured: the true
    count exceeds the best-ray count at most azimuths. So (consensus=True):
      1. scan all azimuths, split over-thick runs by their interior emulsion runs
         (emulsions stay separated even where bases touch) -> per-ray SPLIT counts;
      2. the consensus count N* = the 90th percentile of split counts over rays
         (robust: rays crossing extra touches undercount, noisy rays overcount);
      3. seed at the ray with the max split count; LOCAL-spacing infill (global
         median mis-rounds borderline gaps at 17-28px varying spacing).
    Each seed is snapped to the nearest class-2 RUN along the ray (within +-snap)
    so the walk starts on its own emulsion.
    """
    azs = np.linspace(0, TWO_PI, n_scan, endpoint=False)
    ray_runs = [seed_run_intervals(film, cx, cy, az, r0, r1, min_thick)
                for az in azs]
    if not consensus:
        best_i = int(np.argmax([len(r) for r in ray_runs]))
        best_az = float(azs[best_i])
        best = [0.5 * (a + b) for a, b in ray_runs[best_i]]
        if infill:
            best, _ = infill_spacing_local(best, max_fill=max_fill)
        return best_az, [_snap_to_emul(emul, cx, cy, best_az, rc, snap)
                         for rc in best]
    med_thick = float(np.median([b - a for r in ray_runs for a, b in r]))
    split_counts = np.array([sum(max(1, int(round((b - a) / med_thick)))
                                 for a, b in r) for r in ray_runs])
    n_star = int(np.percentile(split_counts, 90))
    best_i = int(np.argmax(split_counts))
    best_az = float(azs[best_i])
    seeds = []
    for a, b in ray_runs[best_i]:
        seeds.extend(_split_merged_run(emul, cx, cy, best_az, a, b, med_thick))
    n_split = len(seeds)
    n_void = 0
    if infill:
        seeds, n_void = infill_spacing_local(seeds, max_fill=max_fill)
    # A seed count far above the consensus N* means the seeds are not windings.
    # (New-scan symptom: N*=35, split->37, infill->71.)
    if len(seeds) > n_star + max_fill + 2:
        print(f"    WARNING seeds {len(seeds)} >> consensus N*={n_star}: the seed "
              f"ray is finding non-film structure (bore artefacts / noise). "
              f"Trimming to the {n_star + max_fill} outermost.", flush=True)
        seeds = sorted(seeds)[-(n_star + max_fill):]
    if verbose:
        print(f"    seed az={math.degrees(best_az):.0f}deg N*={n_star} "
              f"runs={len(ray_runs[best_i])} split->{n_split} "
              f"infill->{len(seeds)}"
              f"{f' voids-skipped={n_void}' if n_void else ''}", flush=True)
    return best_az, [_snap_to_emul(emul, cx, cy, best_az, rc, snap)
                     for rc in seeds]


def film_seed_multi(film, emul, cx, cy, r0, r1, min_thick, snap, n_scan=180,
                    top_m=3, min_az_sep_deg=40.0, infill=True):
    """UNION seeding from the top-M film-band rays, spread apart in azimuth.

    A single seed ray misses the 1-2 windings that pinch/merge at that azimuth (the
    residual 'winding miss'). Seeding from several well-separated rays and pooling
    the walks recovers them: a winding hidden at one azimuth is seeded from another.
    Redundant walks of the same winding fully overlap (each spans seam->seam) and
    collapse at dedup, so the union only ADDS the missing windings, never doubles.
    Returns a list of (az, [snapped radii]) — walk each seed set from its own az.
    """
    azs = np.linspace(0, TWO_PI, n_scan, endpoint=False)
    runs = [seed_runs(film, cx, cy, az, r0, r1, min_thick) for az in azs]
    counts = np.array([len(s) for s in runs])
    min_sep = math.radians(min_az_sep_deg)
    chosen = []
    for i in np.argsort(counts)[::-1]:                 # most-complete rays first
        az = azs[i]
        if all(abs((az - c + math.pi) % TWO_PI - math.pi) > min_sep
               for c, _ in chosen):                    # keep rays spread in azimuth
            chosen.append((az, runs[i]))
        if len(chosen) >= top_m:
            break
    out = []
    for az, s in chosen:
        bands = infill_spacing(s) if infill else sorted(s)
        out.append((az, [_snap_to_emul(emul, cx, cy, az, rc, snap) for rc in bands]))
    return out


def walk(emul, p0, d0, cx, cy, step, snap_max, coast_max, stop_extent,
         smooth=0.5, jump_max=4.0, min_fwd=0.25, max_rad=0.55, snap_accept=None,
         max_dr=None):
    """Predictor-corrector walk along the emulsion band from p0 in direction d0.

    Robustness constraints (all soft — they steer, not hard-block):
      * NEAREST-RUN snap: the perpendicular window can straddle TWO bands (at a
        pinch, or where our own band is dashed and a neighbour is in range). Instead
        of the centroid of ALL class-2 pixels in the window (which gets dragged
        across the gap onto the neighbour -> winding SKIP), snap to the contiguous
        class-2 RUN whose centre is nearest the predicted band (t=0). This is the
        core anti-skip fix — the walk can't be pulled onto a neighbour it merely
        sees at the window edge.
      * snap_accept: if even the nearest run's centre is farther than this, our band
        is dashed here and what we see is only a neighbour -> COAST straight instead
        of snapping (default = snap_max, i.e. off unless tightened).
      * jump_max: cap the per-step perpendicular snap so the walk can't SUDDENLY
        jump radius (onto a neighbour band); gradual drift still allowed.
      * max_dr: cap the change in RADIUS (about the center) per step. This is the
        primary anti-JUMP lever: a band-to-band jump is a fast radius change
        (~spacing over a few steps) while real eccentricity drifts ~0.05 px/step, so
        a small cap (e.g. 2 px/step) blocks jumps yet leaves eccentric drift free.
        Uses the center only as a rate reference (no circular-ring assumption).
      * min_fwd: force the tangential (azimuthal) component of the heading to stay
        forward -> NO REVERSE (the walk always progresses in its start direction).
      * max_rad: cap the radial component of the heading -> the walk stays roughly
        tangential (can't dive steeply across bands).
    Stops after cumulative azimuth change reaches stop_extent (or coasting / edge).
    """
    H, W = emul.shape
    sa = snap_max if snap_accept is None else snap_accept
    p = np.array(p0, float); d = np.array(d0, float); d /= np.linalg.norm(d)
    # intended forward azimuthal sense from the start heading
    th0 = math.atan2(p[1] - cy, p[0] - cx)
    that0 = np.array([-math.sin(th0), math.cos(th0)])
    sgn = 1.0 if np.dot(d, that0) >= 0 else -1.0
    path = [p.copy()]
    a_prev = th0; cum = 0.0
    ts = np.arange(-snap_max, snap_max + 1.0)
    coast = 0
    for _ in range(20000):
        p_pred = p + step * d
        n = np.array([-d[1], d[0]])                       # perpendicular
        xs = p_pred[0] + ts * n[0]; ys = p_pred[1] + ts * n[1]
        ix = np.clip(np.round(xs).astype(int), 0, W - 1)
        iy = np.clip(np.round(ys).astype(int), 0, H - 1)
        on = emul[iy, ix] > 0.5
        idx = np.where(on)[0]
        coasted = False
        if idx.size > 0:
            # split the in-window class-2 hits into contiguous runs; take the run
            # centre nearest t=0 (our band), NOT the centroid of both bands.
            groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
            centers = np.array([ts[g].mean() for g in groups])
            t_snap = float(centers[np.argmin(np.abs(centers))])
            if abs(t_snap) > sa:                          # only a neighbour visible
                p_new = p_pred; coasted = True
                coast += 1
                if coast > coast_max:
                    break
            else:
                t_snap = float(np.clip(t_snap, -jump_max, jump_max))  # no sudden jump
                p_new = p_pred + t_snap * n
                coast = 0
        else:
            p_new = p_pred; coasted = True
            coast += 1
            if coast > coast_max:
                break
        r_prev = math.hypot(p[0] - cx, p[1] - cy)
        if coasted:
            # COAST at CONSTANT radius: with no band to snap to, move purely
            # tangentially instead of drifting radially along the (tilted) heading
            # onto a neighbour band. This is the primary anti-jump fix — a dash
            # can no longer let the walk climb across the gap.
            thn = math.atan2(p_new[1] - cy, p_new[0] - cx)
            p_new = np.array([cx + r_prev * math.cos(thn), cy + r_prev * math.sin(thn)])
        elif max_dr is not None:
            # cap the per-step radius change on a snap step (blocks a hard jump)
            r_new = math.hypot(p_new[0] - cx, p_new[1] - cy)
            if abs(r_new - r_prev) > max_dr:
                thn = math.atan2(p_new[1] - cy, p_new[0] - cx)
                rc = r_prev + math.copysign(max_dr, r_new - r_prev)
                p_new = np.array([cx + rc * math.cos(thn), cy + rc * math.sin(thn)])
        step_vec = p_new - p
        if np.linalg.norm(step_vec) > 1e-6:
            d_new = step_vec / np.linalg.norm(step_vec)
            d = smooth * d + (1 - smooth) * d_new
            d /= np.linalg.norm(d)
        # enforce forward + limit radial steepness (decompose in local frame)
        th = math.atan2(p_new[1] - cy, p_new[0] - cx)
        rhat = np.array([math.cos(th), math.sin(th)])
        tdir = sgn * np.array([-math.sin(th), math.cos(th)])
        tang = float(np.dot(d, tdir)); rad = float(np.dot(d, rhat))
        tang = max(tang, min_fwd)
        rad = float(np.clip(rad, -max_rad, max_rad))
        d = tang * tdir + rad * rhat; d /= np.linalg.norm(d)
        a = th
        da = (a - a_prev + math.pi) % TWO_PI - math.pi
        cum += da; a_prev = a
        p = p_new; path.append(p.copy())
        if abs(cum) >= stop_extent:
            break
        if not (0 <= p[0] < W and 0 <= p[1] < H):
            break
    return np.array(path)


def dedup_paths(paths, cx, cy, min_sep):
    """Drop near-coincident walks (two windings collapsed onto one band). Sort by
    median radius; keep a path only if it's >= min_sep from the last kept one
    (keeping the LONGER of a close pair). Prevents two fits super close together."""
    if not paths:
        return paths
    rad = [float(np.median(np.hypot(p[:, 0] - cx, p[:, 1] - cy))) for p in paths]
    order = np.argsort(rad)
    kept, kept_r = [], -1e9
    for i in order:
        if rad[i] - kept_r >= min_sep:
            kept.append(paths[i]); kept_r = rad[i]
        elif len(kept[-1]) < len(paths[i]):           # replace with the longer one
            kept[-1] = paths[i]; kept_r = rad[i]
    return kept


def recenter_on_emulsion(path, emul, search=6.0, smooth_sigma=20.0):
    """Undo the inward bias a smoothing spline introduces on convex windings.

    Any low-pass smoother pulls a convex curve toward its chord -> systematically
    INNER (toward the roll center) of the emulsion, worst on the most-curved
    windings. This restores emulsion-centering WITHOUT re-adding wobble: for each
    point measure the signed perpendicular offset to the local class-2 centroid,
    LOW-PASS those offsets (the bias is systematic / low-frequency, the wobble is
    high-frequency and is discarded), then shift the smooth curve by the low-passed
    offset. Offset search stays < half the layer spacing so it locks to the OWN
    emulsion (can't jump to a neighbour). Coast gaps (no emulsion) are interpolated.
    """
    from scipy.ndimage import gaussian_filter1d
    H, W = emul.shape
    N = len(path)
    if N < 5:
        return path
    d = np.gradient(path, axis=0)
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)          # perpendicular per point
    ts = np.arange(-search, search + 1.0)                # (M,)
    # vectorized perpendicular sampling: (N, M) sample grid along each normal
    xs = path[:, 0:1] + ts[None, :] * nrm[:, 0:1]
    ys = path[:, 1:2] + ts[None, :] * nrm[:, 1:2]
    ix = np.clip(np.round(xs).astype(int), 0, W - 1)
    iy = np.clip(np.round(ys).astype(int), 0, H - 1)
    v = emul[iy, ix] > 0.5                               # (N, M) emulsion hits
    den = v.sum(1)
    with np.errstate(invalid="ignore"):
        off = np.where(den > 0, (ts[None, :] * v).sum(1) / den, np.nan)
    good = ~np.isnan(off)
    if good.sum() < 5:
        return path
    off = np.interp(np.arange(N), np.where(good)[0], off[good])   # fill coast gaps
    off = gaussian_filter1d(off, smooth_sigma, mode="nearest")    # keep only the bias
    return path + off[:, None] * nrm


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
                    help="Perpendicular snap search half-window (px); < half spacing.")
    ap.add_argument("--jump-max", type=float, default=4.0,
                    help="Max per-step perpendicular move (px) — no sudden radius jump.")
    ap.add_argument("--snap-accept", type=float, default=None,
                    help="If the nearest class-2 run is farther than this (px), coast "
                         "instead of snapping (only a neighbour is visible). "
                         "Default = snap-max (off). ~6 tightens against skips.")
    ap.add_argument("--max-dr", type=float, default=None,
                    help="Cap radius change per step (px) — anti-jump. ~2 blocks "
                         "band-to-band jumps while leaving eccentric drift free. "
                         "Default off.")
    ap.add_argument("--no-recenter", dest="recenter", action="store_false",
                    help="Disable post-smoothing re-centering onto the emulsion "
                         "(which removes the inward chording bias of the spline).")
    ap.add_argument("--recenter-search", type=float, default=6.0,
                    help="Perpendicular half-window (px) for emulsion re-centering "
                         "(< half the layer spacing).")
    ap.add_argument("--recenter-sigma", type=float, default=20.0,
                    help="Low-pass sigma (points) for the re-centering offset — large "
                         "keeps only the systematic bias, not wobble.")
    ap.add_argument("--no-seed-infill", dest="seed_infill", action="store_false",
                    help="Disable spacing-infill of dropped windings in film seeding.")
    ap.add_argument("--min-sep", type=float, default=8.0,
                    help="Drop walks whose median radius is within this of another.")
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
                                           r0, r1, args.film_min_thick, args.snap_max,
                                           infill=args.seed_infill)
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
                   args.coast_max, stop_extent, jump_max=args.jump_max,
                   snap_accept=args.snap_accept, max_dr=args.max_dr)
        bwd = walk(emul, p0, -tan_ccw, cx, cy, args.step, args.snap_max,
                   args.coast_max, stop_extent, jump_max=args.jump_max,
                   snap_accept=args.snap_accept, max_dr=args.max_dr)
        path = np.vstack([bwd[::-1], fwd[1:]])            # seam -> ... -> seam
        sm = smooth_path(path, args.smooth_px)
        if args.recenter:
            sm = recenter_on_emulsion(sm, emul, args.recenter_search,
                                      args.recenter_sigma)
        paths.append(sm)
    paths = dedup_paths(paths, cx, cy, args.min_sep)      # no two fits super close
    print(f"  {len(paths)} windings after dedup (min-sep {args.min_sep})")

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
