"""Emulsion walk: trace one film winding as an ordered curve on a segmented slice.

  1. SEED (film_seed_azimuth): on the most complete radial ray, count the
     connected film runs (merged runs split at their interior emulsion runs,
     missing windings in-filled from the local spacing) and snap one seed per
     winding onto the class-2 emulsion.
  2. WALK (walk): predictor-corrector along the emulsion in one direction: step
     along the local tangent, snap perpendicular onto the nearest separate
     emulsion run (window < half the winding spacing), coast across short gaps.
  3. smooth_path / recenter_on_emulsion: smoothing spline, then re-lock onto the
     emulsion centre line; dedup_paths removes near-coincident walks.

The walk is local, so the spool's eccentricity is irrelevant, and each band is
followed as a curve, so a curve cannot wander onto a neighbouring winding.
Driven by unroll_whole_roll.walk_anchor.
"""
import math

import numpy as np

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
