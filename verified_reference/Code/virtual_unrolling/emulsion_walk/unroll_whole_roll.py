"""Whole-roll unroll via the center-free EMULSION WALK (walk_emulsion.py).

Pipeline (every walked slice is an "anchor"):
  1. Per anchor: consensus seeding -> walk each winding along the emulsion (both
     directions, stop just short of the seam) -> bare-band rescue. Centre-free
     per-winding (x,y) paths; cached in walk_anchors.npz.
  2. Link windings through z by tracking their r(phi) profiles; drop low-coverage
     tracks; heal radial outliers (--z-heal-px); interpolate (x,y) to every CT slice
     (clamped outside the walked z-range).
  3. Put the strip columns on uniform film arc length (--no-arc-columns: uniform
     azimuth), sample the CT along each path (--transverse-px taps across the
     emulsion), concatenate windings inner->outer -> wholeroll.npy.

The walk (not a polar r(theta) fit) is what changed: no spool center, so eccentricity
is irrelevant, and each band is followed as a curve so curves can't wander between
windings. Anchor walks cached (slow seg load) -> `--use-cached-anchors` re-renders fast.

Usage:
  python -m virtual_unrolling.emulsion_walk.unroll_whole_roll \
      --ct-dir $D_SCAN/ct_slices --seg-dir <segmentation chunks> --out-dir <out> \
      [--inspect-anchors] [--use-cached-anchors] ...
  The exact flag sets used for the results are in Code/slurm/walk_0*.sh.
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from virtual_unrolling.common.spool_center import find_spool_center
from virtual_unrolling.common.volume_io import discover_volumes
from virtual_unrolling.common.volume_io import bilinear_sample
from virtual_unrolling.common.volume_io import (
    load_ct, norm_slice, discover_ct_slices, load_seg_slices,
)
from virtual_unrolling.emulsion_walk.walk import (
    film_seed_azimuth, walk, smooth_path, dedup_paths, recenter_on_emulsion,
)

TWO_PI = 2.0 * math.pi


def walk_anchor(seg2, cx, cy, r0, r1, film_min_thick, snap_max, step, coast_max,
                seam, eps, smooth_px, jump_max, snap_accept=None, max_dr=None,
                recenter_search=6.0, recenter_sigma=20.0, rescue_min_sep=6.0,
                max_infill=3):
    """All per-winding emulsion paths for one seg slice, each spanning the SAME
    azimuth range [seam+eps, seam+2pi-eps]. Walking to the seam (not +-a fixed
    extent from a per-anchor seed) makes arc/azimuth align across anchors, so the
    cross-z interpolation doesn't shear. Returns list of (N,2).

    Single-ray film-band seed + spacing infill; `max_dr` = anti-jump; after
    smoothing each path is re-centred onto the emulsion (undoes the smoothing
    spline's inward chording bias); unclaimed bands are rescued."""
    emul = (seg2 == 2).astype(np.float32)
    film = (seg2 > 0).astype(np.float32)
    seed_az, seeds = film_seed_azimuth(film, emul, cx, cy, r0, r1,
                                       film_min_thick, snap_max, infill=True,
                                       max_fill=max_infill)

    def walk_from(p0, az):
        """Seam-to-seam walk from any point/azimuth (seeds AND rescues)."""
        phi0 = float(np.mod(az - seam, TWO_PI))
        fwd_target = max(0.3, (TWO_PI - eps) - phi0)      # CCW to the seam
        bwd_target = max(0.3, phi0 - eps)                 # CW to the seam
        tan = np.array([-math.sin(az), math.cos(az)])
        fwd = walk(emul, p0, tan, cx, cy, step, snap_max, coast_max, fwd_target,
                   jump_max=jump_max, snap_accept=snap_accept, max_dr=max_dr)
        bwd = walk(emul, p0, -tan, cx, cy, step, snap_max, coast_max, bwd_target,
                   jump_max=jump_max, snap_accept=snap_accept, max_dr=max_dr)
        p = smooth_path(np.vstack([bwd[::-1], fwd[1:]]), smooth_px)
        return recenter_on_emulsion(p, emul, recenter_search, recenter_sigma)

    paths = [walk_from(np.array([cx + rs * math.cos(seed_az),
                                 cy + rs * math.sin(seed_az)]), seed_az)
             for rs in seeds]
    return rescue_bare_bands(paths, emul, cx, cy, seam, eps, rescue_min_sep, walk_from)


def median_radius(path, cx, cy):
    return float(np.median(np.hypot(path[:, 0] - cx, path[:, 1] - cy)))


def rescue_bare_bands(paths, emul, cx, cy, seam, eps, min_sep, walk_from,
                      gap_factor=1.6, cover_thresh=0.2, max_new=8, passes=3):
    """Walk film bands NO seed-ray walk claimed (the 2D fix for the 1D seed miss).

    Seeding is one ray, so a winding hidden there (touching pair, tongue) is never
    walked at this anchor. Detect it in 2D exactly like the eye does on the
    inspection overlay: sort walks by median radius; for each adjacent pair whose
    radial gap is >= gap_factor x the median gap, sample the emulsion mask in a
    perpendicular window midway between them along the whole turn. Sustained
    class-2 presence = a real unclaimed winding -> seed at the strongest contiguous
    stretch and walk it seam-to-seam with the normal machinery. Accept only walks
    that span most of the turn AND stay radially inside the gap (guards against the
    v5 partial-walk overseeding). Multi-missing gaps resolve over `passes`."""
    H, W = emul.shape
    added = 0
    K = 720
    phi_grid = np.linspace(eps, TWO_PI - eps, K)
    for _ in range(passes):
        cur = dedup_paths(list(paths), cx, cy, min_sep)    # radius-sorted
        if len(cur) < 3 or added >= max_new:
            break
        rads = [median_radius(p, cx, cy) for p in cur]
        gaps = np.diff(rads)
        med_gap = float(np.median(gaps))
        new_any = False
        for i in np.where(gaps >= gap_factor * med_gap)[0]:
            if added >= max_new:
                break
            Xa, Ya = resample_phi(cur[i], cx, cy, seam, phi_grid)
            Xb, Yb = resample_phi(cur[i + 1], cx, cy, seam, phi_grid)
            dx, dy = Xb - Xa, Yb - Ya
            dist = np.hypot(dx, dy) + 1e-9
            ux, uy = dx / dist, dy / dist
            mx, my = 0.5 * (Xa + Xb), 0.5 * (Ya + Yb)
            ts = np.linspace(-0.35, 0.35, 9)[None, :] * dist[:, None]
            sx = mx[:, None] + ts * ux[:, None]; sy = my[:, None] + ts * uy[:, None]
            ix = np.clip(np.round(sx).astype(int), 0, W - 1)
            iy = np.clip(np.round(sy).astype(int), 0, H - 1)
            hit = emul[iy, ix] > 0.5                       # (K, 9)
            present = hit.any(1)
            if present.mean() < cover_thresh:
                continue                                   # no unclaimed band here
            idx = np.where(present)[0]
            groups = np.split(idx, np.where(np.diff(idx) > 3)[0] + 1)
            g = max(groups, key=len)                       # longest visible stretch
            j = int(g[len(g) // 2])
            toff = float(ts[j][hit[j]].mean())             # radial centroid there
            p0 = np.array([mx[j] + toff * ux[j], my[j] + toff * uy[j]])
            az0 = math.atan2(p0[1] - cy, p0[0] - cx)
            newp = walk_from(p0, az0)
            ph = np.mod(np.arctan2(newp[:, 1] - cy, newp[:, 0] - cx) - seam, TWO_PI)
            span_ok = (ph.max() - ph.min()) >= 0.6 * (TWO_PI - 2 * eps)
            rmed = median_radius(newp, cx, cy)
            lo = rads[i] + 0.25 * gaps[i]; hi = rads[i + 1] - 0.25 * gaps[i]
            if span_ok and lo < rmed < hi:
                paths.append(newp); added += 1; new_any = True
        if not new_any:
            break
    if added:
        print(f"    rescued {added} bare-band winding(s)", flush=True)
    return paths


def _radius_profile(path, cx, cy, seam, K=96):
    """Coarse r(phi) profile of a walk (NaN outside the walk's phi span).

    The full profile — not the scalar median radius — is what identifies a
    winding: a PARTIAL walk's median is biased by eccentricity (up to ~half the
    winding spacing over half a turn), which fragments scalar tracking into
    near-duplicate tracks; profiles of the same winding agree to ~px wherever
    they overlap, while distinct windings differ >=17px everywhere."""
    th = np.arctan2(path[:, 1] - cy, path[:, 0] - cx)
    phi = np.mod(th - seam, TWO_PI)
    o = np.argsort(phi)
    r = np.hypot(path[o, 0] - cx, path[o, 1] - cy)
    grid = np.linspace(0.0, TWO_PI, K)
    return np.interp(grid, phi[o], r, left=np.nan, right=np.nan)


def dedup_paths_profile(paths, cx, cy, seam, thresh, K=96):
    """Profile-aware per-anchor dedup: two walks are the SAME winding iff their
    r(phi) profiles agree within `thresh` px on their phi overlap (keep the
    longer). Scalar-median dedup misses duplicates whose arcs differ (partial-walk
    eccentricity bias pushes the medians > min_sep apart) — measured on the dense
    cache as alternating twin tracks 7-10px apart. Real neighbours are >=17px
    apart everywhere, so thresh ~10px cannot merge two true windings."""
    if not paths:
        return paths
    profs = [_radius_profile(p, cx, cy, seam, K) for p in paths]
    order = np.argsort([float(np.nanmedian(pr)) for pr in profs])
    kept, kprof = [], []
    for i in order:
        dup = None
        for j in range(len(kept)):
            d = profs[i] - kprof[j]
            m = ~np.isnan(d)
            if m.sum() >= 8 and float(np.median(np.abs(d[m]))) < thresh:
                dup = j
                break
        if dup is None:
            kept.append(paths[i]); kprof.append(profs[i])
        elif len(paths[i]) > len(kept[dup]):
            kept[dup] = paths[i]; kprof[dup] = profs[i]
    return kept


def track_windings(PATHS, cx_a, cy_a, seam, tol, min_sep, K=96, min_overlap=12,
                   merge_px=12.0):
    """Reference-free winding matching: track windings across z-SORTED anchors by
    r(phi)-PROFILE chaining with coherent-drift compensation.

    Replaces the global union-clustering (which needed the ~25-anchor subsample to
    avoid merge-collapse, and so had NO slot for a winding walked only at
    non-subsampled anchors — measured cost: anchors walk 31-39 windings but the
    clustered reference held 29). Tracking compares only CONSECUTIVE anchors, where
    a winding moves ~a px, so slow whole-roll drift can't bridge the 17px winding
    gaps; every anchor contributes; assignment is one-to-one greedy by distance (no
    silent first-come drops). Distance = median |r_walk(phi) - r_track(phi)| over
    the common phi bins (scalar-median matching fragmented partial walks into
    62 tracks at 4-12px spacing on the dense cache). Per anchor the coherent
    radial drift (median signed profile offset of best pairs) is removed before
    matching; unmatched tracks keep drifting with the pack so they re-match after
    a miss streak. Fragment tracks are profile-merged at the end.

    Returns (matched[k][a] = path or None, ref_r sorted inner->outer)."""
    n = len(PATHS)
    tracks = []      # {'prof': (K,) latest profile, 'items': {a: path}, 'rs': [..]}
    for a in range(n):
        profs = [_radius_profile(p, cx_a[a], cy_a[a], seam, K) for p in PATHS[a]]
        if not profs:
            continue
        WP = np.array(profs)                                   # (nw, K)
        if tracks:
            TP = np.array([t["prof"] for t in tracks])         # (nt, K)
            D = WP[:, None, :] - TP[None, :, :]                # (nw, nt, K)
            ov = np.sum(~np.isnan(D), axis=2)
            with np.errstate(all="ignore"):
                signed = np.nanmedian(D, axis=2)
                dist = np.nanmedian(np.abs(D), axis=2)
            dist[ov < min_overlap] = np.inf
            # coherent drift = median signed offset of each walk's best track
            best = np.argmin(dist, axis=1)
            bd = dist[np.arange(len(WP)), best]
            good = bd < tol
            shift = float(np.median(signed[np.arange(len(WP))[good],
                                           best[good]])) if good.any() else 0.0
            with np.errstate(all="ignore"):
                dist = np.nanmedian(np.abs(D - shift), axis=2)
            dist[ov < min_overlap] = np.inf
            cand = sorted((dist[i, j], i, j)
                          for i in range(dist.shape[0])
                          for j in range(dist.shape[1]) if dist[i, j] <= tol)
            used_i, used_j = set(), set()
            for d, i, j in cand:
                if i in used_i or j in used_j:
                    continue
                used_i.add(i); used_j.add(j)
                tracks[j]["items"][a] = PATHS[a][i]
                # update profile: new walk's bins win, keep old where new is NaN
                tracks[j]["prof"] = np.where(np.isnan(WP[i]),
                                             tracks[j]["prof"], WP[i])
                tracks[j]["rs"].append(float(np.nanmedian(WP[i])))
            for j, t in enumerate(tracks):
                if j not in used_j:
                    t["prof"] = t["prof"] + shift    # stale tracks drift along
        else:
            used_i = set()
        for i in range(len(WP)):
            if i not in used_i:
                tracks.append({"prof": WP[i].copy(),
                               "items": {a: PATHS[a][i]},
                               "rs": [float(np.nanmedian(WP[i]))]})
    # merge fragments/twins of the same winding: profiles agree on their phi
    # overlap within merge_px (~half the 17px min spacing — two TRUE windings can
    # never be that close). The higher-coverage track wins conflicts (its walk is
    # kept where both have one), so alternating duplicate twins collapse onto the
    # dominant, better-covered geometry.
    tracks.sort(key=lambda t: float(np.median(t["rs"])))
    merged = []
    for t in tracks:
        target = None
        for mi in range(max(0, len(merged) - 3), len(merged)):
            m = merged[mi]                           # only radius-adjacent tracks
            d = t["prof"] - m["prof"]
            k = ~np.isnan(d)
            if k.sum() >= min_overlap and \
                    float(np.median(np.abs(d[k]))) < merge_px:
                target = mi
                break
            if k.sum() < min_overlap and \
                    abs(np.median(t["rs"]) - np.median(m["rs"])) < merge_px:
                target = mi                          # no overlap: scalar fallback
                break
        if target is not None:
            m = merged[target]
            big, small = ((m, t) if len(m["items"]) >= len(t["items"])
                          else (t, m))
            for a, p in small["items"].items():
                big["items"].setdefault(a, p)
            big["prof"] = np.where(np.isnan(big["prof"]), small["prof"],
                                   big["prof"])
            big["rs"] = big["rs"] + small["rs"]
            merged[target] = big
        else:
            merged.append(t)
    ref_r = np.array([float(np.median(t["rs"])) for t in merged])
    order = np.argsort(ref_r)
    matched = [[merged[o]["items"].get(a) for a in range(n)] for o in order]
    return matched, ref_r[order]


def resample_phi(path, cx, cy, seam, phi_grid):
    """Resample a path to a COMMON azimuth grid phi = wrap(theta - seam).

    Aligns the same film position (same azimuth) across anchors so the cross-z
    interpolation doesn't shear (arc-fraction did not align). Returns (X, Y)."""
    th = np.arctan2(path[:, 1] - cy, path[:, 0] - cx)
    phi = np.mod(th - seam, TWO_PI)
    o = np.argsort(phi); phi = phi[o]
    x = path[o, 0]; y = path[o, 1]
    return np.interp(phi_grid, phi, x), np.interp(phi_grid, phi, y)


def arclen_of(path):
    return float(np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1])).sum())


def arc_equalize(X, Y, cx_a, cy_a, phi_grid, px_per_col=1.0):
    """Re-grid a winding's columns from uniform AZIMUTH to uniform ARC LENGTH.

    THE v13 FIX. resample_phi puts the columns on a uniform phi grid, so each
    column covers ds/dphi = hypot(r, dr/dphi) px of film — and r swings by the
    winding's eccentricity (~80px ptp, measured on v12), so the film length per
    column swings with it. Measured on the v12 geometry: the local scale runs
    0.914..1.060 (+-5%), COHERENTLY across all 36 windings, once per turn; the
    along-film position error peaks at ~90px = 0.1 film cell per turn. On the
    delivered v12 strip the local cell pitch really does oscillate 886..913px
    with a one-winding period — i.e. the picture breathes and the framing creeps
    back and forth every ~6.7 frames. That is a PARAMETERIZATION error, not a
    fit error (the fitted curves are smooth to 0.23px rms).

    Fix: choose the columns so each one spans the same ARC LENGTH of film, using
    the Z-MEDIAN radius profile. Because the resulting phi-per-column map is
    computed once and shared by every z, the cross-z alignment that resample_phi
    exists to provide is preserved EXACTLY — this only changes where the columns
    sit along the turn, not which film position a column means.

    Returns (phi_new, X2, Y2)."""
    if X.shape[1] < 4:
        return phi_grid, X, Y
    # Arc length of the z-MEDIAN CURVE, measured as the polyline it actually is.
    # NOT the analytic hypot(r, dr/dphi)*dphi: resample_phi CLAMPS a partial
    # walk's out-of-span columns to its endpoint, and the analytic form reads
    # that repeated point as r*dphi of film that does not exist — measured at
    # +2.7% of the whole roll, all of it from the two film-end fragment windings
    # (w0 51% clamped, w1 93%), which inflated the strip to 240 cells vs GT ~233.
    # A polyline has zero length across a repeated point, so it is immune. It is
    # also center-free, like the walk itself.
    #
    # The curve is lightly smoothed FIRST: columns are ~1px apart, so raw
    # per-column steps are dominated by sub-pixel jitter, and |jitter| only ever
    # ADDS length (a biased over-estimate). Smoothing at sigma 15 columns is far
    # below the once-per-turn modulation we are correcting (~6000 columns) and
    # its chord-shortening on a ~1000px-radius curve is ~4e-5 — negligible.
    from scipy.ndimage import gaussian_filter1d
    xm = gaussian_filter1d(np.median(X, axis=0).astype(float), 15, mode="nearest")
    ym = gaussian_filter1d(np.median(Y, axis=0).astype(float), 15, mode="nearest")
    step = np.hypot(np.diff(xm), np.diff(ym))
    s = np.concatenate([[0.0], np.cumsum(step)])
    if s[-1] <= 0:
        return phi_grid, X, Y
    K2 = max(2, int(round(s[-1] / max(1e-6, px_per_col))))
    phi_new = np.interp(np.linspace(0.0, s[-1], K2), s, phi_grid)
    n = X.shape[0]
    X2 = np.empty((n, K2), X.dtype); Y2 = np.empty((n, K2), Y.dtype)
    for a in range(n):
        X2[a] = np.interp(phi_new, phi_grid, X[a])
        Y2[a] = np.interp(phi_new, phi_grid, Y[a])
    return phi_new, X2, Y2


def z_heal_windings(kept, cx_a, cy_a, seam, eps_phi, thresh, window=61,
                    max_range=150.0):
    """Repair mid-turn winding JUMPS by cross-z consensus (needs dense anchors).

    A walk that jumps onto a neighbour partway leaves the rest of its own turn on
    the wrong band — a radial outlier of ~one spacing (>=17px) at those azimuths.
    Its z-neighbours almost never jump at the same azimuth (the dense cache has
    ~real z-neighbours a few slices away), so per (winding, phi bin): take the
    running z-MEDIAN of the radius, flag |r - median| > thresh, and replace
    flagged bins by interpolating the winding's OWN radius across clean anchors.
    `window` spans ~3 chunks so a jump repeated across one chunk's near-identical
    slices can't poison its own consensus.

    Rebuilt bins are placed on their column's RAY (theta = seam + phi_grid[c]),
    never at the stored point's angle: a PARTIAL walk's resample_phi output clamps
    every column beyond its span to the endpoint (x,y), so the stored angle there
    is the endpoint azimuth — rebuilding with it swept the interpolated radii
    along one ray = stray radial streaks crossing the roll (seen at z1749).
    Clamped bins are detected by azimuth mismatch and healed too, which EXTENDS
    partial walks with the winding's own cross-z geometry."""
    from scipy.ndimage import median_filter
    n = len(cx_a)
    cxA = np.asarray(cx_a, float)[:, None]
    cyA = np.asarray(cy_a, float)[:, None]
    w = min(window, n if n % 2 == 1 else n - 1)
    if w < 5:
        return kept, 0
    healed, out = 0, []
    for wi, (r0, X, Y, phi) in enumerate(kept):
        # NEVER heal the innermost/outermost winding: they carry the film's two
        # ENDS (inner tongue / outer termination), whose geometry is legitimately
        # partial, non-circular and strongly z-varying — healing them drew chords
        # across the void (the tail is walked at only a minority of anchors, so
        # the median-range gate below can miss it while the clamp-extension
        # quorum still fires on garbage). Both are cut from the video anyway.
        if wi == 0 or wi == len(kept) - 1:
            out.append((r0, X, Y, phi))
            continue
        # the column's ray angle — read from the winding's OWN phi grid, which
        # v13 no longer assumes is uniform (see arc_equalize).
        th = (seam + np.asarray(phi, float))[None, :]
        R = np.hypot(X - cxA, Y - cyA)
        # heal assumes a ~circular band (r(phi) smooth, azimuth ~= column ray,
        # z-drift ~px). Skip windings whose per-anchor radius RANGE says "not a
        # band" (normal: ~2x eccentricity <= ~90px); p85 so a minority of anchors
        # with runaway geometry still trips the gate.
        if float(np.percentile(R.max(axis=1) - R.min(axis=1), 85)) > max_range:
            out.append((r0, X, Y, phi))
            continue
        th_actual = np.arctan2(Y - cyA, X - cxA)
        dphi = np.abs((th_actual - th + np.pi) % TWO_PI - np.pi)
        # skip mostly-CLAMPED fragment windings (short partial stubs, e.g. the
        # sliver tracks near the tongue): healing their few on-ray bins scatters
        # points around the turn and the connecting polyline/render jumps across
        # the roll (the z1749 chord). Their v8-style stub rendering is correct.
        if float((dphi <= math.radians(3.0)).mean()) < 0.5:
            out.append((r0, X, Y, phi))
            continue
        Rm = median_filter(R, size=(w, 1), mode="nearest")
        # radius-jumped bins (on-ray) vs endpoint-clamped bins of partial walks
        bad_a = dphi > math.radians(3.0)
        bad_r = (np.abs(R - Rm) > thresh) & ~bad_a
        heal = np.zeros(R.shape, bool)
        # clamp-EXTENSION needs a real quorum: a partial winding (inner tongue /
        # outer film end) has NO geometry at some azimuths at ANY z — there the
        # few stray walks that do reach are junk, and extending 500 anchors from
        # them drew chords across the void. Jump repair only needs a few clean
        # anchors (jumps are the outliers among mostly-good ones).
        a_need = max(10, int(0.3 * n))
        for c in np.where((bad_r | bad_a).any(axis=0))[0]:
            b = bad_r[:, c] | bad_a[:, c]
            good = np.where(~b)[0]
            if good.size >= a_need:
                m = b                        # heal jumps + extend clamped bins
            elif good.size >= 3:
                m = bad_r[:, c]              # heal jumps only; leave clamp stubs
            else:
                continue
            if not m.any():
                continue
            R[m, c] = np.interp(np.where(m)[0], good, R[good, c])
            heal[:, c] = m
        if heal.any():
            X = np.where(heal, (cxA + R * np.cos(th)).astype(X.dtype), X)
            Y = np.where(heal, (cyA + R * np.sin(th)).astype(Y.dtype), Y)
            healed += int(heal.sum())
        out.append((r0, X, Y, phi))
    return out, healed


def _z_map(z_anchor, A):
    """Per-anchor array A (n, K) -> function of z: linear between anchors, held at
    the nearest walked anchor outside their z range (perforation rows, margins)."""
    return interp1d(z_anchor, A, axis=0, bounds_error=False, fill_value=(A[0], A[-1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ct-dir", required=True)
    ap.add_argument("--seg-dir", required=True,
                    help="segmentation chunks (required by the parser even with "
                         "--use-cached-anchors)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seam-deg", type=float, required=True,
                    help="seam azimuth (deg): the film's outer end, a physical constant "
                         "of the roll (353 on the full-width scan)")
    ap.add_argument("--seam-exclude-deg", type=float, default=0.3,
                    help="walks stop this far short of the seam on each side")
    # walk phase (changing these needs a re-walk)
    ap.add_argument("--smooth-px", type=float, default=0.6)
    ap.add_argument("--snap-max", type=float, default=8.0)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--coast-max", type=int, default=40,
                    help="steps the walk may coast through a gap with no emulsion")
    ap.add_argument("--film-min-thick", type=float, default=6.0)
    ap.add_argument("--jump-max", type=float, default=4.0)
    ap.add_argument("--snap-accept", type=float, default=None,
                    help="coast if the nearest emulsion run is farther than this (px)")
    ap.add_argument("--max-dr", type=float, default=None,
                    help="cap on the radius change per walk step (px): anti-jump")
    ap.add_argument("--max-infill", type=int, default=3,
                    help="max windings one seed gap may claim to hide (wider = void)")
    ap.add_argument("--radial-robust", action="store_true",
                    help="seeding radial range from azimuthal coverage, so bore "
                         "artefacts cannot set the inner radius")
    ap.add_argument("--n-centers", type=int, default=30,
                    help="total find_spool_center calls over the stack, interpolated "
                         "to every anchor")
    ap.add_argument("--recenter-search", type=float, default=6.0)
    ap.add_argument("--recenter-sigma", type=float, default=20.0)
    # tracking / healing / rendering (cache-fast)
    ap.add_argument("--min-sep", type=float, default=6.0)
    ap.add_argument("--dedup-px", type=float, default=10.0,
                    help="per-anchor r(phi)-profile dedup threshold (px)")
    ap.add_argument("--track-merge-px", type=float, default=12.0,
                    help="merge tracks whose profiles agree within this (px)")
    ap.add_argument("--match-tol-px", type=float, default=12.0,
                    help="max radius difference to link a winding across anchors")
    ap.add_argument("--min-coverage", type=int, default=8,
                    help="drop a winding tracked on fewer anchors than this")
    ap.add_argument("--z-heal-px", type=float, default=6.0,
                    help="re-interpolate (anchor, phi) bins deviating more than this "
                         "from the winding's running z-median (a jump to a neighbour); "
                         "0 = off")
    ap.add_argument("--no-arc-columns", dest="arc_columns", action="store_false",
                    help="strip columns on uniform azimuth instead of uniform film "
                         "arc length")
    ap.add_argument("--transverse-px", type=int, default=2,
                    help="CT taps averaged across the emulsion thickness (1 = point)")
    ap.add_argument("--seam-bridge", action="store_true",
                    help="fill the 2*seam-exclude wedge at each winding join with a "
                         "linear blend of both sides")
    ap.add_argument("--use-cached-anchors", action="store_true")
    ap.add_argument("--inspect-anchors", action="store_true",
                    help="after tracking: save matched_walks.npz + overlay montage, exit")
    ap.add_argument("--inspect-max", type=int, default=0,
                    help="with --inspect-anchors: draw only this many anchors (0 = all)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    eps_phi = math.radians(args.seam_exclude_deg)
    cache = os.path.join(args.out_dir, "walk_anchors.npz")

    # ── anchor walks (slow seg load; cached) ──
    if args.use_cached_anchors and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        z_anchor = d["z_anchor"]; cx_a = list(d["cx_a"]); cy_a = list(d["cy_a"])
        PATHS = list(d["paths"]); seam = float(d["seam"])
        seam = math.radians(args.seam_deg)
        print(f"loaded cached walks for {len(z_anchor)} anchors")
    else:
        pairs = discover_volumes(args.seg_dir)
        seam = math.radians(args.seam_deg)
        print(f"{len(pairs)} anchors; seam={args.seam_deg:.1f}deg (forced)")
        import time
        # ── SPOOL CENTRES: `--n-centers` calls over the WHOLE stack (not per chunk),
        #    evenly spread in z, then interpolated to every slice. The centre drifts
        #    smoothly and slowly along the roll, so a few well-separated samples
        #    describe it; per-chunk sampling would cost ~2.5min x 3 x n_chunks (~9h
        #    on the 72-chunk new scan) and add per-chunk noise, since a single
        #    find_spool_center call scatters by tens of px (measured: a 20-slice
        #    chunk gave (1893,1843), (1879,1904), (1870,1902)). ──
        n_ctr = max(1, min(args.n_centers, len(pairs)))
        pick = set(np.unique(np.linspace(0, len(pairs) - 1,
                                        n_ctr).astype(int)).tolist())
        # Sampled chunks are centred FROM THE CHUNK ALREADY IN MEMORY during the walk
        # loop -- a separate pre-pass would re-read ~3min of probabilities per sampled
        # chunk (~90min of pure I/O at 30 samples). Inside the loop each chunk walks
        # with the latest sample as a provisional centre (the walk is centre-FREE
        # after seeding: the centre only sets the seed ray's radial window, so tens of
        # px are harmless there); the per-anchor centres are then REPLACED by the
        # interpolation over all samples once they exist, before anything that is
        # centre-sensitive runs (median-radius tracking, resample_phi's common azimuth
        # grid, arc_equalize).
        cz, ccy, ccx = [], [], []
        t_c0 = time.time()

        z_anchor, cx_a, cy_a, PATHS = [], [], [], []
        for ci, (vp, pp, z0, z1) in enumerate(pairs):
            t0 = time.time()
            slices = load_seg_slices(pp, 1.0)          # every slice of the chunk
            t_load = time.time() - t0
            t_ctr = 0.0
            if ci in pick:                     # centre this chunk (already loaded)
                t1 = time.time()
                s_mid, si_mid = slices[len(slices) // 2]
                cyi, cxi = find_spool_center(s_mid)
                cz.append(float(z0 + si_mid)); ccy.append(cyi); ccx.append(cxi)
                t_ctr = time.time() - t1
                print(f"    centre sample {len(cz)}/{n_ctr} at z{cz[-1]:.0f}: "
                      f"{cyi:.1f},{cxi:.1f} ({t_ctr:.0f}s)", flush=True)
            if not cz:                          # first chunk is always in `pick`
                raise RuntimeError("no centre sample yet -- pick must include chunk 0")
            cy_prov, cx_prov = ccy[-1], ccx[-1]
            t_walk = 0.0
            for si_i, (seg2, sidx) in enumerate(slices):
                cy, cx = cy_prov, cx_prov
                ys, xs = np.where(seg2 > 0); rf = np.hypot(ys - cy, xs - cx)
                if args.radial_robust:
                    # r0/r1 from AZIMUTHAL COVERAGE, not min/max over all labelled
                    # pixels: a handful of bore artefacts (or noise outside the roll)
                    # otherwise drags r0 to ~13px and the seeding then fills the
                    # empty core with fake windings. The film annulus is where a
                    # decent fraction of azimuths carry film.
                    nb = 200
                    edges = np.linspace(0, rf.max() + 1, nb + 1)
                    which = np.clip(np.digitize(rf, edges) - 1, 0, nb - 1)
                    th = np.arctan2(ys - cy, xs - cx)
                    abin = np.clip(((th + math.pi) / TWO_PI * 72).astype(int), 0, 71)
                    cov = np.zeros(nb)
                    seen = np.zeros((nb, 72), bool)
                    seen[which, abin] = True
                    cov = seen.mean(axis=1)
                    hit = np.nonzero(cov > 0.3)[0]
                    if hit.size >= 2:
                        r0_c = float(edges[hit[0]]); r1_c = float(edges[hit[-1] + 1])
                        if si_i == 0:
                            print(f"    radial range: coverage>0.3 -> "
                                  f"r {r0_c:.0f}-{r1_c:.0f} (raw min/max "
                                  f"{rf.min():.0f}-{rf.max():.0f})", flush=True)
                        rf = rf[(rf >= r0_c) & (rf <= r1_c)]
                tw = time.time()
                paths = walk_anchor(seg2, cx, cy, rf.min() + 5, rf.max() - 5,
                                    args.film_min_thick, args.snap_max, args.step,
                                    args.coast_max, seam, eps_phi, args.smooth_px,
                                    args.jump_max, snap_accept=args.snap_accept,
                                    max_dr=args.max_dr,
                                    recenter_search=args.recenter_search,
                                    recenter_sigma=args.recenter_sigma,
                                    rescue_min_sep=args.min_sep,
                                    max_infill=args.max_infill)
                t_walk += time.time() - tw
                z_anchor.append(z0 + sidx); cx_a.append(cx); cy_a.append(cy)
                PATHS.append(paths)
            print(f"  chunk z{z0}-{z1}: {len(slices)} sl | load {t_load:.0f}s "
                  f"center {t_ctr:.0f}s walk {t_walk:.0f}s", flush=True)
        z_anchor = np.array(z_anchor, float)
        # ── final centres: interpolate the samples over z and REPLACE the provisional
        #    per-anchor values (see the note above). Done before tracking /
        #    resample_phi / arc_equalize, all of which are centre-sensitive. ──
        cz = np.array(cz, float); ccy = np.array(ccy); ccx = np.array(ccx)
        if len(cz) >= 2:
            cy_a = list(np.interp(z_anchor, cz, ccy))
            cx_a = list(np.interp(z_anchor, cz, ccx))
        else:
            cy_a = [float(ccy[0])] * len(z_anchor)
            cx_a = [float(ccx[0])] * len(z_anchor)
        msg = (f"  centres: {len(cz)} samples over z{cz[0]:.0f}-{cz[-1]:.0f} in "
               f"{time.time() - t_c0:.0f}s, interpolated to {len(z_anchor)} anchors; "
               f"end-to-end drift "
               f"{np.hypot(ccy[-1] - ccy[0], ccx[-1] - ccx[0]):.1f}px")
        if len(cz) >= 3:
            A = np.stack([cz, np.ones_like(cz)], 1)
            ry = ccy - A @ np.linalg.lstsq(A, ccy, rcond=None)[0]
            rx = ccx - A @ np.linalg.lstsq(A, ccx, rcond=None)[0]
            msg += (f"; max residual from a straight line in z "
                    f"{max(np.abs(ry).max(), np.abs(rx).max()):.1f}px (large = an "
                    f"outlier sample; a single find_spool_center call scatters by "
                    f"tens of px on the new scan)")
        print(msg, flush=True)
        np.savez(cache, z_anchor=z_anchor, cx_a=cx_a, cy_a=cy_a, seam=seam,
                 paths=np.array(PATHS, dtype=object))

    # sort anchors by z and SMOOTH the per-chunk centers across z. find_spool_center
    # is noisy chunk-to-chunk; that jitter smears each winding's median radius, which
    # collapses the union-clustering once anchors are dense (many z per chunk -> the
    # jitter bridges the ~22px winding gaps). The true center drifts smoothly, so a
    # z-gaussian recovers stable radii. (Mid-only 25-anchor runs are left untouched.)
    order = np.argsort(np.asarray(z_anchor, float))
    z_anchor = np.asarray(z_anchor, float)[order]
    cx_a = list(np.asarray(cx_a, float)[order]); cy_a = list(np.asarray(cy_a, float)[order])
    PATHS = [PATHS[i] for i in order]
    if len(z_anchor) >= 50:
        from scipy.ndimage import gaussian_filter1d
        sig = max(1.0, len(z_anchor) / 25.0)            # ~1 chunk width
        cx_a = list(gaussian_filter1d(np.asarray(cx_a), sig, mode="nearest"))
        cy_a = list(gaussian_filter1d(np.asarray(cy_a), sig, mode="nearest"))
        print(f"  smoothed per-chunk centers across z (sigma {sig:.0f} anchors)")

    n = len(z_anchor)
    z_to_path = discover_ct_slices(args.ct_dir)

    # dedup near-coincident walks per anchor (tunable here, no re-walk needed).
    # track mode: PROFILE dedup (scalar medians miss duplicate walks with
    # different arcs — the alternating-twin-track source)
    PATHS = [dedup_paths_profile(list(PATHS[a]), cx_a[a], cy_a[a], seam,
                                 args.dedup_px) for a in range(n)]
    print(f"  per-anchor winding counts after dedup(min-sep {args.min_sep}): "
          f"{[len(PATHS[a]) for a in range(n)]}")

    # ── z-sequential tracking: consecutive-anchor chaining, drift-compensated,
    #    one-to-one greedy assignment; every anchor contributes (no subsample) ──
    matched, ref_r = track_windings(PATHS, cx_a, cy_a, seam,
                                    args.match_tol_px, args.min_sep,
                                    merge_px=args.track_merge_px)
    nw = len(ref_r)
    print(f"  z-sequential tracking -> {nw} winding tracks")
    # ── final SPACING dedup: real windings are >=17px apart everywhere, so any
    #    two tracks whose median radii are < min_sep apart are a SPLIT of one
    #    winding that survived tracking (more common at high anchor density —
    #    the full-1184 walk left r647/r648 and r741/r741 pairs). Merge the
    #    smaller-coverage track into the larger (fill its None anchor slots),
    #    keeping the better geometry. ──
    order = np.argsort(ref_r)
    keep_m, keep_r = [], []
    for k in order:
        if keep_r and ref_r[k] - keep_r[-1] < args.min_sep:
            big, small = keep_m[-1], matched[k]
            if sum(p is not None for p in small) > \
                    sum(p is not None for p in big):
                big, small = small, list(big)
            for a in range(n):
                if big[a] is None and small[a] is not None:
                    big[a] = small[a]
            keep_m[-1] = big
            keep_r[-1] = float(np.median([median_radius(big[a], cx_a[a], cy_a[a])
                                          for a in range(n) if big[a] is not None]))
        else:
            keep_m.append(list(matched[k])); keep_r.append(float(ref_r[k]))
    if len(keep_r) < nw:
        print(f"  spacing-dedup: {nw} -> {len(keep_r)} windings "
              f"(merged {nw - len(keep_r)} split-track duplicate(s) < "
              f"{args.min_sep}px)")
    matched, ref_r = keep_m, np.array(keep_r); nw = len(ref_r)
    covered = [sum(p is not None for p in matched[k]) for k in range(nw)]
    print(f"  {nw} windings (track); per-winding "
          f"anchor coverage min/med/max = "
          f"{min(covered)}/{int(np.median(covered))}/{max(covered)}")
    diffs = np.diff(ref_r)
    print(f"  ref_r spacing px: min {diffs.min():.0f} med {np.median(diffs):.0f} "
          f"max {diffs.max():.0f}  (a ~2x-median gap = a winding NEVER walked "
          f"anywhere = a seeding miss)")
    print(f"  ref_r (inner->outer): {[int(r) for r in ref_r]}")
    print(f"  coverage per winding: {covered}")

    # ── build kept windings (union + min-coverage) as per-anchor resampled (X,Y) ──
    kept = []                                          # (r_med, X(n,K), Y(n,K))
    dropped = []                                       # (r_med, coverage)
    for k in range(nw):
        present = [a for a in range(n) if matched[k][a] is not None]
        if len(present) < max(2, args.min_coverage):   # drop spurious low-cov windings
            dropped.append((float(ref_r[k]), len(present)))
            continue
        K = max(2, int(round(np.median([arclen_of(matched[k][a]) for a in present]))))
        phi_grid = np.linspace(eps_phi, TWO_PI - eps_phi, K)   # COMMON azimuth grid
        X = np.empty((n, K), np.float32); Y = np.empty((n, K), np.float32)
        for a in range(n):
            b = a if matched[k][a] is not None else \
                min(present, key=lambda p: abs(p - a))     # fill gaps from nearest z
            X[a], Y[a] = resample_phi(matched[k][b], cx_a[b], cy_a[b], seam, phi_grid)
        kept.append((float(ref_r[k]), X, Y, phi_grid))
    kept.sort(key=lambda t: t[0])
    print(f"  {len(kept)} windings kept (coverage >= {args.min_coverage})")
    # Report the dropped tracks next to the radius gaps they leave behind: a real,
    # merely under-covered track inside a wide gap means --min-coverage is too high.
    if dropped:
        dropped.sort()
        radii = [t[0] for t in kept]
        med_g = float(np.median(np.diff(radii))) if len(radii) > 2 else 0.0
        print(f"  {len(dropped)} tracks dropped below coverage {args.min_coverage}"
              f" (median kept spacing {med_g:.1f}px):")
        for r, c in dropped:
            inside = ""
            for i in range(1, len(radii)):
                if radii[i - 1] < r < radii[i] and med_g > 0 and \
                        (radii[i] - radii[i - 1]) > 1.5 * med_g:
                    inside = (f"   <-- INSIDE a {radii[i]-radii[i-1]:.0f}px gap "
                              f"(r {radii[i-1]:.0f}-{radii[i]:.0f})")
                    break
            print(f"    r {r:7.1f}  coverage {c:5d}/{n}{inside}")

    # ── z-heal (v9): repair mid-turn winding jumps by cross-z radius consensus ──
    if args.z_heal_px > 0 and n >= 7:
        kept, nbins = z_heal_windings(kept, cx_a, cy_a, seam, eps_phi,
                                      args.z_heal_px)
        print(f"  z-heal: repaired {nbins} jumped (anchor,phi) bins "
              f"(>{args.z_heal_px}px from running z-median)")

    # ── v13: ARC-LENGTH columns (the fix — see arc_equalize). Runs LAST, on the
    #    final healed/smoothed geometry, so the profile it equalizes is the one
    #    actually rendered. ──
    if args.arc_columns:
        eq, before = [], sum(t[1].shape[1] for t in kept)
        stretch = []
        for (r, X, Y, p) in kept:
            xm = np.median(X, axis=0).astype(float)
            ym = np.median(Y, axis=0).astype(float)
            ds = np.hypot(np.diff(xm), np.diff(ym))     # px of film per column
            core = ds[len(ds) // 20: -len(ds) // 20] if len(ds) > 40 else ds
            core = core[core > 1e-6]                    # ignore clamped (zero-length)
            if core.size:
                stretch.append(float(np.percentile(core, 99) /
                                     max(np.percentile(core, 1), 1e-6)))
            p2, X2, Y2 = arc_equalize(X, Y, cx_a, cy_a, p, 1.0)
            eq.append((r, X2, Y2, p2))
        kept = eq
        after = sum(t[1].shape[1] for t in kept)
        print(f"  arc-columns: uniform-phi -> uniform arc length "
              f"(1 px/col); cols {before} -> {after}. "
              f"Removed a within-turn stretch of med "
              f"{100 * (np.median(stretch) - 1) / 2:+.1f}% / max "
              f"{100 * (max(stretch) - 1) / 2:+.1f}% (peak-to-mean).")
    total_cols = sum(t[1].shape[1] for t in kept)
    print(f"  whole roll: {len(kept)} windings, {total_cols} cols")

    # ── inspect: draw the delivered windings on each anchor CT ──
    if args.inspect_anchors:
        from PIL import Image as _Im
        mp = np.empty(n, object)
        mp[:] = [[np.column_stack([X[a], Y[a]]) for (_, X, Y, _p) in kept] for a in range(n)]
        np.savez(os.path.join(args.out_dir, "matched_walks.npz"),
                 z_anchor=z_anchor, cx_a=cx_a, cy_a=cy_a, seam=seam, paths=mp)
        ins_idx = (range(n) if args.inspect_max <= 0 else
                   np.unique(np.linspace(0, n - 1, args.inspect_max).astype(int)))
        cmap = plt.get_cmap("hsv"); tiles = []
        for a in ins_idx:
            za = min(z_to_path, key=lambda zz: abs(zz - z_anchor[a]))
            img = norm_slice(load_ct(z_to_path[za]))
            fig, ax = plt.subplots(1, 1, figsize=(7, 7)); ax.imshow(img, cmap="gray")
            for kk, (_, X, Y, _p) in enumerate(kept):
                ax.plot(X[a], Y[a], "-", color=cmap(kk / max(1, len(kept))), lw=0.5)
            ax.set_title(f"z{za} — {len(kept)}w", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            pth = os.path.join(args.out_dir, f"insp_{a:02d}_z{za}.png")
            plt.tight_layout(); plt.savefig(pth, dpi=110); plt.close(); tiles.append(pth)
            print(f"  drew anchor {a} z{za}", flush=True)
        ims = [_Im.open(t).convert("RGB").resize((480, 480)) for t in tiles]
        cols = 5; rows = int(np.ceil(len(ims) / cols))
        mon = _Im.new("RGB", (480 * cols, 480 * rows), "white")
        for j, im in enumerate(ims):
            mon.paste(im, ((j % cols) * 480, (j // cols) * 480))
        mon.save(os.path.join(args.out_dir, "walk_anchors_montage.png"))
        for t in tiles:
            os.remove(t)
        print(f"Done (inspect). montage in {args.out_dir}")
        return

    # ── render: sample raw CT along each winding path (x,y) over all z ──
    win = [(_z_map(z_anchor, X), _z_map(z_anchor, Y), X.shape[1])
           for (_, X, Y, _p) in kept]
    z_list = sorted(z_to_path)
    strip = np.empty((len(z_list), total_cols), dtype=np.float32)
    # TRANSVERSE AVERAGING across the emulsion's own thickness. The render
    # otherwise reads a single infinitesimally-thin line through the middle of a
    # band measured at 4.0px median (p10 3.2, p90 5.0) on real Mickey -- note
    # that is the CLASS-2 EMULSION; the ~18px figure is the class>=1 FILM layer
    # (base + emulsion). Averaging n taps along the path NORMAL integrates
    # through the emulsion the way a densitometer reads film: ~sqrt(n) better
    # SNR, and far less sensitive to sub-pixel path error or to exactly where
    # the segmentation puts the band centre. Risk: where the emulsion is tilted
    # w.r.t. the path normal, or where taps spill past a 3.2px band into the
    # base, this smears along the film -- hence the width sweep.
    tw = max(1, int(args.transverse_px))
    offs = np.linspace(-(tw - 1) / 2.0, (tw - 1) / 2.0, tw)
    if tw > 1:
        print(f"  transverse averaging: {tw} taps at offsets "
              f"{np.round(offs, 2).tolist()} px along the path normal")
    for j, z in enumerate(z_list):
        img = norm_slice(load_ct(z_to_path[z]))
        c = 0
        for fX, fY, K in win:
            xs = fX(z); ys = fY(z)
            if tw == 1:
                strip[j, c:c + K] = bilinear_sample(img, xs, ys)
            else:
                dx = np.gradient(xs); dy = np.gradient(ys)
                L = np.hypot(dx, dy) + 1e-9
                nx, ny = -dy / L, dx / L           # unit normal, per column
                acc = np.zeros(K, np.float32)
                for t in offs:
                    acc += bilinear_sample(img, xs + t * nx, ys + t * ny)
                strip[j, c:c + K] = acc / tw
            c += K
        if j % 200 == 0 or j == len(z_list) - 1:
            print(f"  [{j + 1}/{len(z_list)}] z{z}", flush=True)

    # ── seam bridge: fill the residual 2*eps gap at each winding boundary with a
    #    linear blend of the two (continuous-film) sides -> ~0 visible join ──
    if args.seam_bridge and eps_phi > 0 and len(win) >= 2:
        Ks = [K for _, _, K in win]
        starts = np.cumsum([0] + Ks)
        pieces, n_bridge = [], 0
        for i, K in enumerate(Ks):
            pieces.append(strip[:, starts[i]:starts[i] + K])
            if i < len(Ks) - 1:                       # not after the outermost
                if args.arc_columns:
                    # arc columns are 1 px of FILM each, so the missing
                    # wedge is simply its arc length r_seam*2*eps.
                    # (The uniform-phi formula below would over/under-fill by the
                    # same +-5% the arc re-grid just removed.)
                    _, Xk, Yk, _p = kept[i]
                    r_seam = float(np.median(np.hypot(
                        Xk[:, -1] - np.asarray(cx_a, float),
                        Yk[:, -1] - np.asarray(cy_a, float))))
                    M = int(round(r_seam * 2 * eps_phi))
                else:
                    # missing arc = 2*eps rad; match cols/rad = K/(2pi-2eps)
                    M = int(round(2 * eps_phi * K / (TWO_PI - 2 * eps_phi)))
                if M >= 1:
                    last = strip[:, starts[i] + K - 1:starts[i] + K]     # (Z,1)
                    nxt = strip[:, starts[i + 1]:starts[i + 1] + 1]      # (Z,1)
                    w = np.linspace(0.0, 1.0, M + 2)[1:-1][None, :]      # (1,M)
                    pieces.append((last * (1 - w) + nxt * w).astype(np.float32))
                    n_bridge += M
        strip = np.concatenate(pieces, axis=1).astype(np.float32)
        total_cols = strip.shape[1]
        print(f"  seam-bridge: inserted {n_bridge} blended cols across "
              f"{len(Ks) - 1} winding boundaries -> strip {strip.shape}")

    np.save(os.path.join(args.out_dir, "wholeroll.npy"), strip)
    np.save(os.path.join(args.out_dir, "z_index.npy"), np.array(z_list))
    from PIL import Image as _Im
    stepc = max(1, total_cols // 6000)
    sm = strip[:, ::stepc]; lo, hi = np.percentile(sm, [1, 99])
    sm = np.clip((sm - lo) / (hi - lo + 1e-6), 0, 1)
    _Im.fromarray((sm * 255).astype(np.uint8)).save(
        os.path.join(args.out_dir, "wholeroll_overview.png"))
    print(f"Done. whole roll {strip.shape}. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
