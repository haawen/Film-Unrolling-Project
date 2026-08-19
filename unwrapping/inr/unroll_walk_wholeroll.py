"""Whole-roll unroll via the center-free EMULSION WALK (walk_emulsion.py).

Segmentation exists only for the 25 gapped chunks; the film is ~z-invariant. So:
  1. Per anchor (25 chunk mid-slices): film-band seed -> walk each winding along the
     emulsion (both dirs, meet at seam) -> smooth. Center-free per-winding (x,y) paths.
  2. Match windings across anchors by median radius (robust, monotone). Resample each
     winding's path to a common arc-fraction grid; interpolate (x,y) across z to all
     1184 raw-CT slices.
  3. Render the raw CT along each interpolated path; concatenate windings inner->outer.

The walk (not a polar r(theta) fit) is what changed: no spool center, so eccentricity
is irrelevant, and each band is followed as a curve so curves can't wander between
windings. Anchor walks cached (slow seg load) -> `--use-cached-anchors` re-renders fast.

Usage:
  python -m unwrapping.inr.unroll_walk_wholeroll \
      --ct-dir 01_Mickey_hdf --seg-dir 01_Mickey_3d --out-dir <out> \
      [--inspect-anchors] [--use-cached-anchors] [--smooth-px 0.6]
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

from unwrapping.center_detection import find_spool_center
from unwrapping.inr.geodesic_unwrap import _detect_seam_angle
from unwrapping.inr.unwrap_data_3d import discover_volumes
from unwrapping.inr.sanity_winding_strip import bilinear_sample
from unwrapping.inr.unroll_continuous import (
    load_ct, norm_slice, discover_ct_slices, load_seg_mid, load_seg_slices,
)
from unwrapping.inr.walk_emulsion import (
    film_seed_azimuth, walk, smooth_path, dedup_paths, recenter_on_emulsion,
)

TWO_PI = 2.0 * math.pi


def walk_anchor(seg2, cx, cy, r0, r1, film_min_thick, snap_max, step, coast_max,
                seam, eps, smooth_px, jump_max, snap_accept=None, seed_infill=True,
                max_dr=None, recenter=True, recenter_search=6.0, recenter_sigma=20.0,
                rescue=True, rescue_min_sep=6.0):
    """All per-winding emulsion paths for one seg slice, each spanning the SAME
    azimuth range [seam+eps, seam+2pi-eps]. Walking to the seam (not +-a fixed
    extent from a per-anchor seed) makes arc/azimuth align across anchors, so the
    cross-z interpolation doesn't shear. Returns list of (N,2).

    Single-ray film-band seed + spacing infill (clean ~35). `max_dr` = anti-jump.
    `recenter` = undo the smoothing spline's inward chording bias (re-lock onto the
    emulsion after smoothing)."""
    emul = (seg2 == 2).astype(np.float32)
    film = (seg2 > 0).astype(np.float32)
    seed_az, seeds = film_seed_azimuth(film, emul, cx, cy, r0, r1,
                                       film_min_thick, snap_max, infill=seed_infill)

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
        if recenter:
            p = recenter_on_emulsion(p, emul, recenter_search, recenter_sigma)
        return p

    paths = [walk_from(np.array([cx + rs * math.cos(seed_az),
                                 cy + rs * math.sin(seed_az)]), seed_az)
             for rs in seeds]
    if rescue:
        paths = rescue_bare_bands(paths, emul, cx, cy, seam, eps,
                                  rescue_min_sep, walk_from)
    return paths       # dedup moved to render phase (tunable without re-walking)


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


def _resample_xy(X, Y, K2):
    """Resample per-anchor (n,K1) X,Y to (n,K2) along the phi axis (uniform grid)."""
    nrows, K1 = X.shape
    if K1 == K2:
        return X, Y
    src = np.linspace(0, 1, K1); dst = np.linspace(0, 1, K2)
    Xo = np.empty((nrows, K2), np.float32); Yo = np.empty((nrows, K2), np.float32)
    for a in range(nrows):
        Xo[a] = np.interp(dst, src, X[a]); Yo[a] = np.interp(dst, src, Y[a])
    return Xo, Yo


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
    for wi, (r0, X, Y) in enumerate(kept):
        # NEVER heal the innermost/outermost winding: they carry the film's two
        # ENDS (inner tongue / outer termination), whose geometry is legitimately
        # partial, non-circular and strongly z-varying — healing them drew chords
        # across the void (the tail is walked at only a minority of anchors, so
        # the median-range gate below can miss it while the clamp-extension
        # quorum still fires on garbage). Both are cut from the video anyway.
        if wi == 0 or wi == len(kept) - 1:
            out.append((r0, X, Y))
            continue
        K = X.shape[1]
        th = (seam + np.linspace(eps_phi, TWO_PI - eps_phi, K))[None, :]  # ray angle
        R = np.hypot(X - cxA, Y - cyA)
        # heal assumes a ~circular band (r(phi) smooth, azimuth ~= column ray,
        # z-drift ~px). Skip windings whose per-anchor radius RANGE says "not a
        # band" (normal: ~2x eccentricity <= ~90px); p85 so a minority of anchors
        # with runaway geometry still trips the gate.
        if float(np.percentile(R.max(axis=1) - R.min(axis=1), 85)) > max_range:
            out.append((r0, X, Y))
            continue
        th_actual = np.arctan2(Y - cyA, X - cxA)
        dphi = np.abs((th_actual - th + np.pi) % TWO_PI - np.pi)
        # skip mostly-CLAMPED fragment windings (short partial stubs, e.g. the
        # sliver tracks near the tongue): healing their few on-ray bins scatters
        # points around the turn and the connecting polyline/render jumps across
        # the roll (the z1749 chord). Their v8-style stub rendering is correct.
        if float((dphi <= math.radians(3.0)).mean()) < 0.5:
            out.append((r0, X, Y))
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
        out.append((r0, X, Y))
    return out, healed


def _blend(wa, wb, w):
    """Midline between two windings (r, X(n,K), Y(n,K)) at fraction w in [0,1].
    A never-walked winding (touching-film pair) sits between two clean neighbours;
    for locally-uniform layers its emulsion is their weighted midline."""
    ra, Xa, Ya = wa; rb, Xb, Yb = wb
    K = (Xa.shape[1] + Xb.shape[1]) // 2
    Xa2, Ya2 = _resample_xy(Xa, Ya, K); Xb2, Yb2 = _resample_xy(Xb, Yb, K)
    return ((1 - w) * Xa2 + w * Xb2, (1 - w) * Ya2 + w * Yb2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-dir", required=True)
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--smooth-px", type=float, default=0.6)
    ap.add_argument("--snap-max", type=float, default=8.0)
    ap.add_argument("--step", type=float, default=2.0)
    ap.add_argument("--coast-max", type=int, default=40)
    ap.add_argument("--film-min-thick", type=float, default=6.0)
    ap.add_argument("--jump-max", type=float, default=4.0)
    ap.add_argument("--snap-accept", type=float, default=None,
                    help="Coast if the nearest class-2 run is farther than this (px) "
                         "— anti-skip. Default = snap-max (off). ~6 tightens.")
    ap.add_argument("--max-dr", type=float, default=None,
                    help="Cap radius change per walk step (px) — primary anti-JUMP "
                         "lever. ~2 blocks band-to-band jumps, leaves eccentric drift "
                         "free. Default off. NEEDS a re-walk (walk-phase knob).")
    ap.add_argument("--slice-frac", type=float, default=0.0,
                    help="Walk this FRACTION of each chunk's 20 z-slices (evenly "
                         "spaced) instead of just the mid slice. 0 = mid only (25 "
                         "anchors, default). 0.5/0.75/1.0 -> ~10/15/20 per chunk "
                         "(~250/375/500 anchors) = denser z. NEEDS a re-walk.")
    ap.add_argument("--no-recenter", dest="recenter", action="store_false",
                    help="Disable post-smoothing re-centering onto the emulsion "
                         "(removes the spline's inward chording bias).")
    ap.add_argument("--recenter-search", type=float, default=6.0)
    ap.add_argument("--recenter-sigma", type=float, default=20.0)
    ap.add_argument("--no-seed-infill", dest="seed_infill", action="store_false",
                    help="Disable spacing-infill of dropped windings (recovers "
                         "windings that drop out at the best seed ray).")
    ap.add_argument("--min-sep", type=float, default=6.0)
    ap.add_argument("--no-rescue", dest="rescue", action="store_false",
                    help="Disable the per-anchor bare-band rescue walks (2D "
                         "detection of windings the 1D seed ray missed). "
                         "NEEDS a re-walk (walk-phase knob).")
    ap.add_argument("--ref-mode", choices=["track", "cluster"], default="track",
                    help="Winding matching across anchors. track (default) = "
                         "z-sequential nearest-radius tracking, drift-robust, all "
                         "anchors contribute. cluster = legacy subsample union-"
                         "clustering (measured to drop walked windings: 29 slots "
                         "vs 31-39 walks/anchor on the dense run).")
    ap.add_argument("--dedup-px", type=float, default=10.0,
                    help="track mode: per-anchor PROFILE dedup threshold (px). Two "
                         "walks whose r(phi) agree within this on their overlap = "
                         "same winding. Real neighbours are >=17px apart.")
    ap.add_argument("--track-merge-px", type=float, default=12.0,
                    help="track mode: merge tracks whose profiles agree within "
                         "this (px) — ~half the min winding spacing.")
    ap.add_argument("--cluster-gap", type=float, default=10.0,
                    help="Radius gap (px) separating windings when clustering the "
                         "union of all anchors' walk radii into the reference set "
                         "(~half the layer spacing). Post-match, cache-fast to tune.")
    ap.add_argument("--no-gap-infill", dest="gap_infill", action="store_false",
                    help="Disable inserting never-walked windings (touching-film "
                         "pairs) as the midline of their two flanking walks.")
    ap.add_argument("--z-heal-px", type=float, default=6.0,
                    help="Repair mid-turn winding JUMPS by cross-z consensus: flag "
                         "(anchor, phi) bins whose radius deviates more than this "
                         "(px) from the running z-median and re-interpolate them "
                         "from the winding's clean anchors. Jumps are ~a spacing "
                         "(>=17px), real z-drift <1px/anchor. 0 = off. Cache-fast.")
    ap.add_argument("--z-smooth-anchors", type=float, default=0.0,
                    help="z-AWARENESS (v7): Gaussian-smooth each winding's (x,y) path "
                         "across the z-anchors (sigma in anchor-index units). The "
                         "winding drifts <0.3px/slice, so per-anchor jitter is noise; "
                         "smoothing it cuts inner-winding moire + coherent fills. "
                         "0 = off (v6). ~1.5 mild.")
    ap.add_argument("--min-coverage", type=int, default=8,
                    help="Drop a winding matched in fewer than this many anchors "
                         "(removes spurious near-duplicate windings).")
    ap.add_argument("--seam-exclude-deg", type=float, default=6.0)
    ap.add_argument("--seam-deg", type=float, default=None,
                    help="Force the seam azimuth (deg). The seam is a PHYSICAL "
                         "constant of the roll (~60deg for Mickey); single-slice "
                         "auto-detection mis-fired (27.5deg) on the full-scroll "
                         "seg. Default None = robust multi-chunk auto-detect.")
    ap.add_argument("--match-tol-px", type=float, default=12.0,
                    help="Max median-radius diff to match a winding across anchors.")
    ap.add_argument("--seam-bridge", action="store_true",
                    help="Close the residual seam gap: at each winding boundary "
                         "insert linearly-blended columns filling the 2*seam-exclude "
                         "wedge. The join is CONTINUOUS film (winding k's end sits "
                         "at winding k+1's start radius), so the blend faithfully "
                         "reconstructs the tiny missing sliver -> ~0 visible join. "
                         "Width auto-computed per winding from --seam-exclude-deg.")
    ap.add_argument("--highpass-z-sigma", type=float, default=0.0,
                    help="Remove soft z-intensity shading: subtract a z-Gaussian "
                         "(this sigma) background. 0 = off. ~60 flattens banding.")
    ap.add_argument("--z-step", type=int, default=1)
    ap.add_argument("--max-chunks", type=int, default=0,
                    help="Debug: only walk the first N chunks (timing test). 0 = all.")
    ap.add_argument("--cache-subsample", type=float, default=1.0,
                    help="With --use-cached-anchors, use this fraction of the cached "
                         "anchors (evenly spaced) — 0.5/0.75 render lower density from "
                         "the 100%% walk cache without re-walking.")
    ap.add_argument("--use-cached-anchors", action="store_true")
    ap.add_argument("--inspect-anchors", action="store_true",
                    help="Draw all winding walks on each anchor CT + montage, exit.")
    ap.add_argument("--inspect-max", type=int, default=0,
                    help="With --inspect-anchors: draw only this many evenly-spaced "
                         "anchors (0 = all). matched_walks.npz still covers ALL.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    eps_phi = math.radians(args.seam_exclude_deg)
    cache = os.path.join(args.out_dir, "walk_anchors.npz")

    # ── anchor walks (slow seg load; cached) ──
    if args.use_cached_anchors and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        z_anchor = d["z_anchor"]; cx_a = list(d["cx_a"]); cy_a = list(d["cy_a"])
        PATHS = list(d["paths"]); seam = float(d["seam"])
        if args.seam_deg is not None:
            # Override a wrong cached seam WITHOUT re-walking: the walk paths span
            # the physical band ends (~the true seam) regardless of the target, so
            # only resample_phi/z-heal's seam reference needs correcting. (Cached
            # 01_Mickey_full_3d had seam=27.5deg; true is 60deg -> the real
            # discontinuity landed mid-strip and got bridged = dark sweeps.)
            print(f"  seam override: cached {math.degrees(seam):.1f}deg -> "
                  f"{args.seam_deg:.1f}deg")
            seam = math.radians(args.seam_deg)
        if args.cache_subsample < 1.0:                    # 50/75% density from the
            m = len(z_anchor)                             # full 100% walk cache
            keep = np.unique(np.linspace(
                0, m - 1, max(2, int(round(args.cache_subsample * m)))).astype(int))
            z_anchor = np.asarray(z_anchor)[keep]
            cx_a = [cx_a[i] for i in keep]; cy_a = [cy_a[i] for i in keep]
            PATHS = [PATHS[i] for i in keep]
            print(f"subsampled cache to {len(keep)} anchors "
                  f"(frac {args.cache_subsample})")
        print(f"loaded cached walks for {len(z_anchor)} anchors")
    else:
        pairs = discover_volumes(args.seg_dir)
        if args.seam_deg is not None:
            seam = math.radians(args.seam_deg)          # user-forced (physical const)
            print(f"{len(pairs)} anchors; seam={args.seam_deg:.1f}deg (forced)")
        else:
            # ROBUST seam detection: the seam is a PHYSICAL constant of the roll
            # (spiral radial discontinuity, ~fixed azimuth for all z), but
            # _detect_seam_angle on a SINGLE slice is unreliable — it mis-fired
            # 27.5deg on 01_Mickey_full_3d's first chunk vs the true 60deg (seen on
            # every other cache), and a wrong seam makes the walk span THROUGH the
            # real seam and cross windings there (dark-sweep corruption). So detect
            # on several chunks spread over z and take the CIRCULAR median.
            seam_samples = []
            for ci in range(0, len(pairs), max(1, len(pairs) // 8)):
                s2, _ = load_seg_mid(pairs[ci][1])
                cyi, cxi = find_spool_center(s2); fi = s2 > 0
                yi, xi = np.where(fi); ri = np.hypot(yi - cyi, xi - cxi)
                seam_samples.append(_detect_seam_angle(fi, (cyi, cxi),
                                                       ri.min(), ri.max()))
            ang = np.array(seam_samples)
            seam = float(np.angle(np.mean(np.exp(1j * ang))))   # circular mean
            # circular median-ish: drop samples >30deg from the mean, re-average
            keep = np.abs((ang - seam + math.pi) % TWO_PI - math.pi) < math.radians(30)
            if keep.any():
                seam = float(np.angle(np.mean(np.exp(1j * ang[keep]))))
            print(f"{len(pairs)} anchors; seam={math.degrees(seam):.1f}deg "
                  f"(robust over {len(seam_samples)} chunks: "
                  f"{[round(math.degrees(a)) for a in seam_samples]})")
        import time
        z_anchor, cx_a, cy_a, PATHS = [], [], [], []
        for ci, (vp, pp, z0, z1) in enumerate(pairs):
            if args.max_chunks and ci >= args.max_chunks:
                break
            t0 = time.time()
            # mid-only (default) or `slice_frac` of the chunk's 20 slices (dense z)
            slices = ([load_seg_mid(pp)] if args.slice_frac <= 0
                      else load_seg_slices(pp, args.slice_frac))
            t_load = time.time() - t0
            # center ONCE per chunk (drifts <0.02 px/slice -> constant within 20 z);
            # per-slice find_spool_center was the dense bottleneck (component loop).
            t1 = time.time()
            cy, cx = find_spool_center(slices[len(slices) // 2][0])
            t_ctr = time.time() - t1
            t_walk = 0.0
            for seg2, sidx in slices:
                ys, xs = np.where(seg2 > 0); rf = np.hypot(ys - cy, xs - cx)
                tw = time.time()
                paths = walk_anchor(seg2, cx, cy, rf.min() + 5, rf.max() - 5,
                                    args.film_min_thick, args.snap_max, args.step,
                                    args.coast_max, seam, eps_phi, args.smooth_px,
                                    args.jump_max, snap_accept=args.snap_accept,
                                    seed_infill=args.seed_infill, max_dr=args.max_dr,
                                    recenter=args.recenter,
                                    recenter_search=args.recenter_search,
                                    recenter_sigma=args.recenter_sigma,
                                    rescue=args.rescue, rescue_min_sep=args.min_sep)
                t_walk += time.time() - tw
                z_anchor.append(z0 + sidx); cx_a.append(cx); cy_a.append(cy)
                PATHS.append(paths)
            print(f"  chunk z{z0}-{z1}: {len(slices)} sl | load {t_load:.0f}s "
                  f"center {t_ctr:.0f}s walk {t_walk:.0f}s", flush=True)
        z_anchor = np.array(z_anchor, float)
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
    if args.ref_mode == "track":
        PATHS = [dedup_paths_profile(list(PATHS[a]), cx_a[a], cy_a[a], seam,
                                     args.dedup_px) for a in range(n)]
    else:
        PATHS = [dedup_paths(list(PATHS[a]), cx_a[a], cy_a[a], args.min_sep)
                 for a in range(n)]
    print(f"  per-anchor winding counts after dedup(min-sep {args.min_sep}): "
          f"{[len(PATHS[a]) for a in range(n)]}")

    if args.ref_mode == "track":
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
    else:
        # ── legacy: reference = clustered UNION of subsampled anchors' walk radii ──
        # The median radius over the full seam-to-seam turn is ~eccentricity-
        # invariant, so a winding walked in ANY anchor contributes it. Built from a
        # SPARSE subsample (~25 anchors) — at full density the radii bridge the 22px
        # winding gaps and the greedy clustering merges windings (500 -> 15). Known
        # cost (dense run): 29 slots vs 31-39 walks/anchor -> real windings dropped.
        ref_step = max(1, n // 25)
        ref_idx = range(0, n, ref_step)
        all_r = np.array(sorted(median_radius(p, cx_a[a], cy_a[a])
                                for a in ref_idx for p in PATHS[a]))
        clusters = [[all_r[0]]]
        for v in all_r[1:]:
            if v - clusters[-1][-1] < args.cluster_gap:
                clusters[-1].append(v)
            else:
                clusters.append([v])
        ref_r = np.array([float(np.median(c)) for c in clusters])  # inner->outer
        nw = len(ref_r)
        print(f"  reference from {len(list(ref_idx))} subsampled anchors -> "
              f"{nw} windings")
        # matched[k][a] = the walk at anchor a nearest ref winding k (within tol)
        matched = [[None] * n for _ in range(nw)]
        for a in range(n):
            for p in PATHS[a]:
                rr = median_radius(p, cx_a[a], cy_a[a])
                k = int(np.argmin(np.abs(ref_r - rr)))
                if abs(ref_r[k] - rr) <= args.match_tol_px and matched[k][a] is None:
                    matched[k][a] = p
    covered = [sum(p is not None for p in matched[k]) for k in range(nw)]
    print(f"  {nw} windings ({args.ref_mode}); per-winding "
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
    for k in range(nw):
        present = [a for a in range(n) if matched[k][a] is not None]
        if len(present) < max(2, args.min_coverage):   # drop spurious low-cov windings
            continue
        K = max(2, int(round(np.median([arclen_of(matched[k][a]) for a in present]))))
        phi_grid = np.linspace(eps_phi, TWO_PI - eps_phi, K)   # COMMON azimuth grid
        X = np.empty((n, K), np.float32); Y = np.empty((n, K), np.float32)
        for a in range(n):
            b = a if matched[k][a] is not None else \
                min(present, key=lambda p: abs(p - a))     # fill gaps from nearest z
            X[a], Y[a] = resample_phi(matched[k][b], cx_a[b], cy_a[b], seam, phi_grid)
        kept.append((float(ref_r[k]), X, Y))
    kept.sort(key=lambda t: t[0])
    print(f"  {len(kept)} windings kept (coverage >= {args.min_coverage})")

    # ── infill windings NEVER walked anywhere (touching-film pairs) as the midline
    #    of their two flanking walks: a >1.5x-median gap = round(gap/med)-1 missing ──
    if args.gap_infill and len(kept) >= 3:
        med_gap = float(np.median(np.diff([t[0] for t in kept])))
        filled = [kept[0]]; n_ins = 0
        for i in range(1, len(kept)):
            g = kept[i][0] - kept[i - 1][0]
            m = int(round(g / med_gap)) - 1 if med_gap > 0 else 0
            for j in range(1, m + 1):
                w = j / (m + 1)
                Xb, Yb = _blend(kept[i - 1], kept[i], w)
                filled.append((kept[i - 1][0] + g * w, Xb, Yb)); n_ins += 1
            filled.append(kept[i])
        kept = filled
        print(f"  gap-infill inserted {n_ins} never-walked windings (touching pairs)")

    # ── z-heal (v9): repair mid-turn winding jumps by cross-z radius consensus ──
    if args.z_heal_px > 0 and n >= 7:
        kept, nbins = z_heal_windings(kept, cx_a, cy_a, seam, eps_phi,
                                      args.z_heal_px)
        print(f"  z-heal: repaired {nbins} jumped (anchor,phi) bins "
              f"(>{args.z_heal_px}px from running z-median)")

    # ── z-AWARENESS (v7): smooth each winding's (x,y) path across the z-anchors ──
    if args.z_smooth_anchors > 0 and n >= 3:
        from scipy.ndimage import gaussian_filter1d
        s = args.z_smooth_anchors
        kept = [(r, gaussian_filter1d(X, s, axis=0, mode="nearest"),
                 gaussian_filter1d(Y, s, axis=0, mode="nearest"))
                for (r, X, Y) in kept]
        print(f"  z-smoothed winding paths across anchors (sigma {s} anchors)")
    total_cols = sum(t[1].shape[1] for t in kept)
    print(f"  whole roll: {len(kept)} windings, {total_cols} cols")

    # ── inspect: draw the DELIVERABLE windings (kept + infilled) on each anchor CT ──
    if args.inspect_anchors:
        from PIL import Image as _Im
        mp = np.empty(n, object)
        mp[:] = [[np.column_stack([X[a], Y[a]]) for (_, X, Y) in kept] for a in range(n)]
        np.savez(os.path.join(args.out_dir, "matched_walks.npz"),
                 z_anchor=z_anchor, cx_a=cx_a, cy_a=cy_a, seam=seam, paths=mp)
        ins_idx = (range(n) if args.inspect_max <= 0 else
                   np.unique(np.linspace(0, n - 1, args.inspect_max).astype(int)))
        cmap = plt.get_cmap("hsv"); tiles = []
        for a in ins_idx:
            za = min(z_to_path, key=lambda zz: abs(zz - z_anchor[a]))
            img = norm_slice(load_ct(z_to_path[za]))
            fig, ax = plt.subplots(1, 1, figsize=(7, 7)); ax.imshow(img, cmap="gray")
            for kk, (_, X, Y) in enumerate(kept):
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
    win = [(interp1d(z_anchor, X, axis=0, bounds_error=False, fill_value=(X[0], X[-1])),
            interp1d(z_anchor, Y, axis=0, bounds_error=False, fill_value=(Y[0], Y[-1])),
            X.shape[1]) for (_, X, Y) in kept]
    z_list = sorted(z_to_path)[:: args.z_step]
    strip = np.empty((len(z_list), total_cols), dtype=np.float32)
    for j, z in enumerate(z_list):
        img = norm_slice(load_ct(z_to_path[z]))
        c = 0
        for fX, fY, K in win:
            strip[j, c:c + K] = bilinear_sample(img, fX(z), fY(z)); c += K
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
                # missing arc = 2*eps rad; match the strip's cols/rad = K/(2pi-2eps)
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

    # optional intensity polish: remove soft z-shading (high-pass along z per col)
    if args.highpass_z_sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        bg = gaussian_filter1d(strip, args.highpass_z_sigma, axis=0, mode="nearest")
        strip = strip - bg + float(strip.mean())
        print(f"  applied z high-pass (sigma {args.highpass_z_sigma})")

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
