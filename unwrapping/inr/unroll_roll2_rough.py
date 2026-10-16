"""Rough Roll2 unwrapping by z-interpolation of a handful of segmentation walks.

Roll2 only has 4 non-contiguous segmentation slices (z = 200/700/1200/1700).
This does the cheap thing the user asked for:

  1. walk the emulsion (class 2) on each of the 4 seg slices ONLY -- the walk
     never sees the CT (thresholds rescaled by Roll2's measured ~11.4px winding
     spacing vs Roll1's 21px);
  2. turn each kept walk into an r(phi) profile about that slice's spool centre
     on a common azimuth grid;
  3. MATCH windings one-to-one across the 4 anchors (z-sequential, order-
     preserving nearest-radius) so each "track" is the SAME physical winding at
     every anchor it was found in -- NOT fractional depth, which slides across
     3-11 windings between anchors on this roll;
  4. for every CT z, linearly interpolate each track's r(phi) between the
     anchors where that winding exists; extrapolate from its two nearest such
     anchors outside that span; enforce non-crossing + a sane radius clamp;
  5. --mode geometry : just draw the interpolated winding geometry over the CT
     at a spread of z (anchors + interpolated + extrapolated) for a fit check;
     --mode full : sample the CT along every track over all z, concat the
     tracks inner->outer -> rough strip (+ overview / crops).

It is deliberately rough: only 4 anchors, no winding-count reconciliation, no
z-heal, no seam recovery, no arc-length columns. A winding seen at <2 anchors
is dropped.

Usage (Merlin):
  python -m unwrapping.inr.unroll_roll2_rough \
      --probs-dir data/roll2_probs --anchor-zs 200,700,1200,1700 \
      --ct-raw-dir 02_Sample_raw --z-range 1,1936 \
      --mode geometry --overlay-zs 60,200,430,700,950,1200,1450,1700,1880 \
      --out-dir unwrapping/inr/results/roll2_rough
"""
import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.center_detection import find_spool_center
from unwrapping.inr.geodesic_unwrap import _detect_seam_angle
from unwrapping.inr.sanity_winding_strip import bilinear_sample
from unwrapping.inr.walk_emulsion import (
    seed_run_intervals, film_seed_azimuth, walk, dedup_paths,
    recenter_on_emulsion, smooth_path, TWO_PI,
)

ROLL1_SPACING = 21.0
NPHI = 1440                       # arc samples per full turn; set via --nphi in main
PHI = np.linspace(-math.pi, math.pi, NPHI, endpoint=False)


def _set_nphi(n):
    """Rebuild the module-level arc grid (functions read NPHI/PHI globally)."""
    global NPHI, PHI
    NPHI = int(n)
    PHI = np.linspace(-math.pi, math.pi, NPHI, endpoint=False)


# ----------------------------- CT loading ---------------------------------
def load_ct_slice(z, raw_dir, chunk_dir=None):
    for name in (f"{z:04d}.tiff", f"{z:05d}.tiff", f"{z:04d}.tif"):
        p = os.path.join(raw_dir, name)
        if os.path.exists(p):
            return np.asarray(Image.open(p)).astype(np.float32)
    if chunk_dir:
        z0 = (z // 20) * 20
        for z1 in (z0 + 19, z0 + 15, z0 + 3):
            p = os.path.join(chunk_dir, f"volume_{z0:04d}-{z1:04d}.h5")
            if os.path.exists(p):
                with h5py.File(p, "r") as f:
                    return f["volume"][z - z0].astype(np.float32)
    raise FileNotFoundError(f"no CT slice for z={z}")


# ----------------------------- one anchor --------------------------------
def measure_spacing(emul, cx, cy, r0, r1):
    best_d, best_n, best_az = None, -1, 0.0
    for az in np.linspace(0, TWO_PI, 360, endpoint=False):
        runs = seed_run_intervals(emul, cx, cy, az, r0, r1, min_thick=1.5)
        if len(runs) > best_n:
            best_n = len(runs)
            mids = np.sort([0.5 * (a + b) for a, b in runs])
            best_d = np.diff(mids) if mids.size > 1 else None
            best_az = az
    sp = float(np.median(best_d)) if best_d is not None and best_d.size else np.nan
    return sp, best_n, best_az


def path_to_profile(path, cx, cy, seam_az, seam_half_deg=8.0):
    """(K,2) walk path -> r(PHI) about (cx,cy) + a validity mask.

    The walk spans seam->...->seam (~350 deg), so a wedge at the seam is never
    covered; np.interp (period 2pi) bridges it and we mark it invalid.
    """
    ph = np.arctan2(path[:, 1] - cy, path[:, 0] - cx)
    r = np.hypot(path[:, 0] - cx, path[:, 1] - cy)
    ph = (ph + math.pi) % TWO_PI - math.pi
    o = np.argsort(ph)
    ph, r = ph[o], r[o]
    keep = np.r_[True, np.diff(ph) > 1e-6]
    ph, r = ph[keep], r[keep]
    if ph.size < 8:
        return None, None
    php = np.r_[ph - TWO_PI, ph, ph + TWO_PI]
    rp = np.r_[r, r, r]
    rr = np.interp(PHI, php, rp)
    # validity: a real sample within ~1.5 grid steps AND outside the seam wedge
    dphi = np.min(np.abs((PHI[:, None] - ph[None, :] + math.pi) % TWO_PI - math.pi),
                  axis=1)
    valid = dphi < math.radians(2.0)
    dseam = np.abs((PHI - seam_az + math.pi) % TWO_PI - math.pi)
    valid &= dseam > math.radians(seam_half_deg)
    return rr, valid


def walk_anchor(z, probs_dir, verbose=True):
    ph = os.path.join(probs_dir, f"{z:04d}_Probabilities.h5")
    with h5py.File(ph, "r") as f:
        probs = f["exported_data"][...]
    # roll2_probs h5 is stored VERTICALLY FLIPPED vs Image.open(tiff) / the CT
    # h5 chunks (NIfTI/nnU-Net axis convention -- same class as the Roll1 H<->W
    # transpose). FFT xcorr on all 4 anchors: seg[::-1] aligns at 0 shift.
    seg = np.argmax(probs, axis=-1)[::-1].astype(np.int64)
    emul = (seg == 2).astype(np.float32)
    film = (seg >= 1)
    cy, cx = find_spool_center(seg)
    ys, xs = np.where(film)
    rf = np.hypot(ys - cy, xs - cx)
    r0, r1 = rf.min() + 5, rf.max() - 5
    seam = _detect_seam_angle(film, (cy, cx), rf.min(), rf.max())

    sp, nruns, _ = measure_spacing(emul, cx, cy, r0, r1)
    scale = sp / ROLL1_SPACING if np.isfinite(sp) and sp > 0 else 1.0
    step = max(1.0, 2.0 * scale)
    snap_max = max(3.0, 8.0 * scale)
    snap_accept = max(2.5, 6.0 * scale)
    jump_max = max(2.0, 4.0 * scale)
    max_dr = max(1.0, 2.0 * scale)
    min_sep = max(3.0, 6.0 * scale)
    film_mt = max(2.5, 6.0 * scale)
    rec_search = max(2.0, 6.0 * scale)

    seed_az, seeds = film_seed_azimuth(film.astype(np.float32), emul, cx, cy,
                                       r0, r1, film_mt, snap_max, infill=True,
                                       verbose=False)
    stop_extent = math.pi - math.radians(6.0)
    tan_ccw = np.array([-math.sin(seed_az), math.cos(seed_az)])
    paths = []
    for rs in seeds:
        p0 = np.array([cx + rs * math.cos(seed_az), cy + rs * math.sin(seed_az)])
        fwd = walk(emul, p0, tan_ccw, cx, cy, step, snap_max, 40, stop_extent,
                   jump_max=jump_max, snap_accept=snap_accept, max_dr=max_dr)
        bwd = walk(emul, p0, -tan_ccw, cx, cy, step, snap_max, 40, stop_extent,
                   jump_max=jump_max, snap_accept=snap_accept, max_dr=max_dr)
        pth = np.vstack([bwd[::-1], fwd[1:]])
        pth = smooth_path(pth, 0.6)
        pth = recenter_on_emulsion(pth, emul, rec_search, 20.0)
        paths.append(pth)
    paths = dedup_paths(paths, cx, cy, min_sep)

    profs, valids, medr = [], [], []
    for pth in paths:
        rr, vv = path_to_profile(pth, cx, cy, seam)
        if rr is None:
            continue
        profs.append(rr); valids.append(vv)
        medr.append(float(np.median(rr[vv]) if vv.any() else np.median(rr)))
    order = np.argsort(medr)
    profs = [profs[i] for i in order]
    valids = [valids[i] for i in order]
    medr = [medr[i] for i in order]
    if verbose:
        print(f"  z{z}: spacing={sp:.1f} scale={scale:.3f} seeds={len(seeds)} "
              f"kept={len(profs)} r[{medr[0]:.0f}..{medr[-1]:.0f}] "
              f"seam~{math.degrees(seam):.0f}deg", flush=True)
    return dict(z=z, cx=float(cx), cy=float(cy), seam=float(seam),
               spacing=float(sp), scale=float(scale),
               profs=np.array(profs), valids=np.array(valids),
               medr=np.array(medr))


# ---------------- one-to-one winding matching across anchors -------------
def build_tracks(anchors, tol_mult=1.7, min_anchors=2):
    """Chain each physical winding across the anchors (z-sequential, order-
    preserving nearest-radius match). Returns tracks sorted inner->outer, each:
        {'rad': (A,NPHI) NaN where absent, 'val': (A,NPHI), 'pop': (A,) bool,
         'r_at': (A,) median radius or nan}
    A winding that only appears at one anchor (edge / spurious) is dropped
    (min_anchors). This REPLACES fractional-depth slots: a track is the SAME
    film winding at every anchor, so interpolating it in z stays on one layer.
    """
    A = len(anchors)
    tracks = []                       # each: dict(rad=[None]*A, val=[None]*A, r=[nan]*A)
    for i, rr in enumerate(anchors[0]["profs"]):
        tracks.append(dict(rad=[None] * A, val=[None] * A,
                           r=[np.nan] * A))
        tracks[-1]["rad"][0] = rr
        tracks[-1]["val"][0] = anchors[0]["valids"][i]
        tracks[-1]["r"][0] = float(anchors[0]["medr"][i])

    for a in range(1, A):
        newR = np.asarray(anchors[a]["medr"], float)
        newP = anchors[a]["profs"]
        newV = anchors[a]["valids"]
        tol = tol_mult * float(np.median(np.diff(np.sort(newR)))) if len(newR) > 1 else 20.0
        # last known radius per track
        cur = []
        for t in tracks:
            rlast = next((t["r"][k] for k in range(a - 1, -1, -1)
                          if t["rad"][k] is not None), np.nan)
            cur.append(rlast)
        cur = np.asarray(cur)
        ti = np.argsort(np.where(np.isnan(cur), 1e9, cur))   # tracks by radius
        order_new = np.argsort(newR)
        pt, pn = 0, 0
        used_new = set()
        while pt < len(ti) and pn < len(order_new):
            it = ti[pt]; ino = order_new[pn]
            if np.isnan(cur[it]):
                pt += 1; continue
            d = newR[ino] - cur[it]
            if abs(d) <= tol:
                tracks[it]["rad"][a] = newP[ino]
                tracks[it]["val"][a] = newV[ino]
                tracks[it]["r"][a] = float(newR[ino])
                used_new.add(ino); pt += 1; pn += 1
            elif cur[it] < newR[ino]:
                pt += 1                       # this track has no match at anchor a
            else:
                pn += 1                       # this new winding starts a fresh track
        for ino in order_new:
            if ino in used_new:
                continue
            tr = dict(rad=[None] * A, val=[None] * A, r=[np.nan] * A)
            tr["rad"][a] = newP[ino]; tr["val"][a] = newV[ino]
            tr["r"][a] = float(newR[ino])
            tracks.append(tr)

    out = []
    for t in tracks:
        pop = np.array([r is not None for r in t["rad"]])
        if pop.sum() < min_anchors:
            continue
        rad = np.stack([r if r is not None else np.full(NPHI, np.nan)
                        for r in t["rad"]])
        val = np.stack([v.astype(float) if v is not None else np.full(NPHI, np.nan)
                        for v in t["val"]])
        out.append(dict(rad=rad, val=val, pop=pop, r_at=np.array(t["r"], float)))
    out.sort(key=lambda t: np.nanmean(t["r_at"]))
    return out


def build_geometry(z, anchors, tracks, rmin, rmax):
    """Interp/extrap (cx,cy) + (M,NPHI) per-track radii + validity at z, where
    each row is ONE tracked winding (interpolated only between the anchors where
    that winding was actually found)."""
    zs = np.array([a["z"] for a in anchors], float)
    cx = float(np.interp(z, zs, [a["cx"] for a in anchors]))
    cy = float(np.interp(z, zs, [a["cy"] for a in anchors]))
    if z < zs[0]:
        tag = "extrap<"
    elif z > zs[-1]:
        tag = "extrap>"
    else:
        tag = "anchor" if np.any(np.isclose(zs, z)) else "interp"
    M = len(tracks)
    rad = np.empty((M, NPHI)); val = np.empty((M, NPHI))
    for j, t in enumerate(tracks):
        pj = np.where(t["pop"])[0]
        if z <= zs[pj[0]]:
            a, b = pj[0], pj[1]
        elif z >= zs[pj[-1]]:
            a, b = pj[-2], pj[-1]
        else:
            k = int(np.searchsorted(zs[pj], z)) - 1
            k = int(np.clip(k, 0, len(pj) - 2))
            a, b = pj[k], pj[k + 1]
        w = (z - zs[a]) / (zs[b] - zs[a])
        rad[j] = (1 - w) * t["rad"][a] + w * t["rad"][b]
        val[j] = np.clip((1 - w) * t["val"][a] + w * t["val"][b], 0, 1)
    rad = np.sort(rad, axis=0)                          # non-crossing
    rad = np.clip(rad, 0.85 * rmin, 1.15 * rmax)
    return cx, cy, rad, val, tag


# ------------------------------ overlays --------------------------------
def draw_overlay(z, ct, cx, cy, rad, val, tag, out_png):
    lo, hi = np.percentile(ct, [1, 99])
    disp = np.clip((ct - lo) / (hi - lo + 1e-6), 0, 1)
    M = rad.shape[0]
    cmap = plt.get_cmap("hsv")
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(disp, cmap="gray")
    for j in range(M):
        x = cx + rad[j] * np.cos(PHI)
        y = cy + rad[j] * np.sin(PHI)
        m = val[j] > 0.4
        x = np.where(m, x, np.nan); y = np.where(m, y, np.nan)
        ax.plot(x, y, "-", lw=0.7, color=cmap(j / max(1, M)))
    ax.plot(cx, cy, "c+", ms=12)
    ax.set_title(f"Roll2 rough geometry @ z={z}  [{tag}]   {M} windings")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser()
    proj = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ap.add_argument("--probs-dir", default="data/roll2_probs")
    ap.add_argument("--anchor-zs", default="200,700,1200,1700")
    ap.add_argument("--ct-raw-dir", default="02_Sample_raw")
    ap.add_argument("--ct-chunk-dir", default="02_Sample_3d")
    ap.add_argument("--z-range", default="1,1936")
    ap.add_argument("--nphi", type=int, default=1440,
                    help="Arc samples per turn. 1440 is too coarse for Roll2's "
                         "~8000px outer circumference (frames unresolvable); use "
                         "~8192 for a frame-by-frame strip.")
    ap.add_argument("--mode", choices=["geometry", "full"], default="geometry")
    ap.add_argument("--overlay-zs",
                    default="60,200,430,700,950,1200,1450,1700,1880")
    ap.add_argument("--out-dir",
                    default=os.path.join(proj, "unwrapping/inr/results/roll2_rough"))
    args = ap.parse_args()
    _set_nphi(args.nphi)
    os.makedirs(args.out_dir, exist_ok=True)
    anchor_zs = [int(x) for x in args.anchor_zs.split(",")]
    print(f"NPHI={NPHI} arc samples/turn", flush=True)

    print(f"[1/4] walking {len(anchor_zs)} seg anchors (seg only) ...", flush=True)
    anchors = [walk_anchor(z, args.probs_dir) for z in anchor_zs]
    rmin = min(float(a["medr"][0]) for a in anchors)
    rmax = max(float(a["medr"][-1]) for a in anchors)
    tracks = build_tracks(anchors)
    M = len(tracks)
    pop_hist = np.bincount([int(t["pop"].sum()) for t in tracks],
                           minlength=len(anchors) + 1)
    print(f"[2/4] matched windings -> {M} tracks  "
          f"(present-at-k-anchors histogram k=2..{len(anchors)}: "
          f"{pop_hist[2:].tolist()};  r {rmin:.0f}..{rmax:.0f})", flush=True)

    np.savez_compressed(
        os.path.join(args.out_dir, "roll2_geometry.npz"),
        anchor_zs=np.array(anchor_zs), M=M, rmin=rmin, rmax=rmax, PHI=PHI,
        cx=np.array([a["cx"] for a in anchors]),
        cy=np.array([a["cy"] for a in anchors]),
        seam=np.array([a["seam"] for a in anchors]),
        track_rad=np.stack([t["rad"] for t in tracks]),      # (M,A,NPHI) NaN=absent
        track_val=np.stack([t["val"] for t in tracks]),
        track_pop=np.stack([t["pop"] for t in tracks]),
        track_r_at=np.stack([t["r_at"] for t in tracks]),
        spacing=np.array([a["spacing"] for a in anchors]),
    )

    if args.mode == "geometry":
        ozs = [int(x) for x in args.overlay_zs.split(",")]
        print(f"[3/4] overlays at z = {ozs}", flush=True)
        pngs = []
        for z in ozs:
            ct = load_ct_slice(z, args.ct_raw_dir, args.ct_chunk_dir)
            cx, cy, rad, val, tag = build_geometry(z, anchors, tracks, rmin, rmax)
            p = os.path.join(args.out_dir, f"geom_z{z:04d}_{tag}.png")
            draw_overlay(z, ct, cx, cy, rad, val, tag, p)
            pngs.append(p)
            print(f"    z{z:>4} [{tag}]  center=({cx:.0f},{cy:.0f})  "
                  f"r[{rad.min():.0f}..{rad.max():.0f}]  -> {os.path.basename(p)}",
                  flush=True)
        cols = 3
        rows = (len(pngs) + cols - 1) // cols
        ims = [Image.open(p) for p in pngs]
        w, h = ims[0].size
        s = 620 / w
        w, h = int(w * s), int(h * s)
        canv = Image.new("RGB", (w * cols, h * rows), "white")
        for i, im in enumerate(ims):
            canv.paste(im.resize((w, h)), ((i % cols) * w, (i // cols) * h))
        canv.save(os.path.join(args.out_dir, "geom_montage.png"))
        print(f"[4/4] montage -> {args.out_dir}/geom_montage.png", flush=True)
        return

    # ---- full: sample CT along every slot over all z ----
    z0, z1 = [int(x) for x in args.z_range.split(",")]
    zs = list(range(z0, z1 + 1))
    print(f"[3/4] sampling CT along {M} slots over z {z0}..{z1} ...", flush=True)
    rows_per_slot = [[] for _ in range(M)]
    for zi, z in enumerate(zs):
        try:
            ct = load_ct_slice(z, args.ct_raw_dir, args.ct_chunk_dir)
        except FileNotFoundError:
            for j in range(M):
                rows_per_slot[j].append(rows_per_slot[j][-1] if rows_per_slot[j]
                                        else np.zeros(NPHI, np.float32))
            continue
        cx, cy, rad, val, tag = build_geometry(z, anchors, tracks, rmin, rmax)
        for j in range(M):
            x = cx + rad[j] * np.cos(PHI)
            y = cy + rad[j] * np.sin(PHI)
            row = bilinear_sample(ct, x, y).astype(np.float32)
            row[val[j] <= 0.4] = np.nan
            rows_per_slot[j].append(row)
        if zi % 100 == 0:
            print(f"    z{z} ({zi}/{len(zs)}) [{tag}]", flush=True)
    strips = [np.array(r) for r in rows_per_slot]            # each (Z, NPHI)
    strip = np.concatenate(strips, axis=1)                   # (Z, M*NPHI)
    np.save(os.path.join(args.out_dir, "roll2_rough_strip.npy"), strip)
    fin = np.isfinite(strip)
    loS, hiS = np.percentile(strip[fin], [1, 99])
    sd = np.clip((np.nan_to_num(strip, nan=loS) - loS) / (hiS - loS + 1e-6), 0, 1)
    stepc = max(1, strip.shape[1] // 2400)
    Image.fromarray((np.repeat(sd[:, ::stepc], 3, 0) * 255).astype(np.uint8)).save(
        os.path.join(args.out_dir, "roll2_rough_overview.png"))
    c0 = strip.shape[1] // 2
    Image.fromarray((sd[:, c0:c0 + 1600] * 255).astype(np.uint8)).save(
        os.path.join(args.out_dir, "roll2_rough_crop.png"))
    print(f"[4/4] strip {strip.shape} -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
