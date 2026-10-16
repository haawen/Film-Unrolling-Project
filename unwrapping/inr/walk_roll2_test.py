"""Fit-check: run the CURRENT-BEST emulsion walk (v13 seeding + walk core from
walk_emulsion.py) on the 4 available Roll2 segmentation slices and draw the
walked paths over the CT and over the segmentation.

Roll2 has only 4 single-slice probability maps (z = 200/700/1200/1700), stored
as (H, W, 3) softmax in data/roll2_probs/, with the matching raw CT tiffs in
data/<z>.tiff.  There is no z-stack, so this is a per-slice geometry fit-check
only (no strip / no video).  Every distance threshold in the walk is keyed to
Roll1's ~21 px emulsion spacing; Roll2 is wound much tighter, so we MEASURE its
spacing from the seg and rescale every threshold by that one ratio (the same
transfer rule the full-width scan used).

Usage:
  python -m unwrapping.inr.walk_roll2_test [--out-dir <dir>] [--zs 200,700,1200,1700]
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
from unwrapping.inr.walk_emulsion import (
    seed_run_intervals, film_seed_azimuth, walk, dedup_paths,
    recenter_on_emulsion, smooth_path, TWO_PI,
)

ROLL1_SPACING = 21.0   # px, the reference the walk's thresholds were tuned to


def load_roll2_slice(z, proj, ct_dirs=("02_Sample_raw", "data"), probs_dir=None):
    tif = None
    for d in ct_dirs:
        for name in (f"{z:04d}.tiff", f"{z:05d}.tiff", f"{z:04d}.tif"):
            cand = os.path.join(proj, d, name)
            if os.path.exists(cand):
                tif = cand
                break
        if tif:
            break
    if tif is None:
        raise SystemExit(f"z{z}: no CT tiff in {ct_dirs}")
    ct = np.asarray(Image.open(tif)).astype(np.float32)
    pdir = probs_dir or os.path.join(proj, "data", "roll2_probs")
    ph = os.path.join(pdir, f"{z:04d}_Probabilities.h5")
    with h5py.File(ph, "r") as f:
        probs = f["exported_data"][...]            # (H, W, 3)
    # the roll2_probs h5 is stored VERTICALLY FLIPPED vs Image.open(tiff)
    # (nnU-Net/NIfTI axis convention -- same class of bug as the Roll1 H<->W
    # seg transpose). Verified by FFT xcorr on all 4 anchors: seg[::-1] aligns
    # to the CT with 0 shift, as-loaded needs a z-varying y-shift. Flip to match.
    seg = np.argmax(probs, axis=-1)[::-1].astype(np.int64)
    if seg.shape != ct.shape:
        raise SystemExit(f"z{z}: seg {seg.shape} != ct {ct.shape}")
    return ct, seg


def measure_spacing(emul, film, cx, cy, r0, r1):
    """Median emulsion-run spacing along the ray (in +-2deg band) with the most
    class-2 runs — the local physical winding pitch for Roll2."""
    best_d, best_n = None, -1
    for az in np.linspace(0, TWO_PI, 360, endpoint=False):
        runs = seed_run_intervals(emul, cx, cy, az, r0, r1, min_thick=1.5)
        if len(runs) > best_n:
            best_n = len(runs)
            mids = np.array([0.5 * (a + b) for a, b in runs])
            best_d = np.diff(np.sort(mids)) if mids.size > 1 else None
            best_az = az
    sp = float(np.median(best_d)) if best_d is not None and best_d.size else np.nan
    # film-layer thickness (class>=1 runs) + emulsion thickness (class-2 runs)
    fr = seed_run_intervals((film > 0).astype(np.float32), cx, cy, best_az,
                            r0, r1, min_thick=1.5)
    er = seed_run_intervals(emul, cx, cy, best_az, r0, r1, min_thick=1.0)
    ft = float(np.median([b - a for a, b in fr])) if fr else np.nan
    et = float(np.median([b - a for a, b in er])) if er else np.nan
    return sp, ft, et, best_n, math.degrees(best_az)


def run_slice(z, proj, out_dir):
    ct, seg = load_roll2_slice(z, proj)
    emul = (seg == 2).astype(np.float32)
    film = (seg >= 1)
    cy, cx = find_spool_center(seg)
    ys, xs = np.where(film)
    rf = np.hypot(ys - cy, xs - cx)
    r0, r1 = rf.min() + 5, rf.max() - 5
    seam = _detect_seam_angle(film, (cy, cx), rf.min(), rf.max())

    sp, ft, et, nruns, sp_az = measure_spacing(emul, film, cx, cy, r0, r1)
    scale = sp / ROLL1_SPACING if np.isfinite(sp) and sp > 0 else 1.0

    # every threshold rescaled by the one measured ratio
    step        = max(1.0, 2.0 * scale)
    snap_max    = max(3.0, 8.0 * scale)
    snap_accept = max(2.5, 6.0 * scale)
    jump_max    = max(2.0, 4.0 * scale)
    max_dr      = max(1.0, 2.0 * scale)
    min_sep     = max(3.0, 6.0 * scale)
    film_mt     = max(2.5, 6.0 * scale)
    rec_search  = max(2.0, 6.0 * scale)
    coast_max   = 40
    seam_excl   = 6.0     # Roll2 seam unverified -> keep the wide default wedge

    print(f"\n=== Roll2 z={z} ===")
    print(f"  center=({cx:.0f},{cy:.0f})  r[{r0:.0f},{r1:.0f}]  "
          f"seam~{math.degrees(seam):.1f}deg (unverified)")
    print(f"  MEASURED  spacing={sp:.1f}px  film_thick={ft:.1f}px  "
          f"emul_thick={et:.1f}px  (best ray {sp_az:.0f}deg, {nruns} runs)")
    print(f"  Roll1 spacing {ROLL1_SPACING:.0f}px -> scale={scale:.3f}")
    print(f"  knobs: step={step:.2f} snap_max={snap_max:.2f} "
          f"snap_accept={snap_accept:.2f} jump_max={jump_max:.2f} "
          f"max_dr={max_dr:.2f} min_sep={min_sep:.2f} film_min_thick={film_mt:.2f}")

    seed_az, seeds = film_seed_azimuth(film.astype(np.float32), emul, cx, cy,
                                       r0, r1, film_mt, snap_max, infill=True)
    print(f"  seeded {len(seeds)} windings "
          f"(radii {seeds[0]:.0f}..{seeds[-1]:.0f})")

    stop_extent = math.pi - math.radians(seam_excl)
    tan_ccw = np.array([-math.sin(seed_az), math.cos(seed_az)])
    paths = []
    for rs in seeds:
        p0 = np.array([cx + rs * math.cos(seed_az), cy + rs * math.sin(seed_az)])
        fwd = walk(emul, p0, tan_ccw, cx, cy, step, snap_max, coast_max,
                   stop_extent, jump_max=jump_max, snap_accept=snap_accept,
                   max_dr=max_dr)
        bwd = walk(emul, p0, -tan_ccw, cx, cy, step, snap_max, coast_max,
                   stop_extent, jump_max=jump_max, snap_accept=snap_accept,
                   max_dr=max_dr)
        path = np.vstack([bwd[::-1], fwd[1:]])
        sm = smooth_path(path, 0.6)
        sm = recenter_on_emulsion(sm, emul, rec_search, 20.0)
        paths.append(sm)
    paths = dedup_paths(paths, cx, cy, min_sep)
    print(f"  {len(paths)} windings after dedup (min-sep {min_sep:.1f})")

    # ---- overlays ----
    lo, hi = np.percentile(ct, [1, 99])
    disp = np.clip((ct - lo) / (hi - lo + 1e-6), 0, 1)
    seg_rgb = np.zeros(seg.shape + (3,), np.float32)
    seg_rgb[seg == 1] = (0.50, 0.50, 0.50)
    seg_rgb[seg == 2] = (1.0, 0.15, 0.15)
    cmap = plt.get_cmap("hsv")

    for bg, img, tag in [("ct", disp, "CT"), ("seg", seg_rgb, "SEG  emul=red")]:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(img, cmap="gray" if bg == "ct" else None)
        for k, pth in enumerate(paths):
            ax.plot(pth[:, 0], pth[:, 1], "-", lw=0.8,
                    color=cmap(k / max(1, len(paths))))
        ax.plot(cx, cy, "c+", ms=12)
        ax.set_title(f"Roll2 z={z}: {len(paths)} emulsion walks on {tag}  "
                     f"(spacing {sp:.1f}px, scale {scale:.2f})")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"walk_z{z:04d}_{bg}.png"), dpi=130)
        plt.close()

    # ---- zoom: three wedges, paths vs emulsion pixels at 1:1 ----
    ey, ex = np.where(seg == 2)
    th_e = np.arctan2(ey - cy, ex - cx)
    re = np.hypot(ey - cy, ex - cx)
    rmid = 0.5 * (r0 + r1)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    for ax, wdeg in zip(axes, (90.0, 210.0, 330.0)):
        wc = math.radians(wdeg)
        wband = math.radians(6)
        m = (np.abs((th_e - wc + math.pi) % TWO_PI - math.pi) < wband) \
            & (re > rmid - 160) & (re < rmid + 160)
        if m.sum() < 40:
            ax.set_visible(False); continue
        x0, x1 = max(0, ex[m].min() - 12), ex[m].max() + 12
        y0, y1 = max(0, ey[m].min() - 12), ey[m].max() + 12
        ax.imshow(disp[y0:y1, x0:x1], cmap="gray")
        ax.scatter(ex[m] - x0, ey[m] - y0, s=3, c="cyan", alpha=0.35)
        for pth in paths:
            ax.plot(pth[:, 0] - x0, pth[:, 1] - y0, "-", color="yellow", lw=1.0)
        ax.set_xlim(0, x1 - x0); ax.set_ylim(y1 - y0, 0)
        ax.set_title(f"wedge {wdeg:.0f}deg")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Roll2 z={z}: walk (yellow) vs emulsion px (cyan), 1:1")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"walk_z{z:04d}_zoom.png"), dpi=120)
    plt.close()

    return dict(z=z, spacing=sp, film_thick=ft, emul_thick=et, scale=scale,
               n_seed=len(seeds), n_kept=len(paths),
               r0=float(r0), r1=float(r1))


def main():
    ap = argparse.ArgumentParser()
    proj = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ap.add_argument("--out-dir",
                    default=os.path.join(proj, "unwrapping", "inr", "results",
                                         "walk_roll2_test"))
    ap.add_argument("--zs", default="200,700,1200,1700")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    zs = [int(z) for z in args.zs.split(",")]

    rows = [run_slice(z, proj, args.out_dir) for z in zs]

    print("\n==== SUMMARY ====")
    print(f"{'z':>6} {'spacing':>8} {'film_t':>7} {'emul_t':>7} {'scale':>6} "
          f"{'seed':>5} {'kept':>5}  r-range")
    for r in rows:
        print(f"{r['z']:>6} {r['spacing']:>8.1f} {r['film_thick']:>7.1f} "
              f"{r['emul_thick']:>7.1f} {r['scale']:>6.3f} {r['n_seed']:>5} "
              f"{r['n_kept']:>5}  [{r['r0']:.0f},{r['r1']:.0f}]")
    print(f"\noverlays -> {args.out_dir}")


if __name__ == "__main__":
    main()
