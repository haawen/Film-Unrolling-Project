"""Measure the seam azimuth from a SEGMENTED pairs-dir (volume_* + *_Probabilities).

The walk spans seam-to-seam, so a wrong seam makes it traverse the real radial
discontinuity and scramble windings -- that exact bug cost a full re-walk once
(detected 27.5deg vs the true 60deg from a single slice). Two rules follow, and
this script exists to enforce both:
  * detect on MANY slices and take the circular median, never one slice;
  * never assume a previously measured value -- the old scan's ~60deg is a constant
    of THAT frame, and the new scan is rotated by an unknown angle.

Runs `_detect_seam_angle` (the project's detector, validated on SEG masks -- on raw
intensity masks it fails: an old-frame control returned 12.8deg where the truth is
60deg) over every slice of each chunk.

Prints a machine-readable `SEAM_DEG=<value>` line so a job script can capture it.

Usage:
  python Scripts/diag_seam_seg.py --seg-dir <pairs dir> [--max-chunks 0]
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unwrapping.center_detection import find_spool_center
from unwrapping.inr.geodesic_unwrap import _detect_seam_angle
from unwrapping.inr.unwrap_data_3d import discover_volumes
from unwrapping.inr.unroll_continuous import load_seg_slices


def gap_profile(film, cy, cx, r_min, r_max, n_angles=720):
    """film->air transition count per azimuth: the RAW EVIDENCE behind the seam.

    `_detect_seam_angle` returns only its argmin, which on this finer scan is
    unstable -- per-slice values swung -59.5, -68.5, -102.0, -71.0, +89.0 deg with
    essentially identical centres, because the finer sampling resolves more air gaps
    and segmentation pinholes add spurious transitions, so the minimum wanders.
    Returning the profile lets many slices be SUMMED before deciding, which is
    statistically far stronger than a median of noisy per-slice decisions.
    """
    H, W = film.shape
    ang = np.linspace(-np.pi, np.pi, n_angles, endpoint=False)
    rv = np.arange(int(r_min), int(r_max) + 1)
    cnt = np.zeros(n_angles)
    for i, a in enumerate(ang):
        yy = np.round(cy + rv * np.sin(a)).astype(int)
        xx = np.round(cx + rv * np.cos(a)).astype(int)
        ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        v = film[yy[ok], xx[ok]].astype(np.int8)
        cnt[i] = int(np.sum(np.diff(v) == -1))
    return ang, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--slice-frac", type=float, default=1.0)
    ap.add_argument("--smooth-bins", type=int, default=20,
                    help="azimuth smoothing of the pooled profile (720 bins/360deg)")
    ap.add_argument("--n-centers", type=int, default=3,
                    help="TOTAL find_spool_center calls over all slices given "
                         "(evenly spread in z, interpolated); each is ~63s on the "
                         "old scan, ~2.5min on the new one")
    args = ap.parse_args()

    pairs = discover_volumes(args.seg_dir)
    if args.max_chunks:
        pairs = pairs[:args.max_chunks]
    print(f"{len(pairs)} chunk(s) in {args.seg_dir}")

    angles = []
    prof_sum, prof_n, ang_ref = None, 0, None
    for vp, pp, z0, z1 in pairs:
        slices = load_seg_slices(pp, args.slice_frac)
        # N centres per chunk, evenly scattered in z, then interpolated to every
        # slice (same policy as the walk). find_spool_center is ~63s/call, so
        # per-slice centring costs ~20min for a 20-slice chunk; one centre per chunk
        # ignores drift within it.
        nc = max(1, min(args.n_centers, len(slices)))
        idx = np.unique(np.linspace(0, len(slices) - 1, nc).astype(int))
        cs = [find_spool_center(slices[i][0]) for i in idx]
        si = np.array([slices[i][1] for i in idx], float)
        all_si = np.array([s[1] for s in slices], float)
        cy_per = np.interp(all_si, si, [c[0] for c in cs])
        cx_per = np.interp(all_si, si, [c[1] for c in cs])
        print(f"  chunk z{z0}-{z1}: {nc} centre(s) at slice {idx.tolist()} -> "
              f"{[(round(c[0], 1), round(c[1], 1)) for c in cs]}")
        for k, (seg2, sidx) in enumerate(slices):
            cy, cx = float(cy_per[k]), float(cx_per[k])
            f = seg2 > 0
            ys_, xs_ = np.where(f)
            r_ = np.hypot(ys_ - cy, xs_ - cx)
            ang, cnt = gap_profile(f, cy, cx, r_.min(), r_.max())
            ang_ref = ang
            prof_sum = cnt if prof_sum is None else prof_sum + cnt
            prof_n += 1
            ys, xs = np.where(f)
            r = np.hypot(ys - cy, xs - cx)
            a = _detect_seam_angle(f, (cy, cx), r.min(), r.max())
            angles.append(a)
            print(f"  z{z0 + sidx}: seam {math.degrees(a):+7.2f} deg  "
                  f"centre {cy:.1f},{cx:.1f}", flush=True)

    if not angles:
        raise SystemExit("no slices found")
    a = np.array(angles)
    z = np.mean(np.exp(1j * a))
    mean = float(np.angle(z))
    keep = np.abs((a - mean + math.pi) % (2 * math.pi) - math.pi) < math.radians(30)
    if keep.any():
        mean = float(np.angle(np.mean(np.exp(1j * a[keep]))))
    dev = np.degrees(np.angle(np.exp(1j * (a - mean))))
    print(f"\n  n={len(a)}  circular median {math.degrees(mean):+.2f} deg  "
          f"MAD {np.median(np.abs(dev)):.2f}  sd {np.std(dev):.2f}  "
          f"({int(keep.sum())}/{len(a)} within 30deg)")
    print(f"  spread is the trust signal: MAD of a few deg = usable; tens = do NOT "
          f"walk on it")
    print(f"  [median of per-slice argmins] {math.degrees(mean):+.2f} deg "
          f"-- shown for contrast only; unreliable when the per-slice values swing")

    from scipy.ndimage import uniform_filter1d
    pm = prof_sum / max(prof_n, 1)
    sm = uniform_filter1d(pm, size=args.smooth_bins, mode="wrap")
    k = int(np.argmin(sm))
    seam = float(ang_ref[k])
    base, dip = float(np.median(sm)), float(sm[k])
    print(f"\n  POOLED PROFILE over {prof_n} slice(s): seam "
          f"{math.degrees(seam):+.2f} deg  <- USE THIS")
    print(f"    dip {dip:.2f} vs baseline {base:.2f} gaps "
          f"({100 * (base - dip) / max(base, 1e-9):.1f}% below) -- the seam is where "
          f"ONE winding gap is missing, so expect roughly one gap of ~{base:.0f}; "
          f"a shallow dip means NOT trustworthy")
    print(f"    profile min/median/max {sm.min():.2f}/{np.median(sm):.2f}/"
          f"{sm.max():.2f}")
    print(f"SEAM_DEG={math.degrees(seam):.2f}")


if __name__ == "__main__":
    main()
