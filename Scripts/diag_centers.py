"""Dump and score the per-anchor spool-centre track held in a walk cache.

Why this matters: the centre feeds resample_phi's common azimuth grid, so a
centre error dx,dy displaces a column along the film by

    ds = r * dphi ~ dx*sin(phi) - dy*cos(phi)

which is INDEPENDENT of radius and sinusoidal in azimuth -- i.e. a wrong centre
at one z shifts that strip ROW back and forth once per turn, against its
neighbours in z. In the rendered video (whose horizontal axis is z) that reads as
a band of the picture sliding along the film relative to the rest. A centre error
is therefore a candidate for any artifact that is confined in z.

find_spool_center scatters by tens of px per call on this scan, so the check is
not "is any sample odd" but "does the interpolated track leave a straight line" —
the true drift over the whole stack is ~30 px and within a batch is sub-px.

Usage:
  python Scripts/diag_centers.py --caches a/walk_anchors.npz b/... [--zbin 50]
"""
import argparse
import os

import numpy as np


def robust_line(z, v, n_iter=4, clip=3.0):
    keep = np.ones(len(z), bool)
    for _ in range(n_iter):
        p = np.polyfit(z[keep], v[keep], 1)
        r = v - np.polyval(p, z)
        mad = np.median(np.abs(r - np.median(r))) + 1e-9
        new = np.abs(r - np.median(r)) < clip * 1.4826 * mad
        if new.sum() < 0.5 * len(z) or (new == keep).all():
            break
        keep = new
    return np.polyval(np.polyfit(z[keep], v[keep], 1), z), keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--zbin", type=int, default=50)
    ap.add_argument("--flag-px", type=float, default=10.0)
    args = ap.parse_args()

    for c in args.caches:
        d = np.load(c, allow_pickle=True)
        z = np.asarray(d["z_anchor"], float)
        cx = np.asarray(d["cx_a"], float)
        cy = np.asarray(d["cy_a"], float)
        print(f"\n=== {os.path.relpath(c)} ===")
        print(f"  {len(z)} anchors z{z.min():.0f}-{z.max():.0f}")
        print(f"  cx {cx.min():.1f}-{cx.max():.1f} (ptp {np.ptp(cx):.1f}), "
              f"cy {cy.min():.1f}-{cy.max():.1f} (ptp {np.ptp(cy):.1f})")
        fx, kx = robust_line(z, cx)
        fy, ky = robust_line(z, cy)
        rx, ry = cx - fx, cy - fy
        d2 = np.hypot(rx, ry)
        print(f"  residual from a robust line: median {np.median(d2):.2f} px, "
              f"p90 {np.percentile(d2, 90):.2f}, max {d2.max():.2f}")
        bad = np.flatnonzero(d2 > args.flag_px)
        if len(bad):
            # report contiguous z runs rather than every anchor
            runs = np.split(bad, np.flatnonzero(np.diff(bad) > 1) + 1)
            print(f"  {len(bad)} anchors off the line by > {args.flag_px} px, "
                  f"in {len(runs)} runs:")
            for r in runs:
                print(f"    z{z[r[0]]:.0f}-{z[r[-1]]:.0f} ({len(r)} anchors), "
                      f"max {d2[r].max():.1f} px  "
                      f"[dx {rx[r][np.argmax(d2[r])]:+.1f}, "
                      f"dy {ry[r][np.argmax(d2[r])]:+.1f}]")
        else:
            print(f"  no anchor off the line by more than {args.flag_px} px")

        print(f"  {'z range':>15} {'cx':>9} {'cy':>9} {'|resid|':>9}")
        for s in range(0, len(z), args.zbin):
            e = min(s + args.zbin, len(z))
            print(f"  {z[s]:7.0f}-{z[e-1]:7.0f} {cx[s:e].mean():9.1f} "
                  f"{cy[s:e].mean():9.1f} {d2[s:e].max():9.1f}")


if __name__ == "__main__":
    main()
