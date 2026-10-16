"""Per-anchor walk quality vs z — find where the emulsion walk starts failing.

Measures, per anchor, on the RAW walked paths:
  * radial jumps   -- |dr| between consecutive path points above --jump-px
  * kinks          -- direction reversals in r (a walk oscillating across a band)
  * coasting       -- consecutive points further apart than --step-px * 3, i.e.
                      stretches where the walk found nothing to snap to
  * span           -- azimuthal coverage, to separate "short walk" from "bad walk"

Ignores anything within --seam-guard degrees of the seam: every path legitimately
stops eps short of it, and that structural gap otherwise dominates every count
(it is what made the z1965 overlay look broken at first glance).

Usage:
  python Scripts/diag_walk_quality.py --npz .../matched_walks.npz --zbin 50
"""
import argparse
import math

import numpy as np

TWO_PI = 2.0 * math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--jump-px", type=float, default=6.0)
    ap.add_argument("--step-px", type=float, default=2.5,
                    help="The walk's --step; coasting is a gap > 3x this.")
    ap.add_argument("--seam-guard", type=float, default=8.0,
                    help="Degrees around the seam to ignore.")
    ap.add_argument("--zbin", type=int, default=50)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    P = d["paths"]
    cx = np.asarray(d["cx_a"], float)
    cy = np.asarray(d["cy_a"], float)
    z = np.asarray(d["z_anchor"], float)
    seam = float(d["seam"])
    n = len(P)
    print(f"{args.npz}\n  {n} anchors z{z.min():.0f}-{z.max():.0f}, "
          f"seam {math.degrees(seam) % 360:.1f} deg, "
          f"ignoring +-{args.seam_guard} deg around it")

    jumps = np.zeros(n); kinks = np.zeros(n); coast = np.zeros(n)
    npath = np.zeros(n); span = np.zeros(n)
    for a in range(n):
        J = K = C = 0; sp = []
        paths = [p for p in P[a] if p is not None]
        npath[a] = len(paths)
        for p in paths:
            p = np.asarray(p)
            if p.ndim != 2 or len(p) < 5:
                continue
            th = np.arctan2(p[:, 1] - cy[a], p[:, 0] - cx[a])
            phi = np.mod(th - seam, TWO_PI)
            r = np.hypot(p[:, 0] - cx[a], p[:, 1] - cy[a])
            # mask out the structural seam gap at both ends of the walk
            g = math.radians(args.seam_guard)
            ok = (phi > g) & (phi < TWO_PI - g)
            ok2 = ok[:-1] & ok[1:]
            dr = np.diff(r)
            seg = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
            J += int(np.sum((np.abs(dr) > args.jump_px) & ok2))
            C += int(np.sum((seg > 3 * args.step_px) & ok2))
            s = np.sign(dr)
            K += int(np.sum((s[:-1] * s[1:] < 0) & ok2[:-1] & ok2[1:]))
            sp.append(float(phi.max() - phi.min()))
        jumps[a] = J; kinks[a] = K; coast[a] = C
        span[a] = np.median(sp) if sp else np.nan

    print(f"\n  {'z range':>15} {'paths':>7} {'jumps':>8} {'coast':>8} "
          f"{'kinks/pt':>9} {'span deg':>9}")
    for s in range(0, n, args.zbin):
        e = min(s + args.zbin, n)
        print(f"  {int(z[s]):6d}-{int(z[e-1]):6d} {npath[s:e].mean():7.1f} "
              f"{jumps[s:e].mean():8.1f} {coast[s:e].mean():8.1f} "
              f"{kinks[s:e].mean():9.1f} {math.degrees(np.nanmedian(span[s:e])):9.1f}")

    base = slice(0, min(300, n))
    print(f"\n  baseline (first {base.stop} anchors): jumps {jumps[base].mean():.1f}, "
          f"coast {coast[base].mean():.1f}, kinks {kinks[base].mean():.1f}")
    worst = np.argsort(-jumps)[:10]
    print("  10 worst anchors by radial jumps:")
    for a in sorted(worst):
        print(f"    z{z[a]:.0f}: jumps {jumps[a]:.0f} coast {coast[a]:.0f} "
              f"kinks {kinks[a]:.0f} paths {npath[a]:.0f}")


if __name__ == "__main__":
    main()
