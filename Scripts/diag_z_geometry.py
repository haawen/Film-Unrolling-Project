"""Per-winding radius vs z from a matched_walks.npz — where does geometry break?

The overlay montages are drawn at 480 px for a 3738 px CT, so a path 20-30 px off
its band shows up as ~3 px and looks fine. This measures instead: for every kept
winding, the median radius at every anchor, then flags anchors where a winding
jumps in z, where two windings converge below half the pitch, or where a winding
has no walk at all.

Written to settle where the dense new-scan render tears (strip rows for z>~1600)
when the raw walks, the spool centres and z-heal have all been cleared.

Usage:
  python Scripts/diag_z_geometry.py --npz .../matched_walks.npz --pitch 26
"""
import argparse
import math

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--pitch", type=float, default=26.0,
                    help="Winding pitch (px). Two windings closer than half of "
                         "this have effectively merged.")
    ap.add_argument("--jump-px", type=float, default=10.0,
                    help="Flag |dr/danchor| above this.")
    ap.add_argument("--zbin", type=int, default=50,
                    help="Summarise per this many anchors.")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    P = d["paths"]
    cx = np.asarray(d["cx_a"], float)
    cy = np.asarray(d["cy_a"], float)
    z = np.asarray(d["z_anchor"], float)
    n = len(P)
    nw = max(len(p) for p in P)
    print(f"{args.npz}\n  {n} anchors z{z.min():.0f}-{z.max():.0f}, up to {nw} windings")

    R = np.full((nw, n), np.nan)
    for a in range(n):
        for k, p in enumerate(P[a]):
            if p is None:
                continue
            p = np.asarray(p)
            if p.ndim != 2 or len(p) == 0:
                continue
            R[k, a] = float(np.median(np.hypot(p[:, 0] - cx[a], p[:, 1] - cy[a])))

    miss = np.isnan(R).sum(axis=0)
    jump = np.zeros(n)
    dR = np.abs(np.diff(R, axis=1))
    jump[1:] = np.nansum(dR > args.jump_px, axis=0)
    close = np.zeros(n)
    for a in range(n):
        r = np.sort(R[np.isfinite(R[:, a]), a])
        if len(r) > 1:
            close[a] = int(np.sum(np.diff(r) < 0.5 * args.pitch))

    print(f"\n  per-{args.zbin}-anchor summary "
          f"(missing = winding with no walk; jumps = |dr| > {args.jump_px}px; "
          f"merged = neighbour gap < {0.5*args.pitch:.0f}px)")
    print(f"  {'z range':>15} {'missing/anchor':>15} {'jumps/anchor':>13} "
          f"{'merged/anchor':>14}")
    for s in range(0, n, args.zbin):
        e = min(s + args.zbin, n)
        print(f"  {int(z[s]):6d}-{int(z[e-1]):6d} {miss[s:e].mean():15.2f} "
              f"{jump[s:e].mean():13.2f} {close[s:e].mean():14.2f}")

    print("\n  per-winding radius drift across z (first vs last valid anchor):")
    for k in range(nw):
        ok = np.nonzero(np.isfinite(R[k]))[0]
        if len(ok) < 2:
            print(f"    w{k:02d}: only {len(ok)} anchors")
            continue
        r0, r1 = R[k, ok[0]], R[k, ok[-1]]
        big = int(np.sum(np.abs(np.diff(R[k, ok])) > args.jump_px))
        flag = "   <-- JUMPY" if big > 0.02 * len(ok) else ""
        print(f"    w{k:02d}: r {r0:7.1f} -> {r1:7.1f}  over z{z[ok[0]]:.0f}-"
              f"{z[ok[-1]]:.0f}  ({len(ok)} anchors, {big} jumps){flag}")


if __name__ == "__main__":
    main()
