"""Where does each walked winding TERMINATE? — a seam probe from the walk itself.

The physical seam is the film's OUTER END: the outermost winding exists only up
to that azimuth, and beyond it the roll has one fewer layer.  The walk is told a
seam and then walks phi in [eps, 2pi-eps] about it, so an interior winding always
reports the full 2pi-2eps span; a winding that reports LESS has hit real film
geometry, and the azimuth where it stops is a direct read of a film terminus.

Unlike the stop-azimuth clustering in Scripts/diag_seam_gap.py (which failed its
control -- the old and new scans gave near-identical 251/349 deg clusters, an
impossible coincidence for two different rolls), this reports per winding with
its radius, so the OUTERMOST winding's terminus can be read on its own instead of
being pooled with the inner tongue and with partial mid-roll walks.

RUN THE OLD SCAN FIRST: its seam is independently known to be 60 deg.

Usage:
  python Scripts/diag_seam_terminus.py --npz .../matched_walks.npz --expect-deg 60
"""
import argparse
import math

import numpy as np

TWO_PI = 2.0 * math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--anchors", type=int, default=20,
                    help="How many anchors to sample (evenly spread in z).")
    ap.add_argument("--full-tol-deg", type=float, default=1.0,
                    help="A winding within this of the full target span counts "
                         "as 'never terminates'.")
    ap.add_argument("--expect-deg", type=float, default=None)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    seam = float(d["seam"])
    P = d["paths"]
    cx_a = np.asarray(d["cx_a"], float)
    cy_a = np.asarray(d["cy_a"], float)
    na = len(P)
    sel = np.unique(np.linspace(0, na - 1, min(args.anchors, na)).astype(int))
    nw = min(len(P[a]) for a in sel)
    print(f"{args.npz}\n  {na} anchors ({len(sel)} sampled), {nw} windings; "
          f"seam = {math.degrees(seam) % 360:.2f} deg (phi = 0)")

    span_target = None
    rows = []
    for k in range(nw):
        los, his, rs = [], [], []
        for a in sel:
            p = np.asarray(P[a][k])
            th = np.arctan2(p[:, 1] - cy_a[a], p[:, 0] - cx_a[a])
            phi = np.degrees(np.mod(th - seam, TWO_PI))
            los.append(phi.min()); his.append(phi.max())
            rs.append(float(np.hypot(p[:, 0] - cx_a[a], p[:, 1] - cy_a[a]).mean()))
        lo, hi, r = np.median(los), np.median(his), np.median(rs)
        rows.append((k, r, lo, hi, hi - lo))
        span_target = max(span_target or 0.0, hi - lo)

    print(f"  full span (interior windings) = {span_target:.1f} deg\n")
    print("  wnd  radius     phi_lo   phi_hi     span    ABS start   ABS end")
    term = []
    for k, r, lo, hi, span in rows:
        a0 = (math.degrees(seam) + lo) % 360.0
        a1 = (math.degrees(seam) + hi) % 360.0
        flag = ""
        if span < span_target - args.full_tol_deg:
            flag = "  <-- TERMINATES"
            term.append((k, r, a0, a1, lo, hi))
        print(f"  w{k:02d}  {r:7.0f}   {lo:7.1f}  {hi:7.1f}  {span:7.1f}   "
              f"{a0:7.1f}   {a1:7.1f}{flag}")

    # A bound only counts as a TERMINUS if it is strictly inside the walk's own
    # [phi_lo_target, phi_hi_target] range. A path that stops AT the target
    # stopped because it was told to, and its absolute azimuth is then seam+-eps
    # by construction -- reporting that as a measurement of the seam is circular
    # (it made this probe's first old-scan "control" read 59.7 vs a known 60.0,
    # which proved nothing at all).
    lo_t = min(r[2] for r in rows)
    hi_t = max(r[3] for r in rows)
    tol = args.full_tol_deg
    print(f"\n  walk targets phi {lo_t:.1f}..{hi_t:.1f}; a bound within {tol} deg "
          f"of those is the walk stopping on command, NOT a film end.\n"
          f"  film termini (bounds strictly inside the target range):")
    found = []
    for k, r, lo, hi, span in rows:
        a0 = (math.degrees(seam) + lo) % 360.0
        a1 = (math.degrees(seam) + hi) % 360.0
        which = []
        if lo > lo_t + tol:
            which.append(("CW", a0))
        if hi < hi_t - tol:
            which.append(("CCW", a1))
        for side, aa in which:
            found.append((k, r, side, aa))
            print(f"    w{k:02d} r{r:.0f}: {side} end at ABS {aa:.1f} deg")
    if not found:
        print("    none -- every winding runs to the walk's own targets, so this "
              "probe says nothing about where the film ends.")

    outer = max(found, key=lambda t: t[1]) if found else None
    if outer is not None:
        k, r, side, aa = outer
        print(f"\n  OUTERMOST real terminus: w{k:02d} (r {r:.0f}) {side} end -> "
              f"the film's outer end, i.e. the seam, is at ABS {aa:.1f} deg")
        if args.expect_deg is not None:
            dd = abs((aa - args.expect_deg + 180.0) % 360.0 - 180.0)
            print(f"  known seam {args.expect_deg:.1f} deg -> off by {dd:.1f} deg"
                  + ("   PASS" if dd < 15 else "   FAIL"))


if __name__ == "__main__":
    main()
