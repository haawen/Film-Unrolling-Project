"""Diff two matched_walks.npz (pre / post --enforce-radial-order) per (anchor, winding).

For every row that changed, score BOTH versions by how far the row's radius sits
from its own winding's z-neighbours (median over the +-W nearest anchors of the
SAME version, excluding itself): a correct row agrees with its neighbours (~1 px),
a swapped / jumped row sits ~one spacing (26 px) off. Prints where rows changed and
whether the change moved them toward (fix) or away from (harm) consensus.
"""
import argparse
import numpy as np


def radii(P, cx, cy, a, k):
    p = np.asarray(P[a][k], float)
    return np.hypot(p[:, 0] - cx[a], p[:, 1] - cy[a])


def col_phi(P, cx, cy, seam, k):
    """Seam-relative azimuth of each column (shared by every anchor after
    arc_equalize; median over anchors for robustness)."""
    n = len(P)
    th = np.stack([np.arctan2(np.asarray(P[a][k])[:, 1] - cy[a],
                              np.asarray(P[a][k])[:, 0] - cx[a]) for a in range(0, n, 37)])
    ph = np.angle(np.mean(np.exp(1j * (th - seam)), 0)) % (2 * np.pi)
    return ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--win", type=int, default=5)
    ap.add_argument("--tol", type=float, default=0.5, help="px; row 'changed' above this")
    args = ap.parse_args()
    A = np.load(args.old, allow_pickle=True); B = np.load(args.new, allow_pickle=True)
    z = np.asarray(A["z_anchor"], float)
    cx = np.asarray(A["cx_a"], float); cy = np.asarray(A["cy_a"], float)
    PA, PB = A["paths"], B["paths"]
    n, nw = len(z), len(PA[0])
    W = args.win
    changed = []
    for k in range(nw):
        RA = np.stack([radii(PA, cx, cy, a, k) for a in range(n)])
        RB = np.stack([radii(PB, cx, cy, a, k) for a in range(n)])
        if RB.shape[1] != RA.shape[1]:          # arc_equalize K can differ by a column
            seam = float(A["seam"])
            pa, pb = col_phi(PA, cx, cy, seam, k), col_phi(PB, cx, cy, seam, k)
            o = np.argsort(pb)
            RB = np.stack([np.interp(pa, pb[o], RB[a][o]) for a in range(n)])
        d = np.abs(RA - RB).max(1)
        for a in np.nonzero(d > args.tol)[0]:
            nb = [b for b in range(max(0, a - W), min(n, a + W + 1)) if b != a]
            ea = float(np.median(np.abs(RA[a] - np.median(RA[nb], 0))))
            eb = float(np.median(np.abs(RB[a] - np.median(RB[nb], 0))))
            changed.append((z[a], k, float(np.median(np.abs(RA[a] - RB[a]))), ea, eb))
    print(f"{len(changed)} changed (anchor, winding) rows of {n * nw}")
    if not changed:
        return
    c = np.array(changed)
    fix = c[:, 4] < c[:, 3] - 2; harm = c[:, 4] > c[:, 3] + 2
    print(f"  moved TOWARD z-consensus (fix): {fix.sum()}   AWAY (harm): {harm.sum()}   "
          f"neutral: {(~fix & ~harm).sum()}")
    print(f"  consensus error old -> new: median {np.median(c[:, 3]):.1f} -> "
          f"{np.median(c[:, 4]):.1f} px, p90 {np.percentile(c[:, 3], 90):.1f} -> "
          f"{np.percentile(c[:, 4], 90):.1f}")
    for lab, m in [("w5/w6", np.isin(c[:, 1], [5, 6])), ("other", ~np.isin(c[:, 1], [5, 6]))]:
        if m.any():
            print(f"  {lab}: {m.sum()} rows, z{c[m, 0].min():.0f}-{c[m, 0].max():.0f}, "
                  f"fix {(fix & m).sum()} harm {(harm & m).sum()}, consensus err "
                  f"{np.median(c[m, 3]):.1f} -> {np.median(c[m, 4]):.1f} px")
    print("  rows with z>=780 or harm (z, winding, |dR| old-new, err old, err new):")
    for r, h in zip(changed, harm):
        if r[0] >= 780 or h:
            print(f"    z{r[0]:.0f} w{r[1]}  {r[2]:5.1f}  {r[3]:5.1f} -> {r[4]:5.1f}"
                  + ("  HARM" if h else ""))
    zb = np.histogram(c[:, 0], bins=[470, 500, 530, 600, 700, 760, 800, 1000, 1400, 1950])
    print("  changed rows per z band:", dict(zip([f"{a}-{b}" for a, b in
          zip(zb[1][:-1], zb[1][1:])], zb[0].tolist())))


if __name__ == "__main__":
    main()
