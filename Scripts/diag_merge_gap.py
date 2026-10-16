"""Is it safe to merge these walk caches across their z gaps?

`merge_walk_caches.py` hands the tracker anchors from chunks that are hundreds
of slices apart, and the tracker matches windings by RADIUS within
`--match-tol-px`.  If a winding's radius drifts by more than about half the
winding pitch across a gap, the tracker will hand it to its neighbour and the
render silently interleaves two windings.  The new-scan notes checked this once
for the 201-slice picture-band gaps (outermost winding drifting 2-3 px against a
26 px pitch) and flagged that it must be re-checked before merging any other
set -- which is exactly what adding the perf-band chunks does, stretching the
gaps to 250 slices (z250->z500) and 180 (z1880->z2060).

Prints, per chunk, the radius ladder of the tracked windings, then the drift of
the outermost winding across each gap against the pitch and the tolerance.

    python Scripts/diag_merge_gap.py --caches a/matched_walks.npz b/... [...]
"""

import argparse

import numpy as np


def ladder(path):
    """Median radius of each winding in a chunk, plus its z range."""
    d = np.load(path, allow_pickle=True)
    z = np.asarray(d["z_anchor"], float)
    cx = np.asarray(d["cx_a"], float)
    cy = np.asarray(d["cy_a"], float)
    paths = d["paths"]
    a = len(z) // 2                       # middle anchor of the chunk
    r = []
    for p in paths[a]:
        p = np.asarray(p, float)
        r.append(float(np.median(np.hypot(p[:, 0] - cx[a], p[:, 1] - cy[a]))))
    return z, np.array(sorted(r))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--match-tol-px", type=float, default=15.0)
    args = ap.parse_args()

    info = []
    for c in args.caches:
        z, r = ladder(c)
        info.append((float(z.mean()), z.min(), z.max(), r, c))
        pitch = np.median(np.diff(r)) if len(r) > 1 else np.nan
        print(f"{c}")
        print(f"  z {z.min():.0f}-{z.max():.0f}   {len(r)} windings   "
              f"r {r.min():.0f}-{r.max():.0f}   median pitch {pitch:.1f} px")
    info.sort()

    print("\n=== drift of the OUTERMOST winding across each gap ===")
    print(f"(tracker matches within {args.match_tol_px:.0f} px; a drift above "
          f"half the winding pitch would assign it to its neighbour)")
    ok = True
    for (za, _, z0a, ra, ca), (zb, z1b, _, rb, cb) in zip(info, info[1:]):
        gap = z1b - z0a
        d_out = rb[-1] - ra[-1]
        pitch = 0.5 * (np.median(np.diff(ra)) + np.median(np.diff(rb)))
        flag = "OK" if abs(d_out) < min(args.match_tol_px, 0.5 * pitch) else "RISK"
        if flag == "RISK":
            ok = False
        print(f"  z{z0a:.0f} -> z{z1b:.0f}   gap {gap:5.0f} sl   "
              f"r_out {ra[-1]:.0f} -> {rb[-1]:.0f}  drift {d_out:+6.1f} px   "
              f"pitch {pitch:.1f}   {flag}")
    print("\nVERDICT:", "safe to merge" if ok else
          "NOT safe -- a winding could be handed to its neighbour")


if __name__ == "__main__":
    main()
