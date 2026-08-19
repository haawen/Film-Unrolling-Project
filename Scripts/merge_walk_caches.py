"""Merge several per-chunk walk caches into one `walk_anchors.npz`.

The v13 renderer already handles gapped anchors: it fits `interp1d` per winding
over `z_anchor` with `bounds_error=False, fill_value=(X[0], X[-1])`, so between
chunks it interpolates and OUTSIDE the anchor range it clamps to the nearest
anchor's geometry. Feeding it the union of the chunks that walked cleanly, and
simply omitting the ones that failed, therefore renders the whole stack while
reusing the nearest working z for the parts that have no usable walk -- with no
new walking.

Safe here because the roll barely drifts: the outermost winding sits at r 1700,
1698, 1699, 1696, 1693, 1690, 1691 across z560..z1880, i.e. ~2-3 px over 220-slice
gaps, against a 26 px winding pitch and a 15 px `--match-tol-px`. Check that again
before merging a different set -- if a gap's drift approached half the pitch the
z-sequential tracker could assign a winding to its neighbour across the gap.

Bore phantoms take care of themselves: a track that only exists in one chunk gets
coverage ~20/160, so a `--min-coverage` above that drops it.

Usage:
  python Scripts/merge_walk_caches.py --out <dir>/walk_anchors.npz \
      --caches a/walk_anchors.npz b/walk_anchors.npz ...
"""
import argparse
import math
import os

import numpy as np


def refit_centers(z, cx, cy, n_iter=4, clip=3.0):
    """Repair find_spool_center outliers in the per-anchor spool-centre track.

    The true centre track is very nearly linear: over z520-1960 it runs
    cx 1904->1873 and cy 1879->1895, and WITHIN every well-behaved batch the
    residual from a straight line is <=0.7 px. But a single find_spool_center call
    scatters by tens of px on this scan, and one bad sample poisons every anchor
    it is interpolated across -- the dense run's b0 sample at z480 came back
    cx 1812 against ~1904 for its neighbours, a ~90 px outlier that the batch's own
    log flagged ("end-to-end drift 86.5px; max residual from a straight line in z
    48.9px"). A 90 px centre error at r=1000 is ~5deg, i.e. ~90 px of arc: exactly
    the cross-z misalignment class that feeds resample_phi's common azimuth grid.

    Iterative sigma-clipping rather than a plain least-squares fit, because the
    bad samples are what we are trying to remove and they would otherwise drag the
    line toward themselves.

    The line is used only to DETECT the outliers; the repair then INTERPOLATES
    from the surviving anchors. Forcing every anchor onto the fitted line instead
    would import the line's own error wherever the real track curves -- measured
    at ~3 px at the z1960 end, small but pointless to introduce when the good part
    of the track is already correct.
    """
    z = np.asarray(z, float)
    out = []
    for v in (np.asarray(cx, float), np.asarray(cy, float)):
        keep = np.ones(len(z), bool)
        for _ in range(n_iter):
            p = np.polyfit(z[keep], v[keep], 1)
            r = v - np.polyval(p, z)
            mad = np.median(np.abs(r - np.median(r))) + 1e-9
            new = np.abs(r - np.median(r)) < clip * 1.4826 * mad
            if new.sum() < 0.5 * len(z) or (new == keep).all():
                break
            keep = new
        fixed = v.copy()
        if (~keep).any() and keep.any():
            fixed[~keep] = np.interp(z[~keep], z[keep], v[keep])
        out.append((fixed, keep, float(np.max(np.abs(v - fixed)))))
    return out[0], out[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--z-range", nargs="+", default=None,
                    help="Per-cache z window as lo:hi (one per --caches entry, "
                         "'-' to keep all). Lets a later cache REPLACE part of an "
                         "earlier one without leaving duplicate z, which would "
                         "break the render's interp1d (it needs strictly "
                         "increasing z).")
    ap.add_argument("--refit-centers", action="store_true",
                    help="Replace per-anchor centres with a robust line in z, "
                         "rejecting find_spool_center outliers (see refit_centers).")
    args = ap.parse_args()

    if args.z_range and len(args.z_range) != len(args.caches):
        raise SystemExit(f"--z-range needs one entry per cache "
                         f"({len(args.caches)}), got {len(args.z_range)}")

    Z, CX, CY, P = [], [], [], []
    seam = None
    for ci, c in enumerate(args.caches):
        d = np.load(c, allow_pickle=True)
        s = float(d["seam"])
        if seam is None:
            seam = s
        elif abs((math.degrees(s - seam) + 180) % 360 - 180) > 0.01:
            raise SystemExit(
                f"seam mismatch: {c} has {math.degrees(s):.2f} deg, first cache "
                f"has {math.degrees(seam):.2f}. Merging caches walked against "
                f"different seams would put their phi grids out of register.")
        z = np.asarray(d["z_anchor"], float)
        cxc = np.asarray(d["cx_a"], float)
        cyc = np.asarray(d["cy_a"], float)
        pl = list(d["paths"])
        keep = np.ones(len(z), bool)
        win = ""
        if args.z_range and args.z_range[ci] != "-":
            lo, _, hi = args.z_range[ci].partition(":")
            lo = float(lo) if lo else -np.inf
            hi = float(hi) if hi else np.inf
            keep = (z >= lo) & (z <= hi)
            win = f"  [kept z{lo:.0f}-{hi:.0f}: {int(keep.sum())}/{len(z)}]"
        z = z[keep]; cxc = cxc[keep]; cyc = cyc[keep]
        pl = [p for p, k in zip(pl, keep) if k]
        if len(z) == 0:
            print(f"  {os.path.relpath(c):<55} EMPTY after z-range, skipped")
            continue
        Z.append(z); CX.append(cxc); CY.append(cyc); P.extend(pl)
        nw = [len(p) for p in pl]
        print(f"  {os.path.relpath(c):<55} {len(z):4d} anchors z{z.min():.0f}-"
              f"{z.max():.0f}, {min(nw)}-{max(nw)} paths/anchor{win}")

    z = np.concatenate(Z); cx = np.concatenate(CX); cy = np.concatenate(CY)
    order = np.argsort(z)
    z = z[order]; cx = cx[order]; cy = cy[order]
    P = [P[i] for i in order]

    gaps = np.diff(z)
    big = np.nonzero(gaps > 1.5)[0]
    print(f"\nmerged: {len(z)} anchors, z{z.min():.0f}-{z.max():.0f}, "
          f"seam {math.degrees(seam):.2f} deg")
    print(f"  {len(big)} gaps: " +
          ", ".join(f"z{z[i]:.0f}->{z[i+1]:.0f} ({gaps[i]:.0f})" for i in big))
    print(f"  centre spread over the merged set: cx {cx.min():.1f}-{cx.max():.1f}, "
          f"cy {cy.min():.1f}-{cy.max():.1f}")

    if args.refit_centers:
        (cx_f, keep_x, dx), (cy_f, keep_y, dy) = refit_centers(z, cx, cy)
        rej = np.nonzero(~(keep_x & keep_y))[0]
        print(f"  refit centres: rejected {len(rej)} of {len(z)} anchors as "
              f"find_spool_center outliers"
              + (f" (z {z[rej].min():.0f}-{z[rej].max():.0f})" if len(rej) else ""))
        print(f"    max |old-new|: cx {dx:.1f}px, cy {dy:.1f}px")
        print(f"    new track: cx {cx_f[0]:.1f}->{cx_f[-1]:.1f}, "
              f"cy {cy_f[0]:.1f}->{cy_f[-1]:.1f} over z{z[0]:.0f}-{z[-1]:.0f}")
        cx, cy = cx_f, cy_f

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    obj = np.empty(len(P), dtype=object)
    for i, p in enumerate(P):
        obj[i] = p
    np.savez(args.out, z_anchor=z, cx_a=cx, cy_a=cy, seam=seam, paths=obj)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
