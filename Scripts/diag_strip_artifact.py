"""Localise a rendered-strip artifact by differencing two strips of equal shape.

Built for the dense-vs-sparse new-scan renders, which came out the same shape
(2361, 259281) from the same seam/pitch, so a block-wise |difference| points
straight at the region where one went wrong.

Also reports, per column block, the strip's own vertical (z) discontinuity
energy: a geometry tear shows up as hard steps ALONG z that content does not
produce, so it flags the artifact even without a reference strip.

Usage:
  python Scripts/diag_strip_artifact.py --a <dense>/wholeroll.npy \
      --b <sparse>/wholeroll.npy --n-windings 34
"""
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="strip under test")
    ap.add_argument("--b", default=None, help="reference strip (same shape)")
    ap.add_argument("--n-windings", type=int, default=34,
                    help="Only to translate a column into a winding index; the "
                         "render lays windings out contiguously, so this is "
                         "approximate (arc_equalize gives them unequal widths).")
    ap.add_argument("--rblock", type=int, default=64)
    ap.add_argument("--cblock", type=int, default=2048)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    A = np.load(args.a, mmap_mode="r")
    print(f"A {args.a} {A.shape}")
    Z, C = A.shape
    nr, nc = Z // args.rblock, C // args.cblock

    # ── self-diagnostic: hard steps along z ──
    print("\nvertical (z) step energy per column block "
          "(a geometry tear steps along z; content does not):")
    step = np.zeros(nc)
    for j in range(nc):
        sl = np.asarray(A[:, j * args.cblock:(j + 1) * args.cblock],
                        dtype=np.float32)
        step[j] = float(np.mean(np.abs(np.diff(sl, axis=0))))
    med = float(np.median(step))
    order = np.argsort(-step)[:args.top]
    for j in sorted(order):
        c0 = j * args.cblock
        w = c0 / max(1, C) * args.n_windings
        print(f"  cols {c0:7d}-{c0+args.cblock:7d}  (~winding {w:5.1f})  "
              f"step {step[j]:7.3f}  = {step[j]/med:5.2f}x median")

    if not args.b:
        return

    B = np.load(args.b, mmap_mode="r")
    print(f"\nB {args.b} {B.shape}")
    if A.shape != B.shape:
        raise SystemExit("shapes differ -- cannot difference directly")

    print(f"\nblock |A-B|, blocks {args.rblock} rows x {args.cblock} cols:")
    D = np.zeros((nr, nc), dtype=np.float32)
    for i in range(nr):
        a = np.asarray(A[i * args.rblock:(i + 1) * args.rblock, :],
                       dtype=np.float32)
        b = np.asarray(B[i * args.rblock:(i + 1) * args.rblock, :],
                       dtype=np.float32)
        d = np.abs(a - b)
        for j in range(nc):
            D[i, j] = d[:, j * args.cblock:(j + 1) * args.cblock].mean()
    dmed = float(np.median(D))
    print(f"  median block diff {dmed:.4f}, max {D.max():.4f} "
          f"({D.max()/max(dmed,1e-6):.1f}x median)")

    flat = np.argsort(-D, axis=None)[:args.top]
    print(f"\n  worst {args.top} blocks (row = strip z index, col = arc):")
    for k in flat:
        i, j = np.unravel_index(k, D.shape)
        r0 = i * args.rblock
        c0 = j * args.cblock
        w = c0 / max(1, C) * args.n_windings
        print(f"    z-rows {r0:5d}-{r0+args.rblock:5d}  cols {c0:7d}-"
              f"{c0+args.cblock:7d}  (~winding {w:5.1f})  diff {D[i,j]:.4f}"
              f"  = {D[i,j]/max(dmed,1e-6):5.1f}x median")

    # collapse to see whether the damage is bounded in z, in arc, or both
    print("\n  row profile (mean over columns), worst 12 z-blocks:")
    rp = D.mean(axis=1)
    for i in np.argsort(-rp)[:12]:
        print(f"    z-rows {i*args.rblock:5d}-{(i+1)*args.rblock:5d}  "
              f"{rp[i]:.4f}  = {rp[i]/max(float(np.median(rp)),1e-6):5.1f}x median")
    print("\n  col profile (mean over rows), worst 12 arc-blocks:")
    cp = D.mean(axis=0)
    for j in np.argsort(-cp)[:12]:
        print(f"    cols {j*args.cblock:7d}-{(j+1)*args.cblock:7d}  "
              f"(~winding {j*args.cblock/max(1,C)*args.n_windings:5.1f})  "
              f"{cp[j]:.4f}  = {cp[j]/max(float(np.median(cp)),1e-6):5.1f}x median")


if __name__ == "__main__":
    main()
