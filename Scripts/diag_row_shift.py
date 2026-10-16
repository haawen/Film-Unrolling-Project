"""Per-row along-film shift in a rendered strip -- finds and CHARACTERISES a tear.

Adjacent strip rows are adjacent CT slices of the same film, so their content is
essentially identical; any systematic column offset between row z and row z-1 is
geometry error, not content. This measures that offset directly, by 1D phase
correlation per column block, and then uses HOW the offset varies with column to
say what caused it:

  * SPOOL CENTRE. A centre error (dx,dy) displaces a column by
    ds ~ dx*sin(phi) - dy*cos(phi), which is independent of radius. So the shift
    is the SAME AMPLITUDE in every winding and completes exactly one cycle per
    turn. Fit that and the residual tells you whether the centre explains it.
  * ONE MIS-TRACKED WINDING. Shift confined to the columns of a single winding,
    zero elsewhere.
  * Z-HEAL / INTERPOLATION. Shift roughly constant across all columns.

Prints shift vs z (to locate the tear), then for the worst z, shift vs column
together with the best-fit centre-error amplitude, so the three cases above can
be told apart instead of guessed at.

Usage:
  python Scripts/diag_row_shift.py --strip .../wholeroll.npy --nblocks 32
"""
import argparse

import numpy as np


def shift_1d(a, b, maxlag):
    """Sub-pixel shift taking b onto a, by parabolic fit on the correlation."""
    a = a - a.mean()
    b = b - b.mean()
    n = 1 << int(np.ceil(np.log2(len(a) + maxlag)) + 1)
    A = np.fft.rfft(a, n)
    B = np.fft.rfft(b, n)
    c = np.fft.irfft(A * np.conj(B), n)
    c = np.concatenate([c[-maxlag:], c[:maxlag + 1]])
    k = int(np.argmax(c))
    if 0 < k < len(c) - 1:
        d = c[k - 1] - 2 * c[k] + c[k + 1]
        sub = 0.5 * (c[k - 1] - c[k + 1]) / d if d != 0 else 0.0
    else:
        sub = 0.0
    return (k - maxlag) + sub, float(c[k] / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip", required=True)
    ap.add_argument("--nblocks", type=int, default=32)
    ap.add_argument("--blocklen", type=int, default=2048)
    ap.add_argument("--maxlag", type=int, default=40)
    ap.add_argument("--z-first", type=int, default=40)
    ap.add_argument("--z", nargs=2, type=int, default=None,
                    help="STRIP ROW range (z - z_first). Keep this inside the "
                         "picture band: the packaging slices outside it are "
                         "featureless, so their correlation peaks are noise that "
                         "saturates at +-maxlag and wins any 'worst row' ranking.")
    ap.add_argument("--min-q", type=float, default=0.35,
                    help="Reject a block whose normalised correlation peak is "
                         "below this -- it has nothing to align.")
    ap.add_argument("--min-blocks", type=int, default=8,
                    help="Rows with fewer surviving blocks are not scored.")
    ap.add_argument("--pitch", type=float, default=1130.14)
    ap.add_argument("--cols-per-turn", type=float, default=None,
                    help="Columns per winding, for the centre-error fit. "
                         "Defaults to total_cols / --n-windings.")
    ap.add_argument("--n-windings", type=int, default=34)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    A = np.load(args.strip, mmap_mode="r")
    Z, C = A.shape
    z0, z1 = (0, Z) if args.z is None else args.z
    starts = np.linspace(0.02 * C, 0.95 * C - args.blocklen,
                         args.nblocks).astype(int)
    print(f"{args.strip} {A.shape}; {args.nblocks} blocks of {args.blocklen} "
          f"cols, max lag {args.maxlag}")

    S = np.full((Z, args.nblocks), np.nan, np.float32)
    Q = np.zeros((Z, args.nblocks), np.float32)
    prev = None
    for z in range(z0, z1):
        row = np.asarray(A[z], dtype=np.float32)
        if prev is not None:
            for j, s in enumerate(starts):
                a = prev[s:s + args.blocklen]
                b = row[s:s + args.blocklen]
                if a.std() < 1e-3 or b.std() < 1e-3:
                    continue
                S[z, j], Q[z, j] = shift_1d(a, b, args.maxlag)
        prev = row

    S[Q < args.min_q] = np.nan
    nb = np.sum(np.isfinite(S), axis=1)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(S, axis=1)
        rob = np.nanmedian(np.abs(S - med[:, None]), axis=1)
    med[nb < args.min_blocks] = np.nan
    print(f"\nrows scored: {int(np.isfinite(med).sum())} of {z1 - z0} "
          f"(rest had < {args.min_blocks} blocks above q={args.min_q})")

    # A tear is a step, so rank by |median shift| AND require the blocks to
    # agree: incoherent large shifts are content or noise, not geometry.
    score = np.where(np.isfinite(med), np.abs(med) / (1.0 + rob), 0.0)
    print(f"\nrow-to-row shift, worst {args.top} rows "
          f"(coherent = large median, small spread):")
    order = np.argsort(-np.nan_to_num(score))[:args.top]
    for z in sorted(order):
        vx = z * 1504.0 / (Z - 1)
        print(f"  z{z + args.z_first:5d} (video x {vx:4.0f})  median shift "
              f"{med[z]:+7.2f} px   spread {rob[z]:6.2f}   "
              f"blocks {nb[z]:3d}   score {score[z]:5.2f}")

    print(f"\ncumulative drift (sum of median shifts) every 100 rows, "
          f"z{z0 + args.z_first}-{z1 + args.z_first}:")
    cum = np.nancumsum(np.nan_to_num(med))
    for z in range(z0, z1, 100):
        print(f"  z{z + args.z_first:5d}  cum {cum[z]:+8.2f} px")

    zb = int(order[0])
    cpt = args.cols_per_turn or (C / args.n_windings)
    print(f"\nshift vs column at the worst row z{zb + args.z_first} "
          f"(cols/turn {cpt:.0f}):")
    phi = 2 * np.pi * (starts + args.blocklen / 2) / cpt
    y = S[zb]
    ok = np.isfinite(y)
    print(f"  {'col':>9} {'winding':>8} {'shift':>8}")
    for j in np.nonzero(ok)[0]:
        print(f"  {starts[j]:9d} {starts[j]/cpt:8.2f} {y[j]:8.2f}")
    if ok.sum() >= 3:
        M = np.column_stack([np.sin(phi[ok]), -np.cos(phi[ok]),
                             np.ones(ok.sum())])
        coef, *_ = np.linalg.lstsq(M, y[ok], rcond=None)
        res = y[ok] - M @ coef
        print(f"\n  centre-error fit  dx {coef[0]:+.2f} px, dy {coef[1]:+.2f} px, "
              f"constant {coef[2]:+.2f} px")
        print(f"  residual rms {np.sqrt((res**2).mean()):.2f} px vs "
              f"signal rms {np.sqrt((y[ok]**2).mean()):.2f} px")


if __name__ == "__main__":
    main()
