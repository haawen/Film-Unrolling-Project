"""Remove the broad soft BANDS (fixed-z, tens-to-~100 rows wide) from a new-scan strip.

What they are (measured 2026-10-21, doc newscan-fullwidth s14): fixed in the
VOLUME (they sit at nearly the same z over many frames and shots), produced by the
merged reconstruction, and confined to the thin emulsion layer -- the film base and
the air gap beside it carry none (corr ~0), so those cannot serve as background.
In the strip domain (norm_slice'd CT; high value = dark in the positive video) the
bands are ADDITIVE and level-dependent: full strength in the bright wall (s near
the base level ~0.38; slope 1.0 vs the wall band), ~0.74 at s 0.36-0.42, ~0.3 in
the mid tones, none in the dark content (which norm_slice clips at 1.0).

Data-only estimate:
  1. wall mask = pixels whose LOCALLY SMOOTHED value is below `--wall-thr` (a
     per-pixel threshold would let a band push its own pixels out of the mask and
     bias the estimate), eroded away from content;
  2. F = masked mean of s in `--bw`-column bins, smoothed `--sz` rows in z and
     `--sa` columns along the film (several frames, so the picture averages out
     while the volume-fixed bands survive);
  3. b = F - gaussian_z(F, `--sz-keep`): band-scale structure only, so the smooth
     falloff across the film width (present in GT too) is kept;
  4. s_corr = s - w(s) * b, w = 1 below `--w-full`, fading to 0 at `--w-zero`.
Writes a strip in the SAME value domain, so every downstream chain consumes it.

  python -m unwrapping.inr.band_correct <in.npy> <out.npy> --z-index z_index.npy
"""
import argparse

import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, gaussian_filter1d


def _row_median_along(F, ok, win_bins):
    """Per row, running MEDIAN of F over `win_bins` bins along the film, ignoring
    unsupported bins (filled by interpolation along the row first)."""
    from scipy.ndimage import median_filter
    G = F.copy()
    x = np.arange(F.shape[1])
    for r in range(F.shape[0]):
        k = ok[r]
        if k.sum() >= 2 and not k.all():
            G[r, ~k] = np.interp(x[~k], x[k], F[r, k])
    return median_filter(G, size=(1, int(win_bins) | 1), mode="nearest")


def band_field(s, sa=3000.0, sz=1.5, sz_keep=150.0, bw=100, wall_thr=0.45,
               erode=2, rows=None, agg="mean", med_win=6000.0):
    """Binned additive band field b (Z, nbins) and bin centres.

    agg="median" encodes that the bands are PURELY VERTICAL in the video, i.e.
    constant along the film over many frames: per row, the wall level is the
    running median over `med_win` columns (dozens of bins), so anything present in
    fewer than half of them -- characters, objects, one shot's shading -- is
    rejected, and only then smoothed by `sa`. That robustness is what allows a
    narrower `sz_keep` (the 60-100-row bands a 150-row cutoff partly keeps)."""
    Z, N = s.shape
    rows = np.ones(Z, bool) if rows is None else rows
    nb = int(np.ceil(N / bw))
    num = np.zeros((Z, nb), np.float32); den = np.zeros((Z, nb), np.float32)
    for j in range(nb):                                  # bin-wise: never a full copy
        c0, c1 = j * bw, min(N, (j + 1) * bw)
        p0, p1 = max(0, c0 - 20), min(N, c1 + 20)        # context for the smoothing
        blk = np.nan_to_num(np.asarray(s[:, p0:p1], np.float32))
        loc = gaussian_filter(blk, (2.0, 8.0))
        m = binary_erosion(loc < wall_thr, iterations=erode)[:, c0 - p0:c1 - p0]
        m[~rows] = False
        b = blk[:, c0 - p0:c1 - p0]
        num[:, j] = (b * m).sum(1); den[:, j] = m.sum(1)
    sb = sa / bw
    if agg == "median":
        num_z = gaussian_filter1d(num, sz, axis=0, mode="nearest")
        den_z = gaussian_filter1d(den, sz, axis=0, mode="nearest")
        okb = den_z > 0.05 * bw                          # this bin has wall here
        Fb = np.where(okb, num_z / np.maximum(den_z, 1e-6), 0.0)
        Fm = _row_median_along(Fb, okb, med_win / bw)
        F = gaussian_filter1d(Fm, sb, axis=1, mode="nearest")
        frac = gaussian_filter1d(okb.astype(np.float32), med_win / bw / 2.0,
                                 axis=1, mode="nearest")
        sup = (frac > 0.25) & rows[:, None]              # wall in enough of the window
        F = np.where(sup, F, 0.0)
    else:
        num_s = gaussian_filter(num, (sz, sb), mode="nearest")
        den_s = gaussian_filter(den, (sz, sb), mode="nearest")
        sup = (den_s > 0.03 * bw) & rows[:, None]        # enough wall nearby
        F = np.where(sup, num_s / np.maximum(den_s, 1e-6), 0.0)
    w = gaussian_filter1d(sup.astype(np.float32), sz_keep, axis=0, mode="nearest")
    L = gaussian_filter1d(F, sz_keep, axis=0, mode="nearest") / np.maximum(w, 1e-6)
    band = np.where(sup, F - L, 0.0).astype(np.float32)
    return band, (np.arange(nb) + 0.5) * bw


def apply(s, band, centres, w_full=0.40, w_zero=0.60, out=None, chunk=20000):
    Z, N = s.shape
    out = np.empty((Z, N), np.float32) if out is None else out
    step = centres[1] - centres[0]
    for c0 in range(0, N, chunk):
        c1 = min(N, c0 + chunk)
        jf = np.clip((np.arange(c0, c1) + 0.5 - centres[0]) / step, 0, len(centres) - 1)
        j0 = np.floor(jf).astype(int); j1 = np.minimum(j0 + 1, len(centres) - 1)
        t = (jf - j0).astype(np.float32)
        b = band[:, j0] * (1 - t) + band[:, j1] * t
        blk = np.asarray(s[:, c0:c1], np.float32)
        wgt = np.clip((w_zero - blk) / (w_zero - w_full), 0.0, 1.0)
        out[:, c0:c1] = blk - wgt * b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--z-index", default=None,
                    help="z_index.npy of the strip; restricts estimation and "
                         "correction to --z-lo..--z-hi (the picture band)")
    ap.add_argument("--z-lo", type=int, default=520)
    ap.add_argument("--z-hi", type=int, default=1930)
    ap.add_argument("--sa", type=float, default=3000.0)
    ap.add_argument("--sz", type=float, default=1.5)
    ap.add_argument("--sz-keep", type=float, default=150.0)
    ap.add_argument("--wall-thr", type=float, default=0.45)
    ap.add_argument("--w-full", type=float, default=0.40)
    ap.add_argument("--w-zero", type=float, default=0.60)
    ap.add_argument("--agg", choices=["mean", "median"], default="mean",
                    help="median = enforce the vertical-only (constant along the "
                         "film) nature of the bands; see band_field")
    ap.add_argument("--med-win", type=float, default=6000.0,
                    help="--agg median: window along the film (columns)")
    ap.add_argument("--save-band", default=None)
    args = ap.parse_args()
    s = np.load(args.inp)                    # in RAM: column slices of a memmap crawl
    rows = None
    if args.z_index:
        z = np.load(args.z_index)
        rows = (z >= args.z_lo) & (z <= args.z_hi)
    band, centres = band_field(s, args.sa, args.sz, args.sz_keep,
                               wall_thr=args.wall_thr, rows=rows, agg=args.agg,
                               med_win=args.med_win)
    sel = band[rows] if rows is not None else band
    print(f"band field over picture band: std {sel.std():.4f}, p1/p99 "
          f"{np.percentile(sel, [1, 99])}", flush=True)
    if args.save_band:
        np.savez(args.save_band, band=band, centres=centres)
    out = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float32, shape=s.shape)
    apply(s, band, centres, args.w_full, args.w_zero, out=out)
    out.flush()
    print(f"wrote {args.out} {s.shape}")


if __name__ == "__main__":
    main()
