"""Geometry-aware radiometric normalization of an unrolled strip.

Divides out a content-masked, anisotropic illumination field toward a single
GLOBAL background level, so brightness/shading bands that run along the film
(constant z, varying arc) or across it (constant arc, varying z) are flattened
while the picture content is left alone. Output is in the SAME value domain as
wholeroll.npy, so make_film_video consumes it unchanged.

Written for the old scan's v9 strip, where the residual artifact turned out to be
BROAD BRIGHTNESS/SHADING rather than geometry -- four geometric hypotheses had
been tested and rejected first. The same question is open on the new full-width
scan, hence the parameterisation.

SIGMAS SCALE WITH THE SCAN. The defaults here are the old scan's (sz 3, sa 60,
frame pitch 905). The new full-width reconstruction is 1.2508x finer in-plane
(pitch 1130), so pass --sz 3.75 --sa 75 to keep the field the same size in film
units. Too small an --sa eats picture content; too large stops following the band.

Usage:
  python -m unwrapping.inr.geo_normalize --src .../wholeroll.npy \
      --dst .../wholeroll_geonorm.npy --sz 3.75 --sa 75
"""
import argparse
import time

import numpy as np
from scipy import ndimage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--sz", type=float, default=3.0,
                    help="Illumination-field sigma ALONG z (rows).")
    ap.add_argument("--sa", type=float, default=60.0,
                    help="Illumination-field sigma ALONG the arc (columns).")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Correction strength; 1.0 = full de-shade.")
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--pad", type=int, default=300)
    args = ap.parse_args()

    mm = np.load(args.src, mmap_mode="r")
    H, W = mm.shape
    print(f"strip {mm.shape}  sz={args.sz} sa={args.sa} alpha={args.alpha}")

    # global scale + global background target level (from a column subsample)
    cols_s = np.arange(2000, W - 2000, 30)
    samp = np.asarray(mm[:, cols_s], np.float32)
    lo, hi = np.percentile(samp, [1, 99])
    to_v = lambda s: 1.0 - np.clip((s - lo) / (hi - lo + 1e-6), 0, 1)
    vs = to_v(samp)
    bg = ~ndimage.binary_dilation((vs < 0.40) | (vs > 0.97), iterations=4)
    G = float(np.median(vs[bg]))
    print(f"lo/hi {lo:.4f}/{hi:.4f}  global bg level G={G:.4f}")

    out = np.lib.format.open_memmap(args.dst, mode="w+", dtype=np.float32,
                                    shape=(H, W))
    t0 = time.time()
    for a in range(0, W, args.chunk):
        b = min(W, a + args.chunk)
        a2, b2 = max(0, a - args.pad), min(W, b + args.pad)
        s = np.asarray(mm[:, a2:b2], np.float32)
        v = to_v(s)
        m = (~ndimage.binary_dilation((v < 0.40) | (v > 0.97),
                                      iterations=4)).astype(np.float32)
        num = ndimage.gaussian_filter(v * m, (args.sz, args.sa))
        den = ndimage.gaussian_filter(m, (args.sz, args.sa))
        F = np.where(den > 1e-3, num / np.maximum(den, 1e-6), G)
        F = np.maximum(F, 1e-3)
        v_corr = np.clip(v * np.power(G / F, args.alpha), 0, 1)
        s_corr = lo + (hi - lo) * (1.0 - v_corr)
        out[:, a:b] = s_corr[:, (a - a2):(a - a2) + (b - a)]
        print(f"  cols {a}-{b}  ({time.time() - t0:.0f}s)", flush=True)
    out.flush()
    print(f"saved {args.dst}")


if __name__ == "__main__":
    main()
