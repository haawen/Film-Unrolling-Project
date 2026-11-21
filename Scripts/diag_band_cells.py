"""Per-cell wall band amplitude in two strips (before / after band_correct).

For every frame cell (pitch px along the film) over the picture band: the masked
mean over plain-wall pixels (locally smoothed value < thr) per row, then the
band-scale std (minus a z-gaussian of `--keep` rows). Same metric as the
prototype that showed a ~50% reduction on three windings.
"""
import argparse
import numpy as np
from scipy.ndimage import binary_erosion, gaussian_filter, gaussian_filter1d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--z-index", required=True)
    ap.add_argument("--pitch", type=float, default=1130.13)
    ap.add_argument("--thr", type=float, default=0.45)
    ap.add_argument("--keep", type=float, default=150)
    args = ap.parse_args()
    A = np.load(args.a, mmap_mode="r"); B = np.load(args.b, mmap_mode="r")
    z = np.load(args.z_index); rows = np.nonzero((z >= 600) & (z <= 1900))[0]
    r0, r1 = rows[0], rows[-1] + 1
    N = A.shape[1]; n = int(N // args.pitch)
    ra, rb, cnt = [], [], []
    for i in range(n):
        c0, c1 = int(i * args.pitch), int((i + 1) * args.pitch)
        a = np.asarray(A[r0:r1, c0:c1], np.float32); b = np.asarray(B[r0:r1, c0:c1], np.float32)
        m = binary_erosion(gaussian_filter(a, (2, 8)) < args.thr, iterations=2)
        k = m.sum(1); ok = k > 50
        if ok.mean() < 0.6:
            continue
        pa = (a * m).sum(1) / np.maximum(k, 1); pb = (b * m).sum(1) / np.maximum(k, 1)
        pa = np.interp(np.arange(len(pa)), np.nonzero(ok)[0], pa[ok])
        pb = np.interp(np.arange(len(pb)), np.nonzero(ok)[0], pb[ok])
        ra.append((pa - gaussian_filter1d(pa, args.keep)).std())
        rb.append((pb - gaussian_filter1d(pb, args.keep)).std()); cnt.append(i)
    ra, rb = np.array(ra), np.array(rb)
    print(f"{len(ra)} of {n} cells with enough wall")
    print(f"wall band-scale std: before median {np.median(ra):.4f}  after {np.median(rb):.4f}"
          f"  (ratio median {np.median(rb / ra):.2f}, p10 {np.percentile(rb / ra, 10):.2f}, "
          f"p90 {np.percentile(rb / ra, 90):.2f})")
    for q in (0, len(cnt) // 4, len(cnt) // 2, 3 * len(cnt) // 4, len(cnt) - 1):
        print(f"  cell {cnt[q]:3d}: {ra[q]:.4f} -> {rb[q]:.4f}")


if __name__ == "__main__":
    main()
