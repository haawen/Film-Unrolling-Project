"""Refine the OLD -> NEW transform around the coarse solution, and settle the
z-direction convention at the same time.

Coarse result (Scripts/diag_zip_resize_match.py --rot-sweep, 800px working res):
MIRROR = True, theta ~= 134-140 deg, translation ~(-17, +3) old px, consistent
across two independent slices, top-8 all mirrored and contiguous in angle. Note
137 deg is NOT a multiple of 90, which is why the 8 axis-convention candidates all
failed -- the relation is a mirror plus a free rotation, as "merged_RIGID" implies.

Two things this settles:
  1. theta and scale, refined at high resolution with a fine grid. If the content
     truly matches (same scroll, same movie), the peak must SHARPEN and rise well
     above the coarse 0.14; if it stays broad and low, the pixel content does not
     match and no transform will be trustworthy.
  2. THE Z DIRECTION -- never tested until now. Old kept band z832-2015 is the
     picture area and the zip's picture band is z470-1975; I assumed they run the
     same way round. If z is flipped, every pair compared so far matched one edge
     of the film against the other, which would break content matching while
     leaving geometry-based estimators indifferent (the winding curves change by
     only 3.5-7px over 350 slices). Scoring both pairings answers it directly.

Usage:
  python Scripts/diag_zip_refine.py --old-dir 01_Mickey_hdf \
      --zip-dir 01_Mickey_sprockets --out-dir <dir> --old-z 848 \
      --zip-z-same 490 --zip-z-flip 1960 --theta 137 --theta-window 8
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter, rotate, zoom


def downsample(a, s):
    """Downsample by s WITH an anti-alias prefilter (omitting it is what wrecked
    the first attempt: 21/26px rings aliased into noise)."""
    if abs(s - 1.0) < 1e-6:
        return a.astype(np.float32)
    sigma = 0.5 * np.sqrt(max(s * s - 1.0, 0.0))
    return zoom(gaussian_filter(a.astype(np.float32), sigma), 1.0 / s, order=1)


def prep(a, hp):
    b = a.astype(np.float32)
    b = b - gaussian_filter(b, hp)
    b -= b.mean()
    n = np.linalg.norm(b)
    return b / n if n > 0 else b


def crop(a, half):
    cy, cx = a.shape[0] / 2, a.shape[1] / 2
    y0 = max(0, int(cy - half)); x0 = max(0, int(cx - half))
    return a[y0:y0 + 2 * half, x0:x0 + 2 * half]


def ncc_peak(A, B):
    n = (max(A.shape[0], B.shape[0]), max(A.shape[1], B.shape[1]))
    Ap = np.zeros(n, np.float32); Ap[:A.shape[0], :A.shape[1]] = A
    Bp = np.zeros(n, np.float32); Bp[:B.shape[0], :B.shape[1]] = B
    cc = np.fft.irfft2(np.fft.rfft2(Ap) * np.conj(np.fft.rfft2(Bp)), s=n)
    k = int(np.argmax(cc))
    dy, dx = np.unravel_index(k, cc.shape)
    if dy > n[0] // 2:
        dy -= n[0]
    if dx > n[1] // 2:
        dx -= n[1]
    return float(cc.flat[k]), int(dy), int(dx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-dir", required=True)
    ap.add_argument("--zip-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--old-z", type=int, required=True)
    ap.add_argument("--zip-z-same", type=int, required=True)
    ap.add_argument("--zip-z-flip", type=int, required=True)
    ap.add_argument("--theta", type=float, default=137.0)
    ap.add_argument("--theta-window", type=float, default=8.0)
    ap.add_argument("--theta-step", type=float, default=0.5)
    ap.add_argument("--scales", type=float, nargs=3, default=[1.235, 1.265, 0.005])
    ap.add_argument("--half", type=int, default=1300)
    ap.add_argument("--work", type=int, default=2000, help="working resolution")
    ap.add_argument("--hp", type=float, default=10.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    oldmap = {}
    for p in glob.glob(os.path.join(args.old_dir, "*.h5")):
        m = re.search(r"_(\d+)\.h5$", os.path.basename(p))
        if m:
            oldmap[int(m.group(1))] = p
    with h5py.File(oldmap[args.old_z], "r") as f:
        O = np.asarray(f["image"], np.float32)
    A0 = crop(O, args.half)
    fo = A0.shape[0] / args.work
    A = prep(downsample(A0, fo), args.hp)
    print(f"old z{args.old_z} {O.shape} -> crop {A0.shape} -> work {A.shape}")

    thetas = np.arange(args.theta - args.theta_window,
                       args.theta + args.theta_window + 1e-9, args.theta_step)
    scales = np.arange(args.scales[0], args.scales[1] + 1e-9, args.scales[2])
    results = {}
    for tag, zz in (("same", args.zip_z_same), ("flip", args.zip_z_flip)):
        with h5py.File(os.path.join(args.zip_dir,
                                    f"Mickey_merged_{zz:04d}.h5"), "r") as f:
            Z = np.asarray(f["image"], np.float32)
        best = None
        grid = np.zeros((len(scales), len(thetas)))
        for i, s in enumerate(scales):
            Zs = crop(downsample(Z, s), args.half)
            Zw = downsample(Zs, Zs.shape[0] / args.work)[:, ::-1]   # mirror=True
            for j, th in enumerate(thetas):
                B = prep(rotate(Zw, th, order=1, reshape=False), args.hp)
                v, dy, dx = ncc_peak(A, B)
                grid[i, j] = v
                if best is None or v > best[0]:
                    best = (v, float(s), float(th), dy * fo, dx * fo)
            print(f"  [{tag}] s {s:.3f}: best ncc over theta "
                  f"{grid[i].max():.4f} at {thetas[int(np.argmax(grid[i]))]:.1f}deg",
                  flush=True)
        v, s, th, dy, dx = best
        med = float(np.median(grid))
        print(f"  [{tag}] zip z{zz}: BEST ncc {v:.4f} (median {med:.4f}, "
              f"{v / max(med, 1e-9):.1f}x) at s {s:.4f} theta {th:+.2f} "
              f"shift {dy:+.1f},{dx:+.1f}", flush=True)
        results[tag] = dict(zip_z=zz, ncc=v, scale=s, theta=th, dy=dy, dx=dx,
                            median=med, grid=grid.tolist())

    rs, rf = results["same"], results["flip"]
    print(f"\n=== Z DIRECTION: same {rs['ncc']:.4f} vs flipped {rf['ncc']:.4f} "
          f"-> {'FLIPPED' if rf['ncc'] > rs['ncc'] else 'SAME'} "
          f"({max(rs['ncc'], rf['ncc']) / max(min(rs['ncc'], rf['ncc']), 1e-9):.2f}x)")
    win = "flip" if rf["ncc"] > rs["ncc"] else "same"
    w = results[win]
    print(f"=== SOLUTION (mirror=True): scale {w['scale']:.4f} theta "
          f"{w['theta']:+.2f} deg shift {w['dy']:+.1f},{w['dx']:+.1f} old px, "
          f"ncc {w['ncc']:.4f} = {w['ncc'] / max(w['median'], 1e-9):.1f}x median")
    print("  A sharp, high peak means the content matched and the transform is "
          "real; a broad low one means it is still not trustworthy.")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for a_, tag in zip(ax, ("same", "flip")):
        g = np.array(results[tag]["grid"])
        im = a_.imshow(g, aspect="auto", origin="lower",
                       extent=[thetas[0], thetas[-1], scales[0], scales[-1]])
        a_.set_title(f"{tag}: peak {results[tag]['ncc']:.4f}")
        a_.set_xlabel("theta (deg)"); a_.set_ylabel("scale")
        plt.colorbar(im, ax=a_)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "refine_grid.png"), dpi=130)
    with open(os.path.join(args.out_dir, "refine.json"), "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "grid"}
                   for k, v in results.items()}, f, indent=1)
    print(f"Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
