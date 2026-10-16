"""Compare v12 (interpolated 500-anchor) vs full-density (1184-slice) strips at the
SAME film content. Crops-at-arc-fraction are different film positions because the
strips differ in length; this matches content via normalized cross-correlation on a
downsampled band, then shows the SAME film window from both so the dark-sweep
artifact (if unique to full-density) is unambiguous vs shared content.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def band(strip, z0=450, z1=700, ds=8):
    b = np.asarray(strip[z0:z1, :], np.float32)
    b = (b - b.mean()) / (b.std() + 1e-6)
    return b[:, ::ds].mean(0)          # 1-D downsampled arc profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v12", default="unwrapping/inr/results/walk_dense_v12/wholeroll.npy")
    ap.add_argument("--full", default="unwrapping/inr/results/walk_dense_full/wholeroll.npy")
    ap.add_argument("--out", default="unwrapping/inr/results/strip_match.png")
    ap.add_argument("--win", type=int, default=6000, help="arc window (px) to show")
    ap.add_argument("--anchor-frac", type=float, default=0.5)
    args = ap.parse_args()
    ds = 8
    sa = np.load(args.v12, mmap_mode="r")
    sb = np.load(args.full, mmap_mode="r")
    print(f"v12 {sa.shape}, full {sb.shape}", flush=True)
    pa, pb = band(sa, ds=ds), band(sb, ds=ds)
    # pick a window in v12, find best-matching offset in full via xcorr
    ca = int(args.anchor_frac * len(pa))
    w = args.win // ds
    ta = pa[ca:ca + w]
    ta = (ta - ta.mean()) / (ta.std() + 1e-6)
    best, bestlag = -1e9, 0
    for lag in range(0, len(pb) - w):
        seg = pb[lag:lag + w]
        seg = (seg - seg.mean()) / (seg.std() + 1e-6)
        s = float(np.dot(ta, seg)) / w
        if s > best:
            best, bestlag = s, lag
    print(f"best xcorr {best:.3f} at full-arc {bestlag*ds} (v12 arc {ca*ds})", flush=True)
    # extract the SAME film window (full res) from both
    a0 = ca * ds; b0 = bestlag * ds
    Wpx = args.win
    ba = np.asarray(sa[:, a0:a0 + Wpx], np.float32)
    bb = np.asarray(sb[:, b0:b0 + Wpx], np.float32)
    fig, axes = plt.subplots(2, 1, figsize=(18, 9))
    for ax, img, tag in [(axes[0], ba, "v12 (500-anchor interpolated)"),
                         (axes[1], bb, "full-density (1184 measured)")]:
        lo, hi = np.percentile(img, [1, 99])
        ax.imshow(np.clip((img - lo) / (hi - lo + 1e-6), 0, 1), cmap="gray",
                  aspect="auto")
        ax.set_title(f"{tag}  (xcorr {best:.2f})", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
