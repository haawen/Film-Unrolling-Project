"""Compare transverse-averaging widths for the emulsion render.

Averaging n taps across the emulsion's 4px thickness should cut NOISE by ~sqrt(n)
while leaving real EDGES alone. If the emulsion is tilted w.r.t. the path normal,
or taps spill past a thin band into the film base, it instead SMEARS the picture
along the film. So the two must be measured separately -- a single "sharpness"
number cannot tell a denoised image from a blurred one, because both reduce it.

  noise  = std of the high-pass residual in FLAT regions only (lowest-gradient
           tiles), where there is no real structure to confuse it
  edge   = p99 of |d/d(arc)| , i.e. the strength of the strongest real edges,
           which blurring lowers and denoising does not
  ratio  = edge / noise  -- the number that actually matters

The strips are pixel-aligned across widths (only the sampling changed, not the
column mapping), so the same crop compares like for like.

Usage:
  python Scripts/diag_transverse_compare.py --strips tw1=<a.npy> tw3=<b.npy> ... \
      --out-dir <dir>
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter


def metrics(A):
    """noise (flat-region HF std), edge strength (p99 |grad| along arc)."""
    A = A.astype(np.float32)
    hp = A - gaussian_filter(A, 2.0)
    g = np.abs(np.diff(A, axis=1))
    # flat tiles = lowest local gradient; measure noise only there
    T = 32
    h, w = A.shape[0] // T * T, (A.shape[1] - 1) // T * T
    gt = g[:h, :w].reshape(h // T, T, w // T, T).mean(axis=(1, 3))
    ht = hp[:h, :w].reshape(h // T, T, w // T, T).std(axis=(1, 3))
    flat = gt <= np.percentile(gt, 20)
    return float(ht[flat].mean()), float(np.percentile(g, 99))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strips", nargs="+", required=True,
                    help="label=path.npy, e.g. tw1=.../wholeroll.npy")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--crop-frac", type=float, default=0.35,
                    help="Where along the strip to cut the visual crop.")
    ap.add_argument("--crop-cols", type=int, default=2400)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    items = []
    for spec in args.strips:
        lab, _, path = spec.partition("=")
        items.append((lab, path))

    print(f"{'width':>8} {'noise(flat)':>12} {'edge p99':>10} {'edge/noise':>11}  "
          f"{'vs first':>9}")
    rows, crops = [], []
    base = None
    for lab, path in items:
        S = np.load(path, mmap_mode="r")
        # sample several windows spread along the roll for the numbers
        ns, es = [], []
        for f in (0.15, 0.35, 0.55, 0.75, 0.9):
            c0 = int(f * (S.shape[1] - 4000))
            A = np.asarray(S[:, c0:c0 + 4000], np.float32)
            n_, e_ = metrics(A)
            ns.append(n_); es.append(e_)
        noise, edge = float(np.mean(ns)), float(np.mean(es))
        snr = edge / max(noise, 1e-9)
        if base is None:
            base = snr
        print(f"{lab:>8} {noise:12.5f} {edge:10.5f} {snr:11.2f}  "
              f"{100*(snr/base-1):+8.1f}%")
        rows.append((lab, noise, edge, snr))
        c0 = int(args.crop_frac * (S.shape[1] - args.crop_cols))
        crops.append((lab, np.asarray(S[:, c0:c0 + args.crop_cols], np.float32)))
        del S

    # visual: same crop at each width, stacked, plus a zoom on one cell
    lo = np.percentile(crops[0][1], 1); hi = np.percentile(crops[0][1], 99)
    from PIL import Image, ImageDraw
    tiles = []
    for lab, A in crops:
        v = 1.0 - np.clip((A - lo) / (hi - lo + 1e-6), 0, 1)       # film negative
        tiles.append((lab, (v * 255).astype(np.uint8)))
    H, W = tiles[0][1].shape
    ds = max(1, W // 2000)
    stack = np.concatenate([np.pad(t[::1, ::ds], ((0, 6), (0, 0)),
                                   constant_values=255) for _, t in tiles], axis=0)
    im = Image.fromarray(stack).convert("RGB")
    dr = ImageDraw.Draw(im)
    for i, (lab, _) in enumerate(tiles):
        dr.text((6, i * (H + 6) + 6), lab, fill=(255, 0, 0))
    im.save(os.path.join(args.out_dir, "transverse_crops.png"))

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    labs = [r[0] for r in rows]
    for k, (t, idx) in enumerate([("noise in flat areas (lower=better)", 1),
                                  ("edge strength p99 (higher=sharper)", 2),
                                  ("edge / noise (higher=better)", 3)]):
        ax[k].bar(labs, [r[idx] for r in rows], color="C0")
        ax[k].set_title(t, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "transverse_metrics.png"), dpi=120)
    print(f"\nsaved transverse_crops.png + transverse_metrics.png -> {args.out_dir}")


if __name__ == "__main__":
    main()
