"""
Render high-quality comparison images of predicted vs GT unwrapped strip.

Purpose: video presets produce strips at the synthetic generator's full
resolution (n_z, arc_length ≈ 20 × 143000). For real visual comparison we
want pixel-for-pixel side-by-side at native resolution, not the auto-
downscaled `gt_compare.png` produced by `surface_eval.py`.

Outputs (per --out-dir):
    full_compare.png       — full strip, predicted above GT, split into N
                              tiled horizontal panels for readability
    crop_<frame_idx>.png   — zoom-in on a single source frame's worth of
                              strip width (≈ frame_w pixels), pred vs GT
    diff.png               — absolute difference map

Usage:
    python -m unwrapping.inr.strip_compare \
        --gt-npz   <preset>/ground_truth.npz \
        --pred-npz <eval_dir>/strip.npz \
        --out-dir  <out>
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_strip_npz(path, key="strip"):
    d = np.load(path)
    if key not in d.files:
        # Some npzs use other keys; try common alternatives.
        for alt in ("strip", "pred_strip", "rendered"):
            if alt in d.files:
                return d[alt].astype(np.float32)
        raise KeyError(f"{path}: no usable strip key (have {d.files})")
    return d[key].astype(np.float32)


def build_full_compare(gt, pred, out_path, n_panels=5, dpi=120):
    """Tile the full strip width into n_panels horizontal stripes,
    each panel showing pred above GT pixel-for-pixel."""
    Zg, Lg = gt.shape
    Zp, Lp = pred.shape
    L = min(Lg, Lp)
    gt = gt[:, :L]
    pred = pred[:, :L]
    panel_w = L // n_panels
    fig_h = (Zg + Zp + 2) * n_panels * 0.04 + 1
    fig, axes = plt.subplots(n_panels, 1, figsize=(20, fig_h),
                              squeeze=False)
    for i in range(n_panels):
        a, b = i * panel_w, (i + 1) * panel_w if i < n_panels - 1 else L
        # Vertically stack predicted (top) and GT (bottom) with a 2-pixel gap.
        gap = np.full((2, b - a), np.nan, dtype=np.float32)
        combined = np.concatenate([pred[:, a:b], gap, gt[:, a:b]], axis=0)
        axes[i, 0].imshow(combined, cmap="gray", aspect="auto",
                          interpolation="nearest", vmin=0, vmax=1)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([Zp // 2, Zp + 2 + Zg // 2])
        axes[i, 0].set_yticklabels(["pred", "GT"])
        axes[i, 0].set_title(f"u ∈ [{a}, {b}]  (panel {i+1}/{n_panels})",
                              fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def build_crops(gt, pred, out_dir, n_crops=4, crop_w=2500, dpi=160):
    """Save n_crops zoom-in images, evenly spaced along u, at native resolution."""
    Zg, Lg = gt.shape
    Zp, Lp = pred.shape
    L = min(Lg, Lp)
    crop_starts = np.linspace(0, L - crop_w, n_crops, dtype=int)
    paths = []
    for k, s in enumerate(crop_starts):
        e = s + crop_w
        fig, ax = plt.subplots(2, 1, figsize=(crop_w / 200, (Zg + Zp) / 8),
                                gridspec_kw={"hspace": 0.05})
        ax[0].imshow(pred[:, s:e], cmap="gray", aspect="auto",
                      interpolation="nearest", vmin=0, vmax=1)
        ax[0].set_title(f"PRED   u ∈ [{s}, {e}]", fontsize=10, loc="left")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].imshow(gt[:, s:e], cmap="gray", aspect="auto",
                      interpolation="nearest", vmin=0, vmax=1)
        ax[1].set_title("GT", fontsize=10, loc="left")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.tight_layout()
        path = os.path.join(out_dir, f"crop_{k:02d}_u{s}_{e}.png")
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()
        paths.append(path)
    return paths


def build_diff(gt, pred, out_path, dpi=120, n_panels=3):
    """Absolute-difference map, tiled across n_panels."""
    Zg, Lg = gt.shape
    Zp, Lp = pred.shape
    Z = min(Zg, Zp)
    L = min(Lg, Lp)
    diff = np.abs(pred[:Z, :L] - gt[:Z, :L])
    panel_w = L // n_panels
    fig, axes = plt.subplots(n_panels, 1, figsize=(20, n_panels * 0.6),
                              squeeze=False)
    for i in range(n_panels):
        a, b = i * panel_w, (i + 1) * panel_w if i < n_panels - 1 else L
        im = axes[i, 0].imshow(diff[:, a:b], cmap="hot", aspect="auto",
                                interpolation="nearest", vmin=0, vmax=0.5)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_title(f"|pred-GT|  u ∈ [{a}, {b}]", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-npz", required=True,
                   help="ground_truth.npz with key 'strip' (the source strip "
                        "in [0, 1] frame-content space).")
    p.add_argument("--pred-npz", required=True,
                   help="eval strip.npz with key 'strip' (predicted strip in "
                        "CT-intensity space).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-panels", type=int, default=5)
    p.add_argument("--n-crops", type=int, default=4)
    p.add_argument("--crop-w", type=int, default=2500)
    p.add_argument("--ct-base", type=float, default=0.12,
                   help="CT intensity for film base (subtract before "
                        "normalizing predicted strip back to [0, 1]).")
    p.add_argument("--ct-max", type=float, default=0.85,
                   help="CT intensity for emulsion at full white.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading GT: {args.gt_npz}")
    gt = load_strip_npz(args.gt_npz)
    print(f"  GT shape: {gt.shape}, range [{gt.min():.3f}, {gt.max():.3f}]")
    print(f"Loading pred: {args.pred_npz}")
    pred_raw = load_strip_npz(args.pred_npz)
    print(f"  pred shape: {pred_raw.shape}, range "
          f"[{pred_raw.min():.3f}, {pred_raw.max():.3f}]")

    # Invert CT intensity model so predicted strip is comparable to GT
    # (which is in [0, 1] source-content space):
    #   intensity = base + content * (max - base)
    #   ⇒ content = (intensity - base) / (max - base)
    span = max(args.ct_max - args.ct_base, 1e-6)
    pred = (pred_raw - args.ct_base) / span
    pred = np.clip(pred, 0.0, 1.0)
    print(f"  pred (inverted to content): range "
          f"[{pred.min():.3f}, {pred.max():.3f}]")

    # Trim to common length
    L = min(gt.shape[1], pred.shape[1])
    if gt.shape[1] != pred.shape[1]:
        print(f"  WARNING: GT width {gt.shape[1]} != pred width {pred.shape[1]}, "
              f"trimming both to {L}.")
    gt = gt[:, :L]
    pred = pred[:, :L]

    # 1) Full strip tiled
    full_path = os.path.join(args.out_dir, "full_compare.png")
    print(f"Writing full compare: {full_path}")
    build_full_compare(gt, pred, full_path, n_panels=args.n_panels)

    # 2) Crops at native resolution
    print(f"Writing {args.n_crops} crop views (crop_w={args.crop_w})")
    build_crops(gt, pred, args.out_dir,
                n_crops=args.n_crops, crop_w=args.crop_w)

    # 3) Difference map
    diff_path = os.path.join(args.out_dir, "diff.png")
    print(f"Writing diff: {diff_path}")
    build_diff(gt, pred, diff_path, n_panels=args.n_panels)

    # 4) Quantitative summary
    diff = np.abs(pred - gt)
    print("")
    print(f"  |pred - GT|  mean={diff.mean():.4f}  median={np.median(diff):.4f}  "
          f"p95={np.percentile(diff, 95):.4f}  max={diff.max():.4f}")


if __name__ == "__main__":
    main()
