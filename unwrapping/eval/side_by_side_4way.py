"""4-way GT-stabilized comparison: OLD (v12) vs 100% vs 75% full-density walk
vs GT optical scan, side by side.

Inputs are the per-strip `film_affine.mp4` (GT-STABILIZED VIEWING COPY,
`stabilize_affine.py --mode affine`, itself run on `frame_match.py`'s
`film_stabilized.mp4`) so all three predictions are already GT-oriented,
GT-scaled and per-frame content-matched to the same GT reel. This script only
needs to add the GT panel (same crop convention as frame_match/stabilize_affine
default: `--gt-crop 0.235,0.12,0.81,0.88`, leader-trimmed) and stack all four
with proportional resampling to a common frame count (panel lengths can differ
by a few frames from GT-skip insertions in film_stabilized.mp4).

Usage:
  python -m unwrapping.eval.side_by_side_4way \
      --old unwrapping/eval/results/frame_match_v12/film_affine.mp4 \
      --p100 unwrapping/eval/results/frame_match_full100/film_affine.mp4 \
      --p75 unwrapping/eval/results/frame_match_full75/film_affine.mp4 \
      --gt data/h265_1080p.mp4 \
      --out unwrapping/eval/results/compare_4way_old_100_75_gt.mp4
"""

import argparse

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .compare_videos import load_video_luma, trim_leader, parse_crop, apply_crop


def normalize(frames):
    stack = np.stack([f[::4, ::4] for f in frames])
    lo, hi = np.percentile(stack, [1, 99])
    return [np.clip((f - lo) / (hi - lo + 1e-6), 0, 1) for f in frames]


def resample(frames, n):
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return [frames[i] for i in idx]


def to_panel(f, H, label, font):
    W = max(2, int(round(H * f.shape[1] / f.shape[0])))
    im = Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8)).resize((W, H))
    im = im.convert("L")
    bar = 34
    canvas = Image.new("L", (W, H + bar), 0)
    canvas.paste(im, (0, bar))
    d = ImageDraw.Draw(canvas)
    tw = d.textlength(label, font=font)
    d.text(((W - tw) // 2, 6), label, fill=255, font=font)
    return canvas


def load_pred(path):
    r = imageio.get_reader(path)
    fr = []
    for f in r:
        f = np.asarray(f, dtype=np.float32)
        if f.ndim == 3:
            f = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
        fr.append(f / 255.0 if f.max() > 1.5 else f)
    r.close()
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="v12 film_affine.mp4")
    ap.add_argument("--p100", required=True, help="100%% film_affine.mp4")
    ap.add_argument("--p75", required=True, help="75%% film_affine.mp4")
    ap.add_argument("--gt", default="data/h265_1080p.mp4")
    ap.add_argument("--gt-crop", type=parse_crop, default=(0.235, 0.12, 0.81, 0.88))
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=420)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--gap", type=int, default=10)
    args = ap.parse_args()

    specs = [
        ("Old (v12)", args.old),
        ("100% (full density)", args.p100),
        ("75% (full density)", args.p75),
    ]

    seqs = []
    for label, path in specs:
        frames = normalize(load_pred(path))
        print(f"{label:22s} {len(frames)} frames  panel {frames[0].shape}")
        seqs.append((label, frames))

    gt = load_video_luma(args.gt)
    gt, n_drop = trim_leader(gt)
    gt = apply_crop(gt, args.gt_crop)
    gt = normalize([g for g in gt])
    print(f"{'GT scan':22s} {len(gt)} frames  panel {gt[0].shape} "
          f"({n_drop}-frame leader dropped)")
    seqs.append(("GT scan", gt))

    n = max(len(f) for _, f in seqs)
    print(f"common timeline: {n} frames @ {args.fps}fps -> {n / args.fps:.1f}s")

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    seqs = [(lab, resample(fr, n)) for lab, fr in seqs]
    gap = np.zeros((args.height + 34, args.gap), np.uint8)

    wr = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                            quality=8, macro_block_size=1)
    for k in range(n):
        panels = [to_panel(fr[k], args.height, lab, font) for lab, fr in seqs]
        parts = []
        for i, p in enumerate(panels):
            parts.append(np.asarray(p))
            if i < len(panels) - 1:
                parts.append(gap)
        row = np.concatenate(parts, axis=1)
        w = row.shape[1] + row.shape[1] % 2
        if w != row.shape[1]:
            row = np.pad(row, ((0, 0), (0, 1)))
        wr.append_data(row)
        if k % 40 == 0:
            print(f"  frame {k}/{n}", flush=True)
    wr.close()
    print(f"Done -> {args.out}")


if __name__ == "__main__":
    main()
