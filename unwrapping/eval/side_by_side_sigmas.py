"""Side-by-side raw-video comparison of z-smooth-anchors sigma variants (no GT,
no stabilization -- just the direct render outputs stacked with labels, for
visually judging whether a render lever changes anything).

Usage:
  python -m unwrapping.eval.side_by_side_sigmas \
      --videos 0=path0.mp4 1.5=path1.mp4 3.0=path2.mp4 6.0=path3.mp4 \
      --out out.mp4
"""

import argparse

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_luma(path):
    r = imageio.get_reader(path)
    fr = []
    for f in r:
        f = np.asarray(f, dtype=np.float32)
        if f.ndim == 3:
            f = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
        fr.append(f / 255.0 if f.max() > 1.5 else f)
    r.close()
    return fr


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
    bar = 32
    canvas = Image.new("L", (W, H + bar), 0)
    canvas.paste(im, (0, bar))
    d = ImageDraw.Draw(canvas)
    tw = d.textlength(label, font=font)
    d.text(((W - tw) // 2, 6), label, fill=255, font=font)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True,
                    help="label=path.mp4 pairs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=420)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--gap", type=int, default=8)
    args = ap.parse_args()

    specs = []
    for kv in args.videos:
        label, path = kv.split("=", 1)
        specs.append((label, path))

    seqs = []
    for label, path in specs:
        frames = normalize(load_luma(path))
        print(f"{label:10s} {len(frames)} frames  panel {frames[0].shape}")
        seqs.append((label, frames))

    n = max(len(f) for _, f in seqs)
    print(f"common timeline: {n} frames @ {args.fps}fps -> {n / args.fps:.1f}s")

    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    seqs = [(lab, resample(fr, n)) for lab, fr in seqs]
    gap = np.zeros((args.height + 32, args.gap), np.uint8)

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
