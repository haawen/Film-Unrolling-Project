"""Merge the two unrolled films + the GT scan into one side-by-side video.

For a FAIR visual comparison the three must be shown consistently:
  - orientation: the walk unrolls are rotated 90 + mirrored vs GT -> undone so
    all three are upright (verified via the readable "The End" title card).
  - crop: GT's perforations/edge print cropped to the picture area; the walks'
    top shading band trimmed.
  - timeline: GT has a dark leader + tail the walks lack; trimmed by per-frame
    activity, then all three proportionally resampled to a common frame count so
    the same film moment lines up across panels.
  - brightness: per-video 1-99 pct normalization.
Panels are stacked left-to-right with labels.

Usage:
  python -m unwrapping.eval.side_by_side --out data/three_way_compare.mp4
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
        fr.append(f)
    r.close()
    return fr


def normalize(frames):
    stack = np.stack([f[::4, ::4] for f in frames])
    lo, hi = np.percentile(stack, [1, 99])
    return [np.clip((f - lo) / (hi - lo + 1e-6), 0, 1) for f in frames]


def crop(f, box):
    if box is None:
        return f
    h, w = f.shape
    x0, y0, x1, y1 = box
    return f[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def orient(f, rot90=0, fliplr=False):
    if rot90:
        f = np.rot90(f, rot90)
    if fliplr:
        f = f[:, ::-1]
    return np.ascontiguousarray(f)


def trim_dark(frames, motion_thresh=0.03, bright_max=0.78):
    """Trim the static leader + tail/end-flash to the film's content span.

    Uses frame-to-frame motion (the leader is static -> diff 0; the dark title
    card still differs frame-to-frame) with a brightness guard to drop the bright
    projector end-flash. Keeps the first..last 'active' frame.
    """
    diff = np.array([0.0] + [np.abs(frames[i] - frames[i - 1]).mean()
                             for i in range(1, len(frames))])
    mean = np.array([f.mean() for f in frames])
    active = (diff > motion_thresh) & (mean < bright_max)
    idx = np.where(active)[0]
    if len(idx) == 0:
        return frames
    return frames[idx[0]:idx[-1] + 1]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/three_way_compare.mp4")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--gap", type=int, default=12)
    args = ap.parse_args()

    # (path, label, rot90, fliplr, crop-box, trim-dark)
    specs = [
        ("unwrapping/inr/results/walk_wholeroll_v6/film_v6.mp4",
         "Walk v6", 1, True, (0.0, 0.10, 1.0, 0.98), False),
        ("unwrapping/inr/results/walk_dense_f100/film_dense_f100.mp4",
         "Dense f100", 1, True, (0.0, 0.10, 1.0, 0.98), False),
        ("data/h265_1080p.mp4",
         "GT scan", 0, False, (0.13, 0.07, 0.80, 0.93), True),
    ]

    seqs = []
    for path, label, r90, flip, box, trim in specs:
        frames = normalize(load_luma(path))
        frames = [orient(crop(f, box), r90, flip) for f in frames]
        if trim:
            frames = trim_dark(frames)
        print(f"{label:16s} {len(frames)} frames  panel {frames[0].shape}")
        seqs.append((label, frames))

    n = max(len(f) for _, f in seqs)  # common timeline
    print(f"common timeline: {n} frames @ {args.fps}fps -> {n/args.fps:.1f}s")

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
