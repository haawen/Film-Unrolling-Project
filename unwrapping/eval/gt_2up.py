"""GT | ours 2-up video, with GT put through EXACTLY the same preparation the
stabilizer applied to our frames.

`stabilize_affine.py` does: load our frames (H,W) -> load GT, trim leader, crop
to the picture area, resize to (H,W) -> pair frame j with GT[idx[j]] where
idx = linspace(0, len(gt)-1, n) -> warp -> crop `extra_crop` off every side ->
resize to (Wf, H). To sit GT beside that output, GT must get the SAME pairing,
the SAME extra-crop and the SAME final resize, otherwise the two panels differ
in framing and scale and the comparison is misleading.

Usage:
  python -m unwrapping.eval.gt_2up \
      --pred unwrapping/eval/results/frame_match_tw3/film_affine.mp4 \
      --pred-source unwrapping/eval/results/frame_match_tw3/film_stabilized_gtsync.mp4 \
      --gt data/h265_1080p.mp4 --out <out.mp4> [--extra-crop 0.04] [--labels]
"""

import argparse
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .stabilize_horizontal import load_luma, _resize
from .compare_videos import load_video_luma, trim_leader, parse_crop, apply_crop


def _font(size):
    try:
        from matplotlib import font_manager
        from PIL import ImageFont
        return ImageFont.truetype(font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans")), size)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, nargs="+",
                    help="one or more stabilized videos to show right of GT")
    ap.add_argument("--pred-source", default=None,
                    help="the video that was fed to the stabilizer; its frame "
                         "count/size define the GT pairing. Default = --pred.")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-crop", default="0.235,0.12,0.81,0.88")
    ap.add_argument("--extra-crop", type=float, default=0.04,
                    help="Must match the stabilizer's --extra-crop.")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--gap", type=int, default=8)
    ap.add_argument("--labels", action="store_true")
    ap.add_argument("--pred-label", nargs="+", default=None,
                    help="one label per --pred (default: the file stem).")
    args = ap.parse_args()

    preds = [load_luma(q) for q in args.pred]
    labels = args.pred_label or [os.path.splitext(os.path.basename(q))[0]
                                 for q in args.pred]
    if len(labels) != len(preds):
        raise SystemExit("--pred-label count must match --pred count")
    src = load_luma(args.pred_source) if args.pred_source else preds[0]
    n = len(src)
    H, W = src[0].shape
    for q, pr in zip(args.pred, preds):
        print(f"pred {os.path.basename(q)}: {len(pr)} frames {pr[0].shape}")
    print(f"pairing basis {n} @ {H}x{W}")

    gt = load_video_luma(args.gt)
    gt, ntrim = trim_leader(gt)
    gt = apply_crop(gt, parse_crop(args.gt_crop))
    gt = [_resize(g, H, W) for g in gt]
    idx = np.linspace(0, len(gt) - 1, n).round().astype(int)
    print(f"GT {len(gt)} frames (leader {ntrim} dropped)")

    # same extra-crop + final resize the stabilizer applied to our frames
    ec = args.extra_crop
    y0, y1 = int(H * ec), int(H * (1 - ec))
    x0, x1 = int(W * ec), int(W * (1 - ec))
    ch, cw = y1 - y0, x1 - x0
    Wf = int(round(H * cw / ch)); Wf += Wf % 2

    m = min(min(len(p) for p in preds), n)
    f = _font(20) if args.labels else None
    lab_h = 30 if f is not None else 0
    wr = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                            quality=9, macro_block_size=1)
    for j in range(m):
        g = gt[idx[j]][y0:y1, x0:x1]
        gimg = Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8)) \
                    .resize((Wf, H))
        tiles = [gimg]
        for pr in preds:
            pi = Image.fromarray((np.clip(pr[j], 0, 1) * 255).astype(np.uint8))
            if pi.size != (Wf, H):
                pi = pi.resize((Wf, H))
            tiles.append(pi)
        ncol = len(tiles)
        canvas = Image.new("RGB",
                           (ncol * Wf + (ncol - 1) * args.gap, H + lab_h), "white")
        for c, t in enumerate(tiles):
            canvas.paste(t.convert("RGB"), (c * (Wf + args.gap), lab_h))
        if f is not None:
            dr = ImageDraw.Draw(canvas)
            for c, lab in enumerate(["GT"] + labels):
                dr.text((c * (Wf + args.gap) + 6, 5), lab, fill=(0, 0, 0), font=f)
        wr.append_data(np.asarray(canvas))
        if j % 60 == 0:
            print(f"  {j}/{m}", flush=True)
    wr.close()
    ncol = len(preds) + 1
    print(f"Done -> {args.out}  ({ncol * Wf + (ncol - 1) * args.gap}x{H + lab_h}, "
          f"{m} frames @{args.fps}fps)")


if __name__ == "__main__":
    main()
