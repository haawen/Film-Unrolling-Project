"""Side-by-side sheets of GT frames vs our unrolled frames.

Reads the ALIGNED exports from unwrapping.eval.compare_videos (`aligned_gt.mp4`
/ `aligned_pred.mp4`), which are already frame-matched 1:1 and registered, so
pair i is simply frame i of each. GT goes on the LEFT.

Produces:
  gt_vs_<tag>_sheet.png     N pairs spread across the film, labelled
  gt_vs_<tag>_pair_<i>.png  a few full-resolution single pairs
  gt_tw1_tw3_sheet.png      optional 3-way, when two preds are given

Usage:
  python Scripts/make_gt_pair_sheets.py --dir <...>/v13_tw3_hires --tag tw3 \
      --out-dir <out> [--extra-dir <...>/v13_tw1_hires --extra-tag tw1]
"""

import argparse
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def read_frames(path):
    rd = imageio.get_reader(path)
    fr = [np.asarray(f) for f in rd]
    rd.close()
    return fr


def to_gray(a):
    return a if a.ndim == 2 else a[..., :3].mean(axis=2)


def font(size):
    try:
        from matplotlib import font_manager
        from PIL import ImageFont
        return ImageFont.truetype(
            font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans")),
            size)
    except Exception:
        return None


def stack_pair(imgs, labels, gap=8, lab_h=30, scale=1.0):
    """imgs left->right, white gap, caption band above."""
    hs = [im.shape[0] for im in imgs]
    H = min(hs)
    tiles = []
    for im in imgs:
        p = Image.fromarray(im.astype(np.uint8))
        if im.shape[0] != H:
            p = p.resize((int(round(im.shape[1] * H / im.shape[0])), H))
        if scale != 1.0:
            p = p.resize((int(p.width * scale), int(p.height * scale)))
        tiles.append(p)
    W = sum(t.width for t in tiles) + gap * (len(tiles) - 1)
    canvas = Image.new("RGB", (W, tiles[0].height + lab_h), "white")
    x = 0
    f = font(20)
    dr = ImageDraw.Draw(canvas)
    for t, lab in zip(tiles, labels):
        canvas.paste(t.convert("RGB"), (x, lab_h))
        dr.text((x + 6, 5), lab, fill=(0, 0, 0), font=f)
        x += t.width + gap
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir with aligned_gt/pred mp4")
    ap.add_argument("--tag", default="pred")
    ap.add_argument("--extra-dir", default=None, help="second pred for a 3-way")
    ap.add_argument("--extra-tag", default="pred2")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-sheet", type=int, default=10)
    ap.add_argument("--skip-first", type=int, default=6,
                    help="Skip the innermost cells (banding / the cut zone).")
    ap.add_argument("--singles", type=int, default=4)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    gt = read_frames(os.path.join(args.dir, "aligned_gt.mp4"))
    pr = read_frames(os.path.join(args.dir, "aligned_pred.mp4"))
    n = min(len(gt), len(pr))
    print(f"{n} aligned pairs, frame {gt[0].shape} / {pr[0].shape}")

    ex = None
    if args.extra_dir:
        ex = read_frames(os.path.join(args.extra_dir, "aligned_pred.mp4"))
        n = min(n, len(ex))
        print(f"  + {len(ex)} frames for {args.extra_tag}")

    idx = np.linspace(args.skip_first, n - 1, args.n_sheet).round().astype(int)

    # ── sheet: GT | tag, stacked vertically, 2 pairs per row ──
    rows, per_row = [], 2
    for k in range(0, len(idx), per_row):
        chunk = idx[k:k + per_row]
        cells = [stack_pair([gt[i], pr[i]], [f"GT  #{i}", f"{args.tag}  #{i}"],
                            scale=0.55) for i in chunk]
        w = sum(c.width for c in cells) + 24 * (len(cells) - 1)
        h = max(c.height for c in cells)
        row = Image.new("RGB", (w, h), "white")
        x = 0
        for c in cells:
            row.paste(c, (x, 0)); x += c.width + 24
        rows.append(row)
    W = max(r.width for r in rows); H = sum(r.height + 14 for r in rows)
    sheet = Image.new("RGB", (W, H), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y)); y += r.height + 14
    p = os.path.join(args.out_dir, f"gt_vs_{args.tag}_sheet.png")
    sheet.save(p); print("saved", p, sheet.size)

    # ── a few full-resolution single pairs ──
    for i in np.linspace(args.skip_first, n - 1, args.singles).round().astype(int):
        im = stack_pair([gt[i], pr[i]], [f"GT  frame {i}", f"{args.tag}  frame {i}"])
        p = os.path.join(args.out_dir, f"gt_vs_{args.tag}_pair_{i:03d}.png")
        im.save(p)
    print(f"saved {args.singles} full-res pairs")

    # ── optional 3-way ──
    if ex is not None:
        rows = []
        for i in np.linspace(args.skip_first, n - 1, 5).round().astype(int):
            rows.append(stack_pair(
                [gt[i], ex[i], pr[i]],
                [f"GT  #{i}", f"{args.extra_tag}  #{i}", f"{args.tag}  #{i}"],
                scale=0.62))
        W = max(r.width for r in rows); H = sum(r.height + 14 for r in rows)
        sheet = Image.new("RGB", (W, H), "white")
        y = 0
        for r in rows:
            sheet.paste(r, (0, y)); y += r.height + 14
        p = os.path.join(args.out_dir,
                         f"gt_{args.extra_tag}_{args.tag}_sheet.png")
        sheet.save(p); print("saved", p, sheet.size)


if __name__ == "__main__":
    main()
