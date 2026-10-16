"""Verify a tightened GT picture-area crop (drop sprockets + film border + bleed).

Extracts several GT frames (after the leader), draws the CURRENT vs a PROPOSED
crop box on each, and shows the proposed-cropped result — so the box can be
checked against real frames (incl. frame-to-frame framing wobble) before locking
it into the eval. Also overlays the proposed box on our phase-locked pred frame.

Usage:
  python -m unwrapping.eval.verify_gt_crop --gt data/h265_1080p.mp4 \
      --pred <film_global.mp4> --out <montage.png> \
      --gt-box 0.16,0.09,0.84,0.91 --cur-box 0.13,0.07,0.80,0.93 \
      --pred-box 0.0,0.12,1.0,0.98 --pred-rot90 1 --pred-fliplr
"""
import argparse
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw


def read_frames(path, idxs):
    rd = imageio.get_reader(path)
    out = {}
    for i, fr in enumerate(rd):
        if i in idxs:
            g = np.asarray(fr).astype(np.float32)
            if g.ndim == 3:
                g = g[..., :3].mean(2)
            out[i] = g
        if i > max(idxs):
            break
    rd.close()
    return [out[i] for i in idxs if i in out]


def to_img(a):
    lo, hi = np.percentile(a, [1, 99])
    return Image.fromarray((np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255)
                           .astype(np.uint8)).convert("RGB")


def draw_box(im, box, color):
    w, h = im.size
    x0, y0, x1, y1 = box
    ImageDraw.Draw(im).rectangle([x0 * w, y0 * h, x1 * w, y1 * h],
                                 outline=color, width=3)
    return im


def crop(a, box):
    h, w = a.shape
    x0, y0, x1, y1 = box
    return a[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True); ap.add_argument("--pred", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-box", required=True)
    ap.add_argument("--cur-box", default=None)
    ap.add_argument("--pred-box", default=None)
    ap.add_argument("--pred-rot90", type=int, default=0)
    ap.add_argument("--pred-fliplr", action="store_true")
    ap.add_argument("--leader", type=int, default=6)
    a = ap.parse_args()
    box = tuple(float(x) for x in a.gt_box.split(","))
    cur = tuple(float(x) for x in a.cur_box.split(",")) if a.cur_box else None

    idxs = [a.leader + k for k in (0, 8, 16, 40, 120, 200)]
    gts = read_frames(a.gt, idxs)
    rows = []
    for g in gts:
        im = to_img(g)
        if cur:
            draw_box(im, cur, (60, 120, 255))       # current = blue
        draw_box(im, box, (255, 40, 40))            # proposed = red
        cropped = to_img(crop(g, box)).resize(im.size)
        rows.append(np.concatenate([np.asarray(im), np.asarray(cropped)], axis=1))
    H = min(r.shape[0] for r in rows)
    mont = np.concatenate([r[:H] for r in rows], axis=0)
    Image.fromarray(mont).save(a.out)
    print(f"GT: box(red)={box} cur(blue)={cur}; wrote {a.out}")

    if a.pred:
        pf = read_frames(a.pred, [2, 10, 30])
        pbox = tuple(float(x) for x in a.pred_box.split(",")) if a.pred_box else None
        tiles = []
        for p in pf:
            if a.pred_rot90:
                p = np.rot90(p, k=a.pred_rot90)
            if a.pred_fliplr:
                p = p[:, ::-1]
            im = to_img(p)
            if pbox:
                draw_box(im, pbox, (255, 40, 40))
            tiles.append(np.asarray(im.resize((300, 400))))
        Image.fromarray(np.concatenate(tiles, axis=1)).save(
            a.out.replace(".png", "_pred.png"))
        print(f"PRED: box={pbox}; wrote {a.out.replace('.png', '_pred.png')}")


if __name__ == "__main__":
    main()
