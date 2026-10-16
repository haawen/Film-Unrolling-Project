"""Dump a full-resolution crop of a rendered strip as PNG, with cell/z rulers.

The rendered strip is (z, arc): row = film width (CT slice index), column = film
length. A video frame is one CELL of `--pitch` columns, and `make_film_video`'s
`--reverse` means video frame f is cell (n_cells-1-f). This script takes the
video-frame numbers you actually saw the artifact at and cuts the corresponding
columns, so a defect can be read back in the coordinates the walk works in
(which z, which winding, which azimuth) instead of in seconds.

Rulers are drawn ON the crop: red ticks + labels every cell along the top, and
z labels down the left, because a crop of 20000 columns is otherwise impossible
to index by eye.

Usage:
  python Scripts/crop_strip_png.py --strip .../wholeroll.npy --out-dir <dir> \
      --frames 165 195 --n-frames 229 --pitch 1130.14 [--z 430 660] [--xscale 2]
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--frames", nargs=2, type=int, required=True,
                    help="Inclusive video-frame range to cut.")
    ap.add_argument("--n-frames", type=int, required=True,
                    help="Total cells in the video (for the --reverse flip).")
    ap.add_argument("--pitch", type=float, required=True)
    ap.add_argument("--no-reverse", action="store_true",
                    help="Set if the video was NOT made with --reverse.")
    ap.add_argument("--z", nargs=2, type=int, default=None,
                    help="Strip ROW range to keep (row = z - z_first).")
    ap.add_argument("--z-first", type=int, default=40,
                    help="CT z of strip row 0, for the ruler labels only.")
    ap.add_argument("--xscale", type=int, default=1,
                    help="Downsample factor along the arc axis.")
    ap.add_argument("--tag", default="crop")
    args = ap.parse_args()

    A = np.load(args.strip, mmap_mode="r")
    Z, C = A.shape
    n = args.n_frames
    f0, f1 = args.frames
    # video frame -> cell index, then cell -> columns
    cells = [f if args.no_reverse else (n - 1 - f) for f in (f1, f0)]
    c0 = max(0, int(min(cells) * args.pitch))
    c1 = min(C, int((max(cells) + 1) * args.pitch))
    r0, r1 = (0, Z) if args.z is None else (max(0, args.z[0]), min(Z, args.z[1]))
    print(f"strip {A.shape}; frames {f0}-{f1} -> cells {min(cells)}-{max(cells)} "
          f"-> cols {c0}-{c1} ({c1-c0}), rows {r0}-{r1}")

    sub = np.asarray(A[r0:r1, c0:c1], dtype=np.float32)
    lo, hi = np.percentile(sub, [0.5, 99.5])
    img = np.clip((sub - lo) / max(hi - lo, 1e-6), 0, 1)
    img = (img * 255).astype(np.uint8)
    if args.xscale > 1:
        img = img[:, ::args.xscale]

    import cv2
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    for c in range(int(min(cells)), int(max(cells)) + 2):
        x = int((c * args.pitch - c0) / args.xscale)
        if not 0 <= x < w:
            continue
        cv2.line(img, (x, 0), (x, 28), (0, 0, 255), 1)
        vf = c if args.no_reverse else (n - 1 - c)
        cv2.putText(img, f"cell{c}/f{vf}", (x + 3, 20), 0, 0.5, (0, 0, 255), 1)
    for r in range(r0 - r0 % 100, r1, 100):
        y = r - r0
        if not 0 <= y < h:
            continue
        cv2.line(img, (0, y), (30, y), (0, 255, 255), 1)
        cv2.putText(img, f"z{r + args.z_first}", (34, y + 5), 0, 0.5,
                    (0, 255, 255), 1)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"{args.tag}_f{f0}-{f1}.png")
    cv2.imwrite(out, img)
    print(f"wrote {out}  {img.shape}")


if __name__ == "__main__":
    main()
