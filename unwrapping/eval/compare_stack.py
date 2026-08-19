"""Stack any number of film videos side by side, labelled, for visual review.

Generic replacement for the one-off side_by_side_*.py scripts, which each
hardcode one comparison.  Takes the same `--margins` / `--size` conventions as
border_stillness.py so that what you look at is what was measured: every video
is cropped to the picture and resampled to one common size before stacking.
Videos of different lengths are resampled proportionally onto the longest, so
frame i of every panel is the same point in the reel.

    python -m unwrapping.eval.compare_stack --videos a.mp4 b.mp4 c.mp4 \
        --labels GT ours ours+affine --margins 0 0 0 --out cmp.mp4
"""

import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--margins", nargs="*", type=float, default=None,
                    help="Per-video render margin to crop off, so every panel "
                         "shows the picture and nothing else.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=460)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--gap", type=int, default=8)
    args = ap.parse_args()
    if len(args.videos) != len(args.labels):
        raise SystemExit("--videos and --labels must match in length")
    margins = args.margins or [0.0] * len(args.videos)

    import cv2
    import imageio.v2 as imageio
    from .border_stillness import load

    vids = [load(p, m) for p, m in zip(args.videos, margins)]
    n = max(len(v) for v in vids)
    H = args.height
    Ws = [int(round(H * v.shape[2] / v.shape[1])) for v in vids]
    Ws = [w + w % 2 for w in Ws]
    print(f"{len(vids)} panels, {n} frames, panel height {H}", flush=True)
    for p, v, w in zip(args.videos, vids, Ws):
        print(f"  {len(v):4d} frames {v.shape[2]}x{v.shape[1]} -> {w}x{H}  {p}",
              flush=True)

    total = sum(Ws) + args.gap * (len(vids) - 1)
    total += total % 2
    wr = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8,
                            macro_block_size=1)
    for i in range(n):
        panels = []
        for v, w, lbl in zip(vids, Ws, args.labels):
            j = int(round(i * (len(v) - 1) / max(1, n - 1)))
            f = cv2.resize(v[j], (w, H))
            lo, hi = np.percentile(f, [1, 99])
            f = np.clip((f - lo) / (hi - lo + 1e-9), 0, 1)
            f = (f * 255).astype(np.uint8)
            cv2.putText(f, lbl, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 5)
            cv2.putText(f, lbl, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
            panels.append(f)
        row = panels[0]
        for p in panels[1:]:
            row = np.concatenate(
                [row, np.zeros((H, args.gap), np.uint8), p], axis=1)
        if row.shape[1] < total:
            row = np.pad(row, ((0, 0), (0, total - row.shape[1])))
        wr.append_data(row)
    wr.close()
    print(f"wrote {args.out}  ({H}x{total}, {n} frames)", flush=True)


if __name__ == "__main__":
    main()
