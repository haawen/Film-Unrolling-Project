"""Measure how still the picture border is in a rendered film video.

Method: cross-correlate each frame's border-region profile against the TEMPORAL
MEDIAN of that profile, and read the sub-pixel shift.  The whole border
structure is used, so the measurement cannot jump between features.

That matters, because the obvious alternative -- "find the strongest gradient
near where the border should be" -- does not work here and produced a fake
result.  Within 60 px of the picture edge there are several dark lines (the
border itself, the film margin, the perf-band edge), so a peak-picker hops
between them from frame to frame and reports ~20 px of motion on a border that
is actually locked.

    python -m unwrapping.eval.border_stillness --videos a.mp4 b.mp4 \
        --labels baseline borderlock --band 0.10
"""

import argparse
import json
import os

import numpy as np


def load(path, margin=0.0, size=None):
    """Frames as (n, H, W) float.

    `margin` crops a render's surround back off (as the fraction of the picture
    it was rendered with) and `size` resamples to a common (W, H).  Both exist
    because comparing videos at different crops or sizes silently measures
    different regions in different units -- the single most common way to get a
    confident, wrong stillness number here.  Pass them, or compare only videos
    that already share a framing.
    """
    import cv2
    cap = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if not fr:
        raise SystemExit(f"no frames in {path}")
    F = np.stack(fr)
    if margin > 0:
        m = margin / (1 + 2 * margin)
        h, w = F.shape[1:]
        F = F[:, int(m * h):int((1 - m) * h), int(m * w):int((1 - m) * w)]
    if size is not None:
        F = np.stack([cv2.resize(f, tuple(size)) for f in F])
    return F


def _shift(prof, ref, max_shift=40):
    """Sub-pixel shift of `prof` relative to `ref` by cross-correlation."""
    a = prof - prof.mean()
    b = ref - ref.mean()
    n = len(a)
    lags = np.arange(-max_shift, max_shift + 1)
    cc = np.empty(len(lags))
    for i, L in enumerate(lags):
        if L < 0:
            x, y = a[-L:], b[:n + L]
        elif L > 0:
            x, y = a[:n - L], b[L:]
        else:
            x, y = a, b
        cc[i] = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9)
    j = int(np.argmax(cc))
    if not (1 <= j < len(cc) - 1):
        return np.nan
    p, q, r = cc[j - 1], cc[j], cc[j + 1]
    d = p - 2 * q + r
    return float(lags[j] + (0.5 * (p - r) / d if abs(d) > 1e-9 else 0.0))


def border_shifts(F, band=0.10, max_shift=40):
    """Per-frame shift of each of the four border regions.

    Axis convention: the render rotates so the picture is upright, so the
    HORIZONTAL axis of the video is ACROSS the film and the VERTICAL is ALONG it.
    """
    n, H, W = F.shape
    bw, bh = int(band * W), int(band * H)
    cols = F.mean(axis=1)          # (n, W) -- profile across the film
    rows = F.mean(axis=2)          # (n, H) -- profile along the film
    regions = {
        "across_low": cols[:, :bw],
        "across_high": cols[:, W - bw:],
        "along_low": rows[:, :bh],
        "along_high": rows[:, H - bh:],
    }
    out = {}
    for k, R in regions.items():
        ref = np.median(R, axis=0)
        out[k] = np.array([_shift(R[i], ref, max_shift) for i in range(n)])
    return out


def median_sharpness(F):
    """Gradient energy of the temporal median -- higher means stiller.

    An independent check on the border measurements above, which read the very
    structure the render is cut on and so can flatter a lock that merely holds
    its own crop still.  Nothing is cut on the static background, so if the
    frames really sit on top of each other the wall and fence survive the
    median sharply, and if they wander the median smears.

    Only comparable between videos of the same content at the same size: it
    also rises with the render's intrinsic sharpness, so it ranks variants of
    one pipeline, not one pipeline against another source.
    """
    import cv2
    m = np.median(F, axis=0).astype(np.float32)
    m = (m - m.mean()) / (m.std() + 1e-9)
    gx = cv2.Sobel(m, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(m, cv2.CV_32F, 0, 1, 3)
    return float(np.mean(gx ** 2 + gy ** 2))


def rob(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan")
    return float(np.median(np.abs(x - np.median(x))) * 1.4826)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--band", type=float, default=0.10,
                    help="Fraction of the frame at each edge treated as the "
                         "border region.")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--margins", nargs="*", type=float, default=None,
                    help="Per-video render margin to crop off before measuring, "
                         "so every video is measured on the picture alone. One "
                         "value per video, or omit for all-zero.")
    ap.add_argument("--size", nargs=2, type=int, default=None,
                    metavar=("W", "H"),
                    help="Resample every video to this size first. Without it, "
                         "px are not comparable between videos of different "
                         "resolutions and the numbers below cannot be ranked.")
    args = ap.parse_args()
    if len(args.videos) != len(args.labels):
        raise SystemExit("--videos and --labels must match in length")
    margins = args.margins or [0.0] * len(args.videos)
    if len(margins) != len(args.videos):
        raise SystemExit("--margins must have one value per video")

    res = {}
    for path, lbl, mg in zip(args.videos, args.labels, margins):
        F = load(path, mg, args.size)
        s = border_shifts(F, args.band)
        across = 0.5 * (s["across_low"] + s["across_high"])
        across_sz = s["across_high"] - s["across_low"]
        along = 0.5 * (s["along_low"] + s["along_high"])
        along_sz = s["along_high"] - s["along_low"]
        d = dict(n=int(F.shape[0]),
                 across_pos=rob(across), across_size=rob(across_sz),
                 along_pos=rob(along), along_size=rob(along_sz),
                 across_jitter=float(np.nanmedian(np.abs(np.diff(across)))),
                 along_jitter=float(np.nanmedian(np.abs(np.diff(along)))),
                 median_sharpness=median_sharpness(F))
        res[lbl] = d
        print(f"\n=== {lbl}  ({d['n']} frames, {F.shape[1]}x{F.shape[2]}) ===")
        print(f"  ACROSS film   position sd {d['across_pos']:6.2f} px   "
              f"size sd {d['across_size']:6.2f} px   "
              f"frame-to-frame {d['across_jitter']:5.2f} px")
        print(f"  ALONG  film   position sd {d['along_pos']:6.2f} px   "
              f"size sd {d['along_size']:6.2f} px   "
              f"frame-to-frame {d['along_jitter']:5.2f} px")
        print(f"  temporal-median sharpness {d['median_sharpness']:7.4f} "
              f"(higher = stiller)")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "border_stillness.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {args.out_dir}/border_stillness.json")


if __name__ == "__main__":
    main()
