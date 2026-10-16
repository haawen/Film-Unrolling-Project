"""Turn the unrolled whole-roll strip into a playable film video.

The arc axis of the strip is the film LENGTH (frame after frame); the z axis is
the film WIDTH. So a movie frame = full-z x (one frame-pitch of arc). We estimate
the frame pitch from the periodicity along the arc (autocorrelation of the
de-trended column profile), slice the strip into frames, and write them in order
as an mp4 of ~`duration` seconds (fps = n_frames / duration).

Usage:
  python -m unwrapping.inr.make_film_video <strip.npy> <out.mp4> [duration_s] \
      [--pitch N] [--rotate] [--reverse] [--height H]
"""

import argparse

import numpy as np
import imageio.v2 as imageio
from PIL import Image


def estimate_pitch(col, lo=250, hi=1600, smooth=2001):
    """Frame pitch (px) = strongest autocorrelation lag of the de-trended profile."""
    k = smooth | 1
    pad = np.pad(col, k // 2, mode="edge")
    sm = np.convolve(pad, np.ones(k) / k, "valid")[:len(col)]
    d = col - sm
    d = d - d.mean()
    ac = np.correlate(d, d, "full")[len(d) - 1:]
    hi = min(hi, len(ac) - 1)
    return lo + int(np.argmax(ac[lo:hi]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npy"); ap.add_argument("out")
    ap.add_argument("duration", nargs="?", type=float, default=9.0)
    ap.add_argument("--pitch", type=int, default=0, help="Override frame pitch (px).")
    ap.add_argument("--rotate", action="store_true", help="Rotate frames 90 deg.")
    ap.add_argument("--reverse", action="store_true", help="Play outer->inner.")
    ap.add_argument("--invert", action="store_true",
                    help="Invert intensities (the CT film image is a NEGATIVE).")
    ap.add_argument("--height", type=int, default=600, help="Output frame height px.")
    args = ap.parse_args()

    s = np.load(args.npy, mmap_mode="r")
    Z, W = s.shape
    col = np.asarray(s).mean(axis=0)
    pitch = args.pitch if args.pitch > 0 else estimate_pitch(col)
    n = W // pitch
    fps = max(6, int(round(n / args.duration)))
    order = range(n - 1, -1, -1) if args.reverse else range(n)
    samp = np.asarray(s[:, ::50], dtype=np.float32)
    lo, hi = np.percentile(samp, [1, 99])
    print(f"strip {s.shape}: pitch={pitch} -> {n} frames, fps={fps}, "
          f"dur={n / fps:.1f}s", flush=True)

    fh = pitch if args.rotate else Z
    fw = Z if args.rotate else pitch
    H = args.height
    Wf = int(round(H * fw / fh)); Wf += Wf % 2
    wr = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=8,
                            macro_block_size=1)
    for c, i in enumerate(order):
        fr = np.asarray(s[:, i * pitch:(i + 1) * pitch], dtype=np.float32)
        fr = np.clip((fr - lo) / (hi - lo + 1e-6), 0, 1)
        if args.invert:
            fr = 1.0 - fr
        if args.rotate:
            fr = np.rot90(fr)
        im = Image.fromarray((fr * 255).astype(np.uint8)).resize((Wf, H))
        wr.append_data(np.asarray(im))
        if c % 50 == 0:
            print(f"  frame {c}/{n}", flush=True)
    wr.close()
    print(f"Done -> {args.out}  ({H}x{Wf}, {n / fps:.1f}s)")


if __name__ == "__main__":
    main()
