"""Turn the unrolled whole-roll strip into a playable film video.

The arc axis of the strip is the film LENGTH (frame after frame); the z axis is
the film WIDTH. So a movie frame = full-z x (one film cell of arc).

Cells are cut at a constant (fractional) pitch with the single best phase of the
periodic frame-line signal (the column-mean profile band-passed around 1/pitch).
The pitch is estimated from the strip (autocorrelation, refined by a phase-slope
fit over the whole strip) unless --pitch is given.

Writes an mp4 of ~`duration` s (fps = n_frames / duration), a boundary diagnostic
`<out>.boundaries.png` (downscaled strip with the cuts drawn) and a 16-frame sheet
`<out>.frames.png` spread over the reel (a pitch error shows as creeping framing).

Usage:
  python -m virtual_unrolling.video.make_film_video <strip.npy> <out.mp4> [duration_s] \
      [--pitch N] [--invert-phase] [--rotate] [--reverse] [--invert] [--height H]
"""

import argparse

import numpy as np
import imageio.v2 as imageio
from PIL import Image


def _detrend(col, smooth=2001):
    k = smooth | 1
    pad = np.pad(col, k // 2, mode="edge")
    sm = np.convolve(pad, np.ones(k) / k, "valid")[:len(col)]
    d = col - sm
    return d - d.mean()


def estimate_pitch(col, lo=250, hi=1600, smooth=2001):
    """Coarse frame pitch (px) = strongest autocorrelation lag (integer).

    Guards against locking onto a SUB-HARMONIC: within-frame periodic structure
    (sprocket/frame-line sub-features) can make ac[p/k] rival ac[p] at the true
    frame pitch p, so a plain argmax sometimes returns p/2 or p/3 (seen on the
    full-density Mickey strips: argmax 293 = the ~880 fundamental / 3, giving 720
    cells instead of ~234). If a multiple k·p (k=3,2) is in range and its
    autocorrelation is comparable, that multiple is the real fundamental."""
    ac = np.correlate(_detrend(col, smooth), _detrend(col, smooth), "full")[len(col) - 1:]
    hi = min(hi, len(ac) - 1)
    p = lo + int(np.argmax(ac[lo:hi]))
    for k in (3, 2):
        if k * p <= hi and ac[k * p] > 0.7 * ac[p]:
            return k * p
    return p


def refine_pitch_phase(col, pitch0, smooth=2001):
    """Most precise pitch: fit a line to the UNWRAPPED phase of the frame-line
    signal over the whole strip; average pitch = 2*pi / slope. Averaging over all
    cells makes any residual constant drift vanish (drift = pitch error)."""
    from scipy.signal import hilbert
    p = periodic_component(col, pitch0)
    ph = np.unwrap(np.angle(hilbert(p)))
    x = np.arange(len(ph))
    lo, hi = int(0.08 * len(x)), int(0.92 * len(x))     # drop edgy ends
    slope = np.polyfit(x[lo:hi], ph[lo:hi], 1)[0]
    return 2.0 * np.pi / abs(slope)


def periodic_component(col, pitch, bandfrac=0.4, smooth=2001):
    """Band-pass the de-trended column profile around 1/pitch -> the periodic
    frame-line signal (drift-adaptive: keeps a band of frequencies, not just one)."""
    k = smooth | 1
    pad = np.pad(col, k // 2, mode="edge")
    sm = np.convolve(pad, np.ones(k) / k, "valid")[:len(col)]
    d = col - sm
    F = np.fft.rfft(d)
    freq = np.fft.rfftfreq(len(d))
    f0 = 1.0 / pitch
    F[np.abs(freq - f0) > bandfrac * f0] = 0.0
    return np.fft.irfft(F, n=len(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npy"); ap.add_argument("out")
    ap.add_argument("duration", nargs="?", type=float, default=9.0)
    ap.add_argument("--pitch", type=float, default=0,
                    help="frame pitch in strip px (fractional); default: estimated")
    ap.add_argument("--invert-phase", action="store_true",
                    help="cut on troughs instead of peaks of the frame-line signal")
    ap.add_argument("--rotate", action="store_true", help="rotate frames 90 deg CCW")
    ap.add_argument("--reverse", action="store_true", help="play outer->inner")
    ap.add_argument("--invert", action="store_true",
                    help="invert intensities (the CT film image is a negative)")
    ap.add_argument("--height", type=int, default=600, help="output frame height px")
    args = ap.parse_args()

    s = np.load(args.npy, mmap_mode="r")
    Z, W = s.shape
    col = np.asarray(s).mean(axis=0)
    pitch_c = int(round(args.pitch)) if args.pitch > 0 else estimate_pitch(col)
    if args.pitch > 0:
        pitch_f = float(args.pitch)                        # user override (exact)
    else:
        pitch_f = refine_pitch_phase(col, pitch_c)         # whole-strip phase slope
    print(f"pitch coarse={pitch_c} fractional={pitch_f:.3f}", flush=True)

    # ── frame boundaries (FRACTIONAL pitch so cuts don't drift) ──
    p = periodic_component(col, pitch_c)
    sig = -p if args.invert_phase else p
    # best single phase = offset maximizing the per-frame boundary signal
    score = [sig[np.round(np.arange(o, W - pitch_f, pitch_f)).astype(int)].sum()
             for o in range(pitch_c)]
    o0 = int(np.argmax(score))
    n = int((W - o0) / pitch_f)
    bounds = np.round(o0 + np.arange(n + 1) * pitch_f).astype(int)
    n = len(bounds) - 1
    if n < 2:
        raise SystemExit(f"only {n} frames detected — check --pitch")
    widths = np.diff(bounds)
    fps = max(6, int(round(n / args.duration)))
    order = range(n - 1, -1, -1) if args.reverse else range(n)
    samp = np.asarray(s[:, ::50], dtype=np.float32)
    lo, hi = np.percentile(samp, [1, 99])
    print(f"strip {s.shape}: pitch~{pitch_f:.2f}, phase-lock=global -> "
          f"{n} cells (width {widths.min()}-{widths.max()}, med {int(np.median(widths))}), "
          f"fps={fps}, dur={n / fps:.1f}s", flush=True)

    # ── boundary diagnostic (downscaled strip + detected cuts) ──
    stepc = max(1, W // 4000)
    sm_img = np.clip((np.asarray(s[:, ::stepc], np.float32) - lo) / (hi - lo + 1e-6), 0, 1)
    if args.invert:
        sm_img = 1.0 - sm_img
    rgb = np.repeat((sm_img[..., None] * 255).astype(np.uint8), 3, axis=2)
    for b in bounds:
        x = int(b // stepc)
        if 0 <= x < rgb.shape[1]:
            rgb[:, max(0, x - 1):x + 1] = np.array([255, 40, 40], np.uint8)
    Image.fromarray(rgb).save(args.out + ".boundaries.png")

    # ── write frames (each cell resized to a common size) ──
    fw = int(np.median(widths))
    fh = fw if args.rotate else Z
    fwid = Z if args.rotate else fw
    H = args.height
    Wf = int(round(H * fwid / fh)); Wf += Wf % 2
    wr = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=8,
                            macro_block_size=1)
    for c, i in enumerate(order):
        fr = np.asarray(s[:, bounds[i]:bounds[i + 1]], dtype=np.float32)
        fr = np.clip((fr - lo) / (hi - lo + 1e-6), 0, 1)
        if args.invert:
            fr = 1.0 - fr
        if args.rotate:
            fr = np.rot90(fr, 1)
        im = Image.fromarray((fr * 255).astype(np.uint8)).resize((Wf, H))
        wr.append_data(np.asarray(im))
        if c % 50 == 0:
            print(f"  frame {c}/{n}", flush=True)
    wr.close()

    # ── frame montage: 16 cells SPREAD across the whole video (start->end), so a
    #    residual pitch DRIFT shows up as the framing creeping across the montage ──
    ordl = list(order)
    idxs = ordl[::max(1, len(ordl) // 16)][:16]
    tiles = []
    for i in idxs:
        fr = np.asarray(s[:, bounds[i]:bounds[i + 1]], dtype=np.float32)
        fr = np.clip((fr - lo) / (hi - lo + 1e-6), 0, 1)
        if args.invert:
            fr = 1.0 - fr
        if args.rotate:
            fr = np.rot90(fr, 1)
        tiles.append(np.asarray(
            Image.fromarray((fr * 255).astype(np.uint8)).resize((160, 240))))
    cols = 8; rows = int(np.ceil(len(tiles) / cols))
    mon = np.zeros((rows * 240, cols * 160), np.uint8)
    for j, t in enumerate(tiles):
        mon[(j // cols) * 240:(j // cols) * 240 + 240,
            (j % cols) * 160:(j % cols) * 160 + 160] = t
    Image.fromarray(mon).save(args.out + ".frames.png")
    print(f"Done -> {args.out}  ({H}x{Wf}, {n / fps:.1f}s); "
          f"diag {args.out}.boundaries.png + .frames.png")


if __name__ == "__main__":
    main()
