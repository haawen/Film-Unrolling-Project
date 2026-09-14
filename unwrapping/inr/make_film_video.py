"""Turn the unrolled whole-roll strip into a playable film video.

The arc axis of the strip is the film LENGTH (frame after frame); the z axis is
the film WIDTH. So a movie frame = full-z x (one film cell of arc).

Two ways to slice the arc into frames:
  * FIXED pitch (--phase-lock none): cut every `pitch` px from column 0. Simple,
    but if the true cell period drifts along the ~200k-px roll the cuts walk out
    of phase with the real cells (looks like a slow scroll, misaligns frame-by-
    frame vs GT).
  * PHASE-LOCKED (default): detect each real cell boundary from the periodic
    frame-line signal (band-pass the column-mean profile around 1/pitch, take its
    peaks — drift-adaptive), cut there, and resize each cell to a common width.
    Every output frame is then one cleanly-framed cell (best for frame-by-frame
    metrics vs the GT scan).

Writes an mp4 of ~`duration` s (fps = n_frames / duration) + a boundary diagnostic
`<out>.boundaries.png` (downscaled strip with the detected cuts drawn) so the cuts
can be verified.

Usage:
  python -m unwrapping.inr.make_film_video <strip.npy> <out.mp4> [duration_s] \
      [--pitch N] [--phase-lock adaptive|global|none] [--invert-phase] \
      [--rotate] [--reverse] [--invert] [--height H]
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


def refine_pitch(col, pitch0, smooth=2001):
    """FRACTIONAL pitch from a HIGH-lag autocorrelation peak (near k*pitch0). A
    tiny error in the coarse integer pitch causes a constant per-frame drift; using
    a peak many cells out and dividing by k averages the error away (sub-px pitch).
    Parabolic interpolation gives sub-integer lag."""
    d = _detrend(col, smooth)
    ac = np.correlate(d, d, "full")[len(d) - 1:]
    n = len(ac)
    k = max(1, (n // pitch0) // 3)                 # go ~1/3 of the way out
    c = k * pitch0; w = pitch0 // 2
    if c + w >= n:
        k = max(1, (n - w - 1) // pitch0); c = k * pitch0
    seg = ac[c - w:c + w]
    j = int(np.argmax(seg)); lag = c - w + j
    if 0 < j < len(seg) - 1:                       # parabolic sub-px peak
        y0, y1, y2 = seg[j - 1], seg[j], seg[j + 1]
        denom = (y0 - 2 * y1 + y2)
        if abs(denom) > 1e-9:
            lag += 0.5 * (y0 - y2) / denom
    return lag / k


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


def detect_boundaries(col, pitch, invert_phase=False):
    """Cell boundaries = peaks of the periodic frame-line component (~pitch apart,
    drift-adaptive). invert_phase flips to troughs (frame-line polarity)."""
    from scipy.signal import find_peaks
    p = periodic_component(col, pitch)
    sig = -p if invert_phase else p
    pk, _ = find_peaks(sig, distance=int(round(pitch * 0.6)))
    return np.asarray(pk)


def tracked_bounds(col, pitch_c, pitch_f, invert_phase=False, sigma_cells=2.0):
    """Drift-TRACKED cell boundaries via the Hilbert phase of the frame-line
    signal. 'global' assumes one constant pitch+phase, but the true frame-line
    phase drifts slowly along the strip (unroll arc stretch + film shrinkage),
    so fixed cuts walk into the cells and the picture visibly creeps. Here the
    unwrapped analytic phase is smoothed (amplitude-weighted, sigma ~2 cells,
    so dark low-signal stretches inherit their neighbours' phase instead of
    injecting noise — the failure mode of the peak-picking 'adaptive' mode)
    and boundaries are placed at each 2*pi crossing: cuts follow the drift and
    every cell stays framed in place."""
    from scipy.signal import hilbert
    from scipy.ndimage import gaussian_filter1d
    p = periodic_component(col, pitch_c)
    z = hilbert(p if not invert_phase else -p)
    amp = np.abs(z)
    ph = np.unwrap(np.angle(z))
    x = np.arange(len(ph), dtype=np.float64)
    lo, hi = int(0.08 * len(x)), int(0.92 * len(x))  # drop edgy ends
    slope, icpt = np.polyfit(x[lo:hi], ph[lo:hi], 1)
    resid = ph - (slope * x + icpt)
    # Only trust the phase where the frame-line amplitude is healthy: in dark
    # stretches (iris-out, End card, banding ends) the analytic phase slips
    # whole 2*pi cycles (observed: double-width cells). Between consecutive
    # trusted samples the genuine residual change is tiny (drift ~0.4 rad even
    # across a long gap), so any ~2*pi*k jump is a slip — remove the integer
    # cycles pairwise and rebuild by cumulative sum, then bridge untrusted
    # spans by interpolation (constant local pitch through the gap).
    trust = amp > 0.35 * np.median(amp)
    if trust.sum() > 100:
        xt, rt = x[trust], resid[trust]
        d = np.diff(rt)
        d -= 2 * np.pi * np.round(d / (2 * np.pi))
        rc = np.concatenate([[rt[0]], rt[0] + np.cumsum(d)])
        resid = np.interp(x, xt, rc)
    sig = sigma_cells * pitch_f
    resid_s = gaussian_filter1d(resid, sig)
    ph_s = np.maximum.accumulate(slope * x + icpt + resid_s)  # keep monotone
    # boundaries at analytic phase 0 mod 2*pi (= the peaks 'global' cuts at)
    k0 = int(np.ceil(ph_s[0] / (2 * np.pi)))
    k1 = int(np.floor(ph_s[-1] / (2 * np.pi)))
    targets = 2 * np.pi * np.arange(k0, k1 + 1)
    return np.interp(targets, ph_s, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npy"); ap.add_argument("out")
    ap.add_argument("duration", nargs="?", type=float, default=9.0)
    ap.add_argument("--pitch", type=float, default=0, help="Override frame pitch (px, may be fractional).")
    ap.add_argument("--no-refine-pitch", action="store_true",
                    help="Skip fractional-pitch refinement (keep the coarse integer).")
    ap.add_argument("--pitch-method", choices=["phase", "autocorr"], default="phase",
                    help="phase = phase-slope fit over whole strip (most drift-free); "
                         "autocorr = high-lag autocorrelation peak.")
    ap.add_argument("--phase-lock",
                    choices=["adaptive", "global", "track", "none"],
                    default="adaptive",
                    help="adaptive = detect each real cell boundary (drift-safe "
                         "but noisy); global = fixed pitch at the best single "
                         "phase; track = smoothed Hilbert-phase boundaries "
                         "(drift-adaptive AND robust — picture stays in place); "
                         "none = fixed pitch from column 0 (legacy).")
    ap.add_argument("--invert-phase", action="store_true",
                    help="Flip boundary polarity (cut on troughs not peaks).")
    ap.add_argument("--phase-offset-frac", type=float, default=0.0,
                    help="Shift every cell boundary by this FRACTION of the pitch "
                         "(0.5 = half a frame). --invert-phase is the same thing at "
                         "exactly 0.5; this is the continuous version, for when the "
                         "automatic phase lands part-way out. It happens: the global "
                         "lock picks the single offset maximising the boundary "
                         "signal, so a strip whose pitch or leading content changed "
                         "can settle on a different phase than the previous render "
                         "at identical flags.")
    ap.add_argument("--bounds-npy", default=None,
                    help="Explicit cell-boundary array (strip px), e.g. the "
                         "GT-validated video_bounds.npy exported by "
                         "unwrapping.eval.frame_match — overrides --phase-lock.")
    ap.add_argument("--rotate", action="store_true", help="Rotate frames 90 deg.")
    ap.add_argument("--rot-k", type=int, default=1, help="Number of CCW 90deg turns when --rotate (1 default; 3 = clockwise).")
    ap.add_argument("--reverse", action="store_true", help="Play outer->inner.")
    ap.add_argument("--invert", action="store_true",
                    help="Invert intensities (the CT film image is a NEGATIVE).")
    ap.add_argument("--height", type=int, default=600, help="Output frame height px.")
    args = ap.parse_args()

    s = np.load(args.npy, mmap_mode="r")
    Z, W = s.shape
    col = np.asarray(s).mean(axis=0)
    pitch_c = int(round(args.pitch)) if args.pitch > 0 else estimate_pitch(col)
    if args.pitch > 0:
        pitch_f = float(args.pitch)                        # user override (exact)
    elif args.no_refine_pitch:
        pitch_f = float(pitch_c)
    elif args.pitch_method == "phase":
        pitch_f = refine_pitch_phase(col, pitch_c)         # whole-strip phase slope
    else:
        pitch_f = refine_pitch(col, pitch_c)               # high-lag autocorr peak
    print(f"pitch coarse={pitch_c} fractional={pitch_f:.3f}", flush=True)

    # ── frame boundaries (FRACTIONAL pitch so cuts don't drift) ──
    if args.bounds_npy:
        bounds = np.unique(np.clip(
            np.round(np.load(args.bounds_npy)).astype(int), 0, W))
    elif args.phase_lock == "none":
        n = int(W / pitch_f)
        bounds = np.round(np.arange(n + 1) * pitch_f).astype(int)
    elif args.phase_lock == "track":
        bounds = np.round(tracked_bounds(col, pitch_c, pitch_f,
                                         args.invert_phase)).astype(int)
    elif args.phase_lock == "global":
        p = periodic_component(col, pitch_c)
        sig = -p if args.invert_phase else p
        # best single phase = offset maximizing the per-frame boundary signal
        score = [sig[np.round(np.arange(o, W - pitch_f, pitch_f)).astype(int)].sum()
                 for o in range(pitch_c)]
        o0 = int(np.argmax(score))
        n = int((W - o0) / pitch_f)
        bounds = np.round(o0 + np.arange(n + 1) * pitch_f).astype(int)
    else:                                                  # adaptive
        bounds = detect_boundaries(col, pitch_c, args.invert_phase)
    bounds = np.asarray(bounds)
    if args.phase_offset_frac:
        # Shift, then drop any cell pushed outside the strip so every emitted
        # frame is still a full pitch wide.
        bounds = bounds + int(round(args.phase_offset_frac * pitch_f))
        bounds = bounds[(bounds >= 0) & (bounds <= W)]
        print(f"  phase shifted by {args.phase_offset_frac:+.2f} pitch "
              f"({int(round(args.phase_offset_frac * pitch_f)):+d} px)", flush=True)
    n = len(bounds) - 1
    if n < 2:
        raise SystemExit(f"only {n} frames detected — check --pitch/--phase-lock")
    widths = np.diff(bounds)
    fps = max(6, int(round(n / args.duration)))
    order = range(n - 1, -1, -1) if args.reverse else range(n)
    samp = np.asarray(s[:, ::50], dtype=np.float32)
    lo, hi = np.percentile(samp, [1, 99])
    print(f"strip {s.shape}: pitch~{pitch_f:.2f}, phase-lock={args.phase_lock} -> "
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
            fr = np.rot90(fr, args.rot_k)
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
            fr = np.rot90(fr, args.rot_k)
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
