"""Standalone, GT-INDEPENDENT film stabilizer: turn the unrolled strip into a
truly STILL movie (the frame/static structure does not drift; only the drawn
character moves), using classic image-registration video stabilization.

Why the plain make_film_video output jitters:
  * cells are cut from the strip at slightly wrong along-arc positions (frame-
    line drift) -> the picture shifts frame to frame;
  * the strip itself has the film content wobbling in z along the arc (a walk
    residual) -> sideways shift;
  * make_film_video resizes each variable-width cell to a common size ->
    per-frame scale wobble.

Fix, in three stages:
  1. RE-CUT UNIFORM windows (fixed size, no per-cell resize -> no scale wobble)
     centered on drift-tracked cell centers detected from the strip's OWN
     frame-line signal (no GT used).
  2. STABILIZE by registering every frame to a LOCAL TEMPORAL-MEDIAN reference
     (the median over +/-W neighbours blurs the moving character away, leaving
     the static frame/background structure, so registration locks onto what
     should be still; median-reference = absolutely anchored, no accumulation
     drift). Translation (sub-pixel phase correlation on edge maps) + a small
     rotation search. Iterated a few times.
  3. CROP to the common valid region (borders sacrificed for a still frame, as
     requested) and write the mp4.

GT is optional and only used, if given, to fix the absolute framing offset
(otherwise the film is centered on its own median) -- never for the per-frame
stabilization.

Usage:
  python -m unwrapping.inr.stabilize_film \
      --strip unwrapping/inr/results/walk_dense_v12/wholeroll.npy \
      --out   unwrapping/inr/results/walk_dense_v12/film_v12_stab.mp4 \
      [--reverse --invert --height 720 --rot-search 1.5 --smooth-win 6]
"""

import argparse

import numpy as np
import imageio.v2 as imageio
from PIL import Image
from scipy.ndimage import sobel, shift as nd_shift, rotate as nd_rotate, \
    gaussian_filter1d

from .make_film_video import (estimate_pitch, refine_pitch_phase,
                              periodic_component, tracked_bounds)


# --------------------------------------------------------------------------- #
# registration primitives (numpy/scipy only)
# --------------------------------------------------------------------------- #
def gmag(f):
    """Normalized Sobel gradient magnitude (intensity-blind edge map)."""
    m = np.hypot(sobel(f, 0), sobel(f, 1))
    m -= m.min()
    mx = m.max()
    return m / mx if mx > 0 else m


def _parabolic(y0, y1, y2):
    d = y0 - 2 * y1 + y2
    return 0.5 * (y0 - y2) / d if abs(d) > 1e-9 else 0.0


def phase_shift(a, b, lim=0.25):
    """Sub-pixel (dy, dx) aligning b onto a via FFT phase correlation, peak
    restricted to +/-lim of the frame (a global shift, dominated by the static
    structure that fills the frame -- a small moving character can't move the
    global peak). Parabolic interpolation -> sub-pixel."""
    A = np.fft.rfft2(a - a.mean())
    B = np.fft.rfft2(b - b.mean())
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12
    r = np.fft.irfft2(R, s=a.shape)
    h, w = a.shape
    my, mx = max(1, int(lim * h)), max(1, int(lim * w))
    rows = np.zeros(h, bool); rows[:my + 1] = rows[-my:] = True
    cols = np.zeros(w, bool); cols[:mx + 1] = cols[-mx:] = True
    r = np.where(rows[:, None] & cols[None, :], r, -np.inf)
    py, px = np.unravel_index(np.argmax(r), r.shape)
    # sub-pixel via parabolic fit on the wrapped neighbourhood
    fy = _parabolic(r[(py - 1) % h, px], r[py, px], r[(py + 1) % h, px])
    fx = _parabolic(r[py, (px - 1) % w], r[py, px], r[py, (px + 1) % w])
    dy = (py if py <= h // 2 else py - h) + fy
    dx = (px if px <= w // 2 else px - w) + fx
    peak = r[py, px]
    return dy, dx, float(peak)


def register(ref, img, rot_search=0.0, rot_step=0.5):
    """(dy, dx, dth) aligning img onto ref. Optional small rotation search:
    for each candidate angle, rotate img, phase-correlate, keep the angle whose
    correlation peak is strongest."""
    rg = gmag(ref)
    if rot_search <= 0:
        dy, dx, _ = phase_shift(rg, gmag(img))
        return dy, dx, 0.0
    best = (-2.0, 0.0, 0.0, 0.0)
    for th in np.arange(-rot_search, rot_search + 1e-9, rot_step):
        im = nd_rotate(img, th, reshape=False, order=1, mode="nearest") \
            if th != 0 else img
        dy, dx, pk = phase_shift(rg, gmag(im))
        if pk > best[0]:
            best = (pk, dy, dx, th)
    return best[1], best[2], best[3]


def warp(img, dy, dx, dth, cval=0.0):
    """Rotate about the centre by dth then translate by (dy, dx)."""
    out = img
    if dth:
        out = nd_rotate(out, dth, reshape=False, order=1, mode="constant",
                        cval=cval)
    if dy or dx:
        out = nd_shift(out, (dy, dx), order=1, mode="constant", cval=cval)
    return out


# --------------------------------------------------------------------------- #
# cutting uniform windows from the strip
# --------------------------------------------------------------------------- #
def strip_norm(strip, z0, z1, sample_step=50):
    samp = np.asarray(strip[z0:z1, ::sample_step], np.float32)
    lo, hi = np.percentile(samp, [1, 99])
    return float(lo), float(hi)


def cut_uniform(strip, centers, hw, z0, z1, lo, hi, invert):
    """Uniform (fixed-size) window per cell centre: strip[z0:z1, c-hw:c+hw],
    normalized (+inverted: the CT film is a negative). No per-cell resize."""
    W = strip.shape[1]
    frames = []
    for c in centers:
        a = int(round(c)) - hw
        a = max(0, min(W - 2 * hw, a))
        f = np.asarray(strip[z0:z1, a:a + 2 * hw], np.float32)
        f = np.clip((f - lo) / (hi - lo + 1e-6), 0, 1)
        if invert:
            f = 1.0 - f
        frames.append(f)
    return frames


# --------------------------------------------------------------------------- #
# stabilization
# --------------------------------------------------------------------------- #
def stabilize(frames, win=6, iters=3, rot_search=1.5, lock=True,
              smooth_sigma=0.0):
    """Lock the static structure via local-temporal-median references.

    frames: list of same-shape float [0,1]. Returns (params, cropbox):
      params[i] = (dy, dx, dth) to apply to frames[i] (absolute, drift-free),
      cropbox = (y0, y1, x0, x1) common valid region after warping.

    lock=True pins the static structure to its temporal median (perfectly
    still). smooth_sigma>0 instead removes only motion faster than that (keeps
    a genuine slow pan) -- off by default (this film is a locked shot).
    """
    n = len(frames)
    params = np.zeros((n, 3))
    stk = np.stack(frames)

    def aligned_stack(p):
        return np.stack([warp(frames[i], *p[i]) for i in range(n)])

    cur = stk
    for it in range(iters):
        for i in range(n):
            lo, hi = max(0, i - win), min(n, i + win + 1)
            ref = np.median(cur[lo:hi], axis=0)
            dy, dx, dth = register(ref, warp(frames[i], *params[i]),
                                   rot_search if it == 0 else 0.0)
            params[i, 0] += dy
            params[i, 1] += dx
            params[i, 2] += dth
        cur = aligned_stack(params)

    # reject correction OUTLIERS: at scene cuts the local-median reference
    # blends two scenes -> a spurious one-frame spike that would ADD a visible
    # jolt. Replace any correction far from its temporal-median with that
    # median (kills the isolated spikes, keeps the small real per-frame fixes).
    from scipy.ndimage import median_filter
    for k in range(3):
        med = median_filter(params[:, k], 5, mode="nearest")
        resid = params[:, k] - med
        mad = np.median(np.abs(resid - np.median(resid))) * 1.4826 + 1e-6
        bad = np.abs(resid) > max(5 * mad, 0.8)
        params[bad, k] = med[bad]

    # remove the DC (keep content centred); optionally keep only slow motion
    if lock:
        params[:, :2] -= params[:, :2].mean(0)
    elif smooth_sigma > 0:
        for k in range(3):
            params[:, k] -= gaussian_filter1d(params[:, k], smooth_sigma,
                                               mode="nearest")
    else:
        params[:, :2] -= params[:, :2].mean(0)

    # common valid crop: shrink by the max applied translation (+ small
    # rotation margin) so no black border enters any frame
    h, w = frames[0].shape
    my = int(np.ceil(np.abs(params[:, 0]).max())) + 1
    mx = int(np.ceil(np.abs(params[:, 1]).max())) + 1
    rot = np.abs(params[:, 2]).max()
    rm_y = int(np.ceil(np.tan(np.deg2rad(rot)) * w / 2)) + 1
    rm_x = int(np.ceil(np.tan(np.deg2rad(rot)) * h / 2)) + 1
    y0, y1 = my + rm_y, h - my - rm_y
    x0, x1 = mx + rm_x, w - mx - rm_x
    return params, (y0, y1, x0, x1)


# --------------------------------------------------------------------------- #
def frame_to_frame_motion(frames, span=(None, None)):
    """Mean |dy|,|dx| of consecutive frames (global shift of static structure).
    A proxy for visible jitter -- lower is stiller. Includes some real content
    motion, so compare relative (before vs after), not to zero."""
    lo = span[0] or 0
    hi = span[1] or len(frames)
    dys, dxs = [], []
    for i in range(lo + 1, hi):
        dy, dx, _ = phase_shift(gmag(frames[i - 1]), gmag(frames[i]))
        dys.append(abs(dy)); dxs.append(abs(dx))
    return float(np.mean(dys)), float(np.mean(dxs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--z-crop", default="0.0,1.0",
                    help="z (film-width) crop lo,hi fractions.")
    ap.add_argument("--margin", type=float, default=1.25,
                    help="uniform window width in units of the cell pitch.")
    ap.add_argument("--win", type=int, default=6,
                    help="+/- frames for the local-median reference.")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--rot-search", type=float, default=1.5,
                    help="max rotation search (deg); 0 = translation only.")
    ap.add_argument("--smooth-sigma", type=float, default=0.0,
                    help="if >0, keep motion slower than this (frames) instead "
                         "of fully locking (for a genuine slow pan).")
    ap.add_argument("--duration", type=float, default=9.0)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--extra-crop", type=float, default=0.0,
                    help="extra fractional border crop after stabilization.")
    args = ap.parse_args()

    print("Loading strip...", flush=True)
    strip = np.load(args.strip, mmap_mode="r")
    Z, W = strip.shape
    zc = [float(x) for x in args.z_crop.split(",")]
    z0, z1 = int(zc[0] * Z), int(zc[1] * Z)

    # GT-independent cell centres from the strip's own frame-line signal
    col = np.asarray(strip[z0:z1], np.float32).mean(0)
    pc = estimate_pitch(col)
    pf = refine_pitch_phase(col, pc)
    bounds = tracked_bounds(col, pc, pf)
    centers = (bounds[:-1] + bounds[1:]) / 2.0
    hw = int(round(args.margin * pf / 2))
    print(f"  strip {strip.shape}, pitch {pf:.1f}, {len(centers)} cells, "
          f"window {2*hw}px arc", flush=True)

    lo, hi = strip_norm(strip, z0, z1)
    frames = cut_uniform(strip, centers, hw, z0, z1, lo, hi, args.invert)

    before = frame_to_frame_motion(frames)
    print(f"  frame-to-frame motion BEFORE: |dy| {before[0]:.2f}px  "
          f"|dx| {before[1]:.2f}px", flush=True)

    print("Stabilizing (local-median reference)...", flush=True)
    params, (y0, y1, x0, x1) = stabilize(
        frames, win=args.win, iters=args.iters, rot_search=args.rot_search,
        lock=(args.smooth_sigma <= 0), smooth_sigma=args.smooth_sigma)
    warped = [warp(frames[i], *params[i]) for i in range(len(frames))]

    after = frame_to_frame_motion(warped)
    print(f"  frame-to-frame motion AFTER : |dy| {after[0]:.2f}px  "
          f"|dx| {after[1]:.2f}px  (crop {y1-y0}x{x1-x0} from "
          f"{frames[0].shape[0]}x{frames[0].shape[1]})", flush=True)

    # crop + optional extra border trim
    ec = args.extra_crop
    cy = int((y1 - y0) * ec / 2); cx = int((x1 - x0) * ec / 2)
    y0 += cy; y1 -= cy; x0 += cx; x1 -= cx
    cropped = [w[y0:y1, x0:x1] for w in warped]

    # render
    order = range(len(cropped) - 1, -1, -1) if args.reverse \
        else range(len(cropped))
    H = args.height
    fh, fw = cropped[0].shape
    Wf = int(round(H * fw / fh)); Wf += Wf % 2
    fps = max(6, int(round(len(cropped) / args.duration)))
    wr = imageio.get_writer(args.out, fps=fps, codec="libx264", quality=8,
                            macro_block_size=1)
    for i in order:
        im = Image.fromarray((np.clip(cropped[i], 0, 1) * 255).astype(np.uint8)
                             ).resize((Wf, H))
        wr.append_data(np.asarray(im))
    wr.close()

    # diagnostic: trajectory + a 16-frame montage
    _save_diag(params, cropped, order, args.out)
    print(f"Done -> {args.out}  ({H}x{Wf}, {len(cropped)/fps:.1f}s @ {fps}fps)",
          flush=True)


def _save_diag(params, cropped, order, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax[0].plot(params[:, 0], label="dy (px)")
    ax[0].plot(params[:, 1], label="dx (px)")
    ax[0].legend(); ax[0].set_ylabel("applied shift")
    ax[0].set_title("stabilization: applied per-frame correction")
    ax[1].plot(params[:, 2], label="rotation (deg)", color="C2")
    ax[1].legend(); ax[1].set_xlabel("cell"); ax[1].set_ylabel("deg")
    fig.tight_layout(); fig.savefig(out + ".traj.png", dpi=110); plt.close(fig)

    ordl = list(order)
    idx = ordl[::max(1, len(ordl) // 16)][:16]
    fh, fw = cropped[0].shape
    th, tw = 240, max(2, int(240 * fw / fh))
    mon = np.zeros((2 * th, 8 * tw), np.uint8)
    for j, i in enumerate(idx):
        t = np.asarray(Image.fromarray((np.clip(cropped[i], 0, 1) * 255
                       ).astype(np.uint8)).resize((tw, th)))
        mon[(j // 8) * th:(j // 8) * th + th, (j % 8) * tw:(j % 8) * tw + tw] = t
    Image.fromarray(mon).save(out + ".frames.png")


if __name__ == "__main__":
    main()
