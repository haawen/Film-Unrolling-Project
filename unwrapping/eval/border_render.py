"""Render the film with the PICTURE BORDER as the video's border.

Each frame is cropped to its own detected border rectangle and resampled to one
common output size.  Framing and stabilisation are therefore the same operation:
holding the border still removes the frame-to-frame translation AND the size
"breathing", which the border measurement shows is the dominant motion here (the
two picture edges are anti-correlated at -0.65, i.e. they move in and out about
the centre rather than together).

Sampling is done ONCE, straight from the strip into the output grid, so the
result is not a warp of an already-resampled video.

Why the border and not the perforations: the perf rows sit in z bands the walk
never traced, so each row reports its own extrapolation error and the two
disagree (-0.07 across the film).  The border lies inside the walked picture
band and carries the same geometry as the content it registers.

    python -m unwrapping.eval.border_render <strip.npy> --border border.npz \
        --out-dir <dir> --reverse --invert
"""

import argparse
import os

import numpy as np


def _fill_and_smooth(y, sigma_frames=1.0, n_sigma=4.0):
    """Interpolate missing frames, drop gross outliers, then smooth lightly.

    Outliers are REMOVED before smoothing rather than by it: one frame whose
    edge was mis-fit by 300 px would otherwise drag a Gaussian mean for several
    frames either side of it.
    """
    from scipy.ndimage import median_filter, gaussian_filter1d
    y = np.asarray(y, float).copy()
    g = np.arange(len(y))
    ok = np.isfinite(y)
    if ok.sum() < 5:
        raise SystemExit("too few border measurements to build a track")
    y = np.interp(g, g[ok], y[ok])
    r = y - median_filter(y, 9, mode="nearest")
    s = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-9
    keep = np.abs(r) < n_sigma * s
    y = np.interp(g, g[keep], y[keep])
    n_bad = int((~ok).sum() + (~keep).sum())
    if sigma_frames > 0:
        y = gaussian_filter1d(y, sigma_frames, mode="nearest")
    return y, n_bad


def register_along(strip, c_lo, c_hi, pitch, z_band, win_frac=0.14,
                   max_shift_frac=0.12, smooth_frames=0.8, verbose=True):
    """Lock the along-film cut by REGISTERING each frame's boundary region
    against the reel median, instead of detecting the frame line.

    Detecting the frame line failed four different ways: this reel is largely a
    single static scene, so persistent content survives averaging and the folded
    profile carries two humps per period whose relative strength flips partway
    along the reel, so every feature-picking scheme locked onto the wrong one
    somewhere.  Registration never has to decide WHICH feature is the frame
    line -- it aligns whatever structure is in the boundary region against the
    same structure averaged over the reel, so the ambiguity cannot arise.  It is
    the same cross-correlation that `border_stillness.py` uses to measure the
    result, which is what showed it works.

    The window is deliberately narrow (a fraction of a pitch about the cut) so
    the profile is dominated by the border rather than by picture content.
    """
    from scipy.ndimage import median_filter, gaussian_filter1d
    z0, z1 = int(z_band[0]), int(z_band[1])
    n = len(c_lo)
    half = int(win_frac * pitch)
    W = strip.shape[1]

    prof = []
    keep = np.zeros(n, bool)
    for i in range(n):
        c = int(round(c_lo[i]))
        a, b = c - half, c + half
        if a < 0 or b > W:
            prof.append(None)
            continue
        prof.append(np.asarray(strip[z0:z1:4, a:b], np.float32).mean(axis=0)
                    .astype(np.float64))
        keep[i] = True
    good = [p for p in prof if p is not None]
    ref = np.median(np.stack(good), axis=0)

    from unwrapping.eval.border_stillness import _shift
    ms = int(max_shift_frac * pitch)
    sh = np.full(n, np.nan)
    for i in range(n):
        if prof[i] is not None:
            sh[i] = _shift(prof[i], ref, ms)

    ok = np.isfinite(sh)
    g = np.arange(n)
    sh = np.interp(g, g[ok], sh[ok])
    r = sh - median_filter(sh, 9, mode="nearest")
    s = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-9
    good_m = np.abs(r) < 4 * s
    nbad = int((~ok).sum() + (~good_m).sum())
    sh = np.interp(g, g[good_m], sh[good_m])
    if smooth_frames > 0:
        sh = gaussian_filter1d(sh, smooth_frames, mode="nearest")
    sh = sh - np.median(sh)                 # keep the reel's overall framing

    info = dict(sd=float(np.std(sh)), ptp=float(np.ptp(sh)), nbad=nbad,
                shift=sh)
    return c_lo + sh, c_hi + sh, info


def _bilinear(slab, z, c, c_off):
    H, Wl = slab.shape
    cc = c - c_off
    z0 = np.clip(np.floor(z).astype(np.int32), 0, H - 2)
    c0 = np.clip(np.floor(cc).astype(np.int32), 0, Wl - 2)
    fz = np.clip(z - z0, 0, 1)
    fc = np.clip(cc - c0, 0, 1)
    v00 = slab[z0, c0]; v01 = slab[z0, c0 + 1]
    v10 = slab[z0 + 1, c0]; v11 = slab[z0 + 1, c0 + 1]
    return ((v00 * (1 - fc) + v01 * fc) * (1 - fz) +
            (v10 * (1 - fc) + v11 * fc) * fz)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strip")
    ap.add_argument("--border", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--video", default="film_borderlock.mp4")
    ap.add_argument("--duration", type=float, default=9.0)
    ap.add_argument("--margin", type=float, default=0.04,
                    help="Surround kept outside the border, as a fraction of "
                         "the picture size. A thin sliver leaves the border "
                         "visible, so a still border is visible proof that the "
                         "lock worked; 0 crops exactly to the picture.")
    ap.add_argument("--smooth-frames", type=float, default=0.8,
                    help="Light smoothing of the border track, in frames. The "
                         "per-frame measurement noise is ~4 px against a 5-12 px "
                         "signal, so a little smoothing helps; too much would "
                         "remove the breathing we are trying to correct.")
    ap.add_argument("--along", choices=["phaselock", "border"],
                    default="phaselock",
                    help="Source of the along-film cuts. phaselock (default) "
                         "reuses make_film_video's global lock, which already "
                         "frames this strip correctly; border uses the detected "
                         "frame lines, which are not yet reliable -- see the "
                         "note in main().")
    ap.add_argument("--invert-phase", action="store_true", default=True,
                    help="Frame-line polarity for the phase lock. Required on "
                         "this scan (its strip carries the perf rows, which are "
                         "periodic at exactly the frame pitch).")
    ap.add_argument("--along-offsets", default=None,
                    help=".npy of per-frame (y,x) offsets from "
                         "stabilize_horizontal --save-offsets, in THAT video's "
                         "pixels. Folded into the sampling here, so the frame "
                         "keeps its full size and correct aspect and is "
                         "resampled once. Applying them to a rendered video "
                         "instead costs a second resampling and a ~29 % crop to "
                         "hide the edges the shifts expose.")
    ap.add_argument("--along-offsets-scale", type=float, default=0.0,
                    help="Strip px per offset px (0 = derive from the render's "
                         "own geometry).")
    ap.add_argument("--along-offsets-sign", type=float, default=1.0,
                    help="Flip if the correction goes the wrong way; the "
                         "derivation is in the note where they are applied.")
    ap.add_argument("--along-register", action="store_true",
                    help="After the phase lock, register each frame's boundary "
                         "region against the reel median to remove the residual "
                         "along-film drift. See register_along().")
    ap.add_argument("--register-win", type=float, default=0.14,
                    help="Half-window for that registration, as a fraction of "
                         "the pitch. Narrow keeps it border-dominated.")
    ap.add_argument("--along-shift", type=float, default=0.0,
                    help="Translate the along-film cut, in fractions of the "
                         "pitch. The phase lock frames a COMPLETE picture but "
                         "not necessarily the picture the frame lines bound: "
                         "measured against GT's aperture (which is fixed to "
                         "GT's own perforations) this render sat 0.128 pitch "
                         "off, putting the frame line 13 %% up from the bottom "
                         "edge with a sliver of the next frame below it.")
    ap.add_argument("--across-shift", type=float, default=0.0,
                    help="Translate the across-film window, in fractions of the "
                         "picture height.")
    ap.add_argument("--across-scale", type=float, default=1.0,
                    help="Scale the across-film window about its centre. The "
                         "detected border runs ~4 %% wider than GT's aperture, "
                         "which is why a perforation shows at one edge.")
    ap.add_argument("--along-scale", type=float, default=1.0,
                    help="Scale the along-film window about its centre.")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--no-rotate", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    s = np.load(args.strip, mmap_mode="r")
    B = np.load(args.border)
    Z, W = s.shape
    pitch = float(B["pitch"])
    lines = np.asarray(B["lines"], float)
    sides = np.asarray(B["sides"], float)
    n = min(len(lines) - 1, len(sides))
    print(f"strip {s.shape}; {n} frames from {args.border}", flush=True)

    z_lo, nb0 = _fill_and_smooth(sides[:n, 0], args.smooth_frames)
    z_hi, nb1 = _fill_and_smooth(sides[:n, 1], args.smooth_frames)

    if args.along == "phaselock":
        # ALONG-film cuts from make_film_video's global phase lock, not from the
        # detected frame lines.
        #
        # The frame lines are the one part of the border that could NOT be
        # detected reliably: this film is largely a single static scene, so
        # persistent content survives averaging over all 229 frames, and the
        # folded column profile carries TWO content humps per period whose
        # relative strength flips partway along the reel.  Four signals were
        # tried -- raw peak, matched filter with a global template, with a local
        # template, and across-z variance -- and all either locked onto the
        # wrong hump or followed the flip, leaving part of the reel cut half a
        # frame out.  The existing global phase lock already frames this strip
        # correctly (229 cells, one complete picture each), so it is used as-is
        # and the border supplies only the ACROSS-film correction, which is the
        # part that was missing.
        from unwrapping.inr.make_film_video import (periodic_component,
                                                    refine_pitch_phase)
        col = np.asarray(s).mean(axis=0)
        pc = int(round(pitch))
        pf = refine_pitch_phase(col, pc)
        per = periodic_component(col, pc)
        sig = -per if args.invert_phase else per
        sc = [sig[np.round(np.arange(o, W - pf, pf)).astype(int)].sum()
              for o in range(pc)]
        o0 = int(np.argmax(sc))
        nb = int((W - o0) / pf)
        bounds = np.round(o0 + np.arange(nb + 1) * pf).astype(float)
        n = min(n, len(bounds) - 1)
        z_lo, z_hi = z_lo[:n], z_hi[:n]
        c_lo, c_hi = bounds[:n], bounds[1:n + 1]
        print(f"  along-film: phase lock, pitch {pf:.2f}, {n} cells", flush=True)

        if args.along_register:
            c_lo, c_hi, info = register_along(
                s, c_lo, c_hi, pitch,
                z_band=(float(np.median(z_lo)), float(np.median(z_hi))),
                win_frac=args.register_win, verbose=True)
            print(f"  along-film registration: shift sd {info['sd']:.2f} px, "
                  f"ptp {info['ptp']:.1f} px, {info['nbad']} frames rejected",
                  flush=True)
    else:
        c_lo = lines[:n]
        c_hi = lines[1:n + 1]
        print(f"  along-film: detected frame lines, {n} cells", flush=True)
    print(f"  side borders: {nb0}+{nb1} frames filled from neighbours", flush=True)
    print(f"  picture height med {np.median(z_hi - z_lo):.1f} px "
          f"(ptp {np.ptp(z_hi - z_lo):.1f}), "
          f"width med {np.median(c_hi - c_lo):.1f} px "
          f"(ptp {np.ptp(c_hi - c_lo):.1f})", flush=True)

    if args.along_offsets:
        off = np.load(args.along_offsets)
        oy = np.asarray(off[:, 0] if off.ndim == 2 else off, float)
        # FRAME ORDER. The offsets were measured on a rendered video, and that
        # render used --reverse, so its frame i is cell n-1-i. Applying them in
        # cell order puts every correction on the wrong frame, backwards.
        if args.reverse:
            oy = oy[::-1]
        if len(oy) < n:
            oy = np.pad(oy, (0, n - len(oy)), mode="edge")
        oy = oy[:n]
        # SCALE. The offsets are in that video's pixels, whose height covered
        # the whole output cell INCLUDING the margin, not one bare pitch.
        sc = args.along_offsets_scale
        if sc <= 0:
            sc = float(np.median(c_hi - c_lo)) * (1 + 2 * args.margin) / 720.0
        # SIGN. np.rot90 makes video row = nC-1-along, so a window shift of
        # +delta in c moves the content DOWN by delta rows -- the same direction
        # nd_shift(+y) moves it. So the window shift has the SAME sign as the
        # offset, not the opposite.
        oy = args.along_offsets_sign * oy * sc
        oy = oy - np.median(oy)
        c_lo = c_lo + oy
        c_hi = c_hi + oy
        print(f"  along-film offsets folded into the sampling: "
              f"scale {sc:.3f} strip px per offset px, sign "
              f"{args.along_offsets_sign:+.0f}, reversed={args.reverse}, "
              f"shift sd {np.std(oy):.1f} px, ptp {np.ptp(oy):.1f} px",
              flush=True)

    # FRAMING. Shift/scale the sampling window itself rather than the rendered
    # frame, so the content that moves into view is real film from the strip
    # instead of replicated edge.
    if args.along_shift or args.along_scale != 1.0:
        d = (c_hi - c_lo)
        mid = 0.5 * (c_lo + c_hi) + args.along_shift * d
        c_lo = mid - 0.5 * d * args.along_scale
        c_hi = mid + 0.5 * d * args.along_scale
        print(f"  along framing: shift {args.along_shift:+.4f} pitch, "
              f"scale {args.along_scale:.4f}", flush=True)
    if args.across_shift or args.across_scale != 1.0:
        d = (z_hi - z_lo)
        mid = 0.5 * (z_lo + z_hi) + args.across_shift * d
        z_lo = mid - 0.5 * d * args.across_scale
        z_hi = mid + 0.5 * d * args.across_scale
        print(f"  across framing: shift {args.across_shift:+.4f}, "
              f"scale {args.across_scale:.4f}", flush=True)

    # output grid: the median picture, plus the margin
    mz = float(np.median(z_hi - z_lo))
    mc = float(np.median(c_hi - c_lo))
    nZ = int(round(mz * (1 + 2 * args.margin)))
    nC = int(round(mc * (1 + 2 * args.margin)))
    print(f"  output {nZ} x {nC} px (margin {100 * args.margin:.0f} %)", flush=True)
    # normalised output coords, running from -margin to 1+margin of the picture
    tz = (np.arange(nZ) / nZ * (1 + 2 * args.margin) - args.margin)[:, None]
    tc = (np.arange(nC) / nC * (1 + 2 * args.margin) - args.margin)[None, :]

    import imageio.v2 as imageio
    from PIL import Image
    samp = np.asarray(s[::8, ::64], np.float32)
    lo_i, hi_i = np.percentile(samp, [1, 99])

    frames = []
    for k in range(n):
        z = z_lo[k] + tz * (z_hi[k] - z_lo[k])
        c = c_lo[k] + tc * (c_hi[k] - c_lo[k])
        z = np.broadcast_to(z, (nZ, nC))
        c = np.broadcast_to(c, (nZ, nC))
        a = int(np.floor(c.min())) - 2
        b = int(np.ceil(c.max())) + 3
        if a < 0 or b > W:
            continue
        slab = np.asarray(s[:, a:b], np.float32)
        frames.append(_bilinear(slab, z, c, a))
        if k % 40 == 0:
            print(f"    frame {k}/{n}", flush=True)
    print(f"rendered {len(frames)} frames", flush=True)
    if not frames:
        raise SystemExit("no frames produced")

    m = len(frames)
    fps = max(6, int(round(m / args.duration)))
    order = list(range(m - 1, -1, -1)) if args.reverse else list(range(m))
    h0, w0 = frames[0].shape
    fh, fw = (w0, h0) if not args.no_rotate else (h0, w0)
    H = args.height
    Wf = int(round(H * fw / fh)); Wf += Wf % 2

    path = os.path.join(args.out_dir, args.video)
    wr = imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                            macro_block_size=1)
    tiles, tidx = [], set(order[::max(1, m // 16)][:16])
    for i in order:
        fr = np.clip((frames[i] - lo_i) / (hi_i - lo_i + 1e-6), 0, 1)
        if args.invert:
            fr = 1.0 - fr
        if not args.no_rotate:
            fr = np.rot90(fr)
        im = Image.fromarray((fr * 255).astype(np.uint8)).resize((Wf, H))
        wr.append_data(np.asarray(im))
        if i in tidx:
            tiles.append(np.asarray(im.resize((240, 180))))
    wr.close()
    print(f"wrote {path}  ({H}x{Wf}, {m / fps:.1f}s)", flush=True)

    cols = 8
    rows = int(np.ceil(len(tiles) / cols))
    mon = np.zeros((rows * 180, cols * 240), np.uint8)
    for j, t in enumerate(tiles):
        mon[(j // cols) * 180:(j // cols) * 180 + 180,
            (j % cols) * 240:(j % cols) * 240 + 240] = t
    Image.fromarray(mon).save(path + ".frames.png")
    np.savez(os.path.join(args.out_dir, "border_track.npz"),
             z_lo=z_lo, z_hi=z_hi, c_lo=c_lo, c_hi=c_hi, margin=args.margin)
    print("done", flush=True)


if __name__ == "__main__":
    main()
