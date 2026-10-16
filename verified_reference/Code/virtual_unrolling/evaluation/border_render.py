"""Render the film with the PICTURE BORDER as the video's border.

Each frame is cropped to its own detected border rectangle and resampled to one
common output size.  Framing and stabilisation are therefore the same operation:
holding the border still removes the frame-to-frame translation AND the size
"breathing", which the border measurement shows is the dominant motion here (the
two picture edges are anti-correlated at -0.65, i.e. they move in and out about
the centre rather than together).

Sampling is done ONCE, straight from the strip into the output grid, so the
result is not a warp of an already-resampled video.

Across the film each frame is locked to its detected picture border (the
perforation rows lie outside the walked z band, so they cannot serve); along the
film the cuts come from the strip's own global phase lock (as make_film_video).
No ground truth is used.

    python -m virtual_unrolling.evaluation.border_render <strip.npy> --border border.npz \
        --out-dir <dir> --reverse --invert [--along-shift F] [--no-refine-pitch]
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
    ap.add_argument("--along-shift", type=float, default=0.0,
                    help="Translate the along-film cut, in fractions of the "
                         "pitch. The phase lock frames a COMPLETE picture but "
                         "not necessarily the picture the frame lines bound: "
                         "measured against GT's aperture (which is fixed to "
                         "GT's own perforations) this render sat 0.128 pitch "
                         "off, putting the frame line 13 %% up from the bottom "
                         "edge with a sliver of the next frame below it.")
    ap.add_argument("--no-refine-pitch", action="store_true",
                    help="keep the pitch stored in border.npz "
                         "(detect_border --pitch) instead of re-estimating it from the "
                         "strip. Needed on the final full-width strip, where the phase-"
                         "slope estimate lands on 1139 px instead of 1130 px.")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--invert", action="store_true")
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

    # ALONG-film cuts from the strip's global phase lock (as make_film_video): the
    # frame lines themselves could not be detected reliably on this largely static
    # scene. The border supplies the ACROSS-film lock.
    from virtual_unrolling.video.make_film_video import (periodic_component,
                                                refine_pitch_phase)
    col = np.asarray(s).mean(axis=0)
    pc = int(round(pitch))
    pf = float(pitch) if args.no_refine_pitch else refine_pitch_phase(col, pc)
    per = periodic_component(col, pc)
    sig = -per                     # frame-line polarity: troughs
    sc = [sig[np.round(np.arange(o, W - pf, pf)).astype(int)].sum()
          for o in range(pc)]
    o0 = int(np.argmax(sc))
    nb = int((W - o0) / pf)
    bounds = np.round(o0 + np.arange(nb + 1) * pf).astype(float)
    n = min(n, len(bounds) - 1)
    z_lo, z_hi = z_lo[:n], z_hi[:n]
    c_lo, c_hi = bounds[:n], bounds[1:n + 1]
    print(f"  along-film: phase lock, pitch {pf:.2f}, {n} cells", flush=True)
    print(f"  side borders: {nb0}+{nb1} frames filled from neighbours", flush=True)
    print(f"  picture height med {np.median(z_hi - z_lo):.1f} px "
          f"(ptp {np.ptp(z_hi - z_lo):.1f}), "
          f"width med {np.median(c_hi - c_lo):.1f} px "
          f"(ptp {np.ptp(c_hi - c_lo):.1f})", flush=True)

    # FRAMING. Shift the sampling window itself rather than the rendered
    # frame, so the content that moves into view is real film from the strip
    # instead of replicated edge.
    if args.along_shift:
        d = (c_hi - c_lo)
        mid = 0.5 * (c_lo + c_hi) + args.along_shift * d
        c_lo = mid - 0.5 * d
        c_hi = mid + 0.5 * d
        print(f"  along framing: shift {args.along_shift:+.4f} pitch", flush=True)
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
    fh, fw = w0, h0                              # frames are rotated 90 deg
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
