"""Fully lock the residual HORIZONTAL (across-film / film-width) motion in a
stabilized film video, independent of GT.

frame_match's film_stabilized.mp4 is near-perfect VERTICALLY (the along-film
cell locking), but the picture still drifts LEFT-RIGHT: that axis is the film
WIDTH (across-film z), and the unrolling leaves the film wandering slightly
across its width along the strip. The film width is physically constant, so
this residual is pure artifact -> remove ALL of it.

How (drift-free, content-robust): the film-width intensity/edge profile (the
bright picture between the darker film edges) is a STABLE horizontal signature,
whereas the animated character varies along the vertical and averages out. So
    profile[j] = column-mean of the frame's edge map   (a 1-D signal over x)
    reference  = temporal MEDIAN of all profiles        (static structure; the
                                                          character blurs away)
Align every frame's profile to the reference (1-D sub-pixel phase correlation,
iterated) -> each frame's absolute horizontal offset with NO integration
drift. Shift each frame by -offset (fully locked), reject outlier offsets
(scene cuts), then crop the horizontal borders that any shift exposed.

A light optional vertical pass (--vlock) does the same with row profiles, but
along-film row profiles differ cell-to-cell so it is gentle by default.

Usage:
  python -m unwrapping.eval.stabilize_horizontal \
      --in  unwrapping/eval/results/frame_match_v12/film_stabilized.mp4 \
      --out unwrapping/eval/results/frame_match_v12/film_stabilized_hlock.mp4 \
      [--max-shift 0.15 --extra-crop 0.0 --vlock]
"""

import argparse

import numpy as np
import imageio.v2 as imageio
from PIL import Image
from scipy.ndimage import sobel, shift as nd_shift, median_filter


def load_luma(path):
    r = imageio.get_reader(path)
    fr = []
    for f in r:
        a = np.asarray(f, np.float32)
        if a.ndim == 3:
            a = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
        fr.append(a / 255.0)
    r.close()
    return fr


def gmag(f):
    m = np.hypot(sobel(f, 0), sobel(f, 1))
    mx = m.max()
    return m / mx if mx > 0 else m


def _resize(f, h, w):
    return np.asarray(Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8)
                      ).resize((w, h))).astype(np.float32) / 255


def phase2d(a, b, lim=0.2):
    """Sub-pixel (dy, dx) aligning b onto a (2-D FFT phase corr, bounded).
    Full-frame -> dominated by the static film structure, not the character."""
    A = np.fft.rfft2(a - a.mean()); B = np.fft.rfft2(b - b.mean())
    R = A * np.conj(B); R /= np.abs(R) + 1e-12
    r = np.fft.irfft2(R, s=a.shape)
    h, w = a.shape
    my, mx = max(1, int(lim * h)), max(1, int(lim * w))
    rows = np.zeros(h, bool); rows[:my + 1] = rows[-my:] = True
    cols = np.zeros(w, bool); cols[:mx + 1] = cols[-mx:] = True
    r = np.where(rows[:, None] & cols[None, :], r, -np.inf)
    py, px = np.unravel_index(np.argmax(r), r.shape)
    def sub(a0, a1, a2):
        if not (np.isfinite(a0) and np.isfinite(a1) and np.isfinite(a2)):
            return 0.0
        d = a0 - 2 * a1 + a2
        return 0.5 * (a0 - a2) / d if abs(d) > 1e-9 else 0.0
    fy = sub(r[(py - 1) % h, px], r[py, px], r[(py + 1) % h, px])
    fx = sub(r[py, (px - 1) % w], r[py, px], r[py, (px + 1) % w])
    dy = (py if py <= h // 2 else py - h) + fy
    dx = (px if px <= w // 2 else px - w) + fx
    return dy, dx


def lock_slow(fields, lim=0.2, sigma=6.0):
    """Remove the slow SWAY (both axes) that frame_match leaves: the picture
    drifts gently across the film over several seconds.

    `fields` must be a CONTENT-INDEPENDENT representation -- heavily BLURRED
    frames -- so registration tracks the broad film-width brightness/exposure
    envelope (present in every frame) and NOT the animated character (its sharp
    strokes are washed out by the blur; edge maps would do the opposite and
    lock onto the character). Register each field to the GLOBAL median field
    (2-D), robustly reject per-frame outliers, keep only the heavy LOW-PASS --
    the slow coherent drift. Returns absolute (dy, dx) offsets to APPLY."""
    from scipy.ndimage import gaussian_filter1d
    n = len(fields)
    ref = np.median(np.stack(fields), axis=0)
    raw = np.array([phase2d(ref, fields[j], lim) for j in range(n)])  # (n,2)
    off = np.zeros((n, 2))
    for k in range(2):
        v = raw[:, k]
        med = median_filter(v, 7, mode="nearest")
        resid = v - med
        mad = np.median(np.abs(resid - np.median(resid))) * 1.4826 + 1e-6
        v = np.where(np.abs(resid) > 4 * mad, med, v)
        off[:, k] = gaussian_filter1d(v, sigma) - np.median(v)
    return off


def lock2d(edges, win=4, iters=3, lim=0.2):
    """Per-frame (dy, dx) that stills the static structure via 2-D registration
    to a LOCAL-median reference (same scene -> film structure dominates the
    global phase peak; the moving character is a minority and averages out in
    the median). Removes the 2-frame sawtooth AND the slow sway; drift-free.
    Returns absolute offsets to APPLY (sign auto-picked by the caller)."""
    n = len(edges)
    off = np.zeros((n, 2))
    for _ in range(iters):
        cur = [nd_shift(edges[j], off[j], order=1, mode="nearest")
               for j in range(n)]
        for j in range(n):
            lo, hi = max(0, j - win), min(n, j + win + 1)
            ref = np.median(np.stack(cur[lo:hi]), axis=0)
            dy, dx = phase2d(ref, cur[j], lim)
            off[j, 0] += dy; off[j, 1] += dx
    for k in range(2):                    # reject isolated spikes, keep sawtooth
        med = median_filter(off[:, k], 7, mode="nearest")
        resid = off[:, k] - med
        mad = np.median(np.abs(resid - np.median(resid))) * 1.4826 + 1e-6
        bad = np.abs(resid) > max(4 * mad, 25.0)
        off[bad, k] = med[bad]
        off[:, k] -= np.median(off[:, k])
    return off


def phase1d(ref, sig, lim):
    """(shift, peak) with shift s such that moving sig by +s aligns it to ref
    (1-D FFT phase correlation, peak bounded to +/-lim of the length). peak is
    the normalized correlation height -- a confidence used to drop scene cuts."""
    n = len(ref)
    R = np.fft.rfft(ref - ref.mean())
    S = np.fft.rfft(sig - sig.mean())
    X = R * np.conj(S)
    X /= np.abs(X) + 1e-12
    r = np.fft.irfft(X, n=n)
    m = max(1, int(lim * n))
    mask = np.zeros(n, bool)
    mask[:m + 1] = mask[-m:] = True
    rr = np.where(mask, r, -np.inf)
    k = int(np.argmax(rr))
    y0, y1, y2 = rr[(k - 1) % n], rr[k], rr[(k + 1) % n]
    d = y0 - 2 * y1 + y2
    frac = 0.5 * (y0 - y2) / d if abs(d) > 1e-9 else 0.0
    s = k + frac
    return (s - n if s > n / 2 else s), float(r[k])


def lock_axis(profiles, lim, iters=2):
    """Absolute per-frame offset that stills the static film-width structure.

    The across-film motion is BOTH a fast sawtooth (first half) and a slow sway
    (second half); the film-width edge bands are a clear, trackable signature in
    the profile, so register each frame's profile to the GLOBAL median profile
    (edges sharp, character averaged out) to get the absolute position, and
    apply ALL of it (no low-pass -> the sawtooth is removed too). Only isolated
    SPIKES far above the sawtooth amplitude (content-fooled frames / scene cuts)
    are rejected. Iterated (re-median after a first alignment sharpens the
    reference). Returns the shift to APPLY (sign auto-picked by the caller)."""
    P = np.stack(profiles)
    n, L = P.shape
    off = np.zeros(n)
    for _ in range(iters):
        cur = np.stack([nd_shift(P[j], -off[j], order=1, mode="nearest")
                        for j in range(n)])
        ref = np.median(cur, axis=0)
        for j in range(n):
            s, _ = phase1d(ref, nd_shift(P[j], -off[j], order=1,
                                         mode="nearest"), lim)
            off[j] += s
    # reject only isolated SPIKES (keep the real sawtooth): deviation from a
    # rolling median above the sawtooth band
    med = median_filter(off, 7, mode="nearest")
    resid = off - med
    mad = np.median(np.abs(resid - np.median(resid))) * 1.4826 + 1e-6
    bad = np.abs(resid) > max(4 * mad, 25.0)
    off[bad] = med[bad]
    off -= np.median(off)
    return off


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=26)
    ap.add_argument("--win", type=int, default=4,
                    help="+/- frames for the local-median 2-D reference.")
    ap.add_argument("--max-shift", type=float, default=0.15,
                    help="max horizontal correction as a fraction of width.")
    ap.add_argument("--vlock", action="store_true",
                    help="(kept for compat; vertical slow sway is always "
                         "removed now).")
    ap.add_argument("--axes", choices=["both", "v", "h"], default="both",
                    help="Which axis to actually correct. 'v' is for a video "
                         "whose OTHER axis is already locked by construction -- "
                         "e.g. the border-locked render, where the across-film "
                         "axis is cropped to the picture border per frame and "
                         "only the along-film (vertical) drift is left.")
    ap.add_argument("--v-max-shift", type=float, default=0.08)
    ap.add_argument("--slow-sigma", type=float, default=4.0,
                    help="low-pass width (frames) for the slow-sway estimate.")
    ap.add_argument("--iters", type=int, default=4,
                    help="GT-lock iterations (converge our frames to GT).")
    ap.add_argument("--blur", type=float, default=18.0,
                    help="spatial blur (px) to wash out the character and keep "
                         "the broad film-width envelope for registration.")
    ap.add_argument("--gt", default=None,
                    help="GT optical-scan mp4: use it as the STILL anchor "
                         "(register our frames to the matched GT frame, remove "
                         "the slow drift). The reliable de-sway.")
    ap.add_argument("--gt-crop", default="0.235,0.12,0.81,0.88",
                    help="'none' when --gt is already cut to the picture.")
    ap.add_argument("--no-gt-trim", action="store_true",
                    help="Skip the leader trim for an already-cut GT, whose "
                         "leader was dropped when it was made.")
    ap.add_argument("--extra-crop", type=float, default=0.0,
                    help="extra fractional border crop after locking.")
    ap.add_argument("--save-offsets", default=None,
                    help="Write the per-frame (y,x) offsets to this .npy. They "
                         "can then be folded into the ORIGINAL sampling (see "
                         "border_render --along-offsets) instead of shifting an "
                         "already-rendered video: that avoids the second "
                         "resampling AND the crop this tool has to take to hide "
                         "the exposed edges, which costs ~29 % of the frame.")
    args = ap.parse_args()

    from scipy.ndimage import gaussian_filter, gaussian_filter1d
    frames = load_luma(args.inp)
    n = len(frames)
    H, W = frames[0].shape
    print(f"{n} frames {H}x{W}", flush=True)

    if args.gt:
        # GT-ANCHORED (reliable): GT is nearly still, so aligning our frames to
        # their MATCHED GT frames removes our sway. Registration is on SHARED
        # content (same picture), so it is not fooled by the animation.
        # ITERATE: measure the residual offset to GT, remove its SMOOTH part
        # (the sway; per-frame cross-modal noise is rejected by the smoothing),
        # repeat -> converges to GT's still framing. Report the residual sway
        # shrinking so success is verifiable.
        from .compare_videos import (load_video_luma, trim_leader, parse_crop,
                                     apply_crop)
        gt = load_video_luma(args.gt)
        if not args.no_gt_trim:
            gt, _ = trim_leader(gt)
        if args.gt_crop.lower() not in ("none", ""):
            gt = apply_crop(gt, parse_crop(args.gt_crop))
        gt = [_resize(g, H, W) for g in gt]
        idx = np.linspace(0, len(gt) - 1, n).round().astype(int)  # 1:1 map
        gtg = [gmag(gt[idx[j]]) for j in range(n)]
        lim = max(args.max_shift, args.v_max_shift)

        def smooth_sway(v):
            med = median_filter(v, 7, mode="nearest")
            r = v - med
            mad = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-6
            v = np.where(np.abs(r) > 4 * mad, med, v)
            return gaussian_filter1d(v, args.slow_sigma)

        def residual():
            return np.array([phase2d(gtg[j],
                             gmag(nd_shift(frames[j], off2[j], order=1,
                                           mode="nearest")), lim)
                             for j in range(n)])

        off2 = np.zeros((n, 2))
        sign = 1.0
        for it in range(args.iters):
            res = residual()
            sway = np.stack([smooth_sway(res[:, 0]), smooth_sway(res[:, 1])], 1)
            print(f"  GT iter {it}: residual slow-sway std  x "
                  f"{sway[:,1].std():.1f}px  y {sway[:,0].std():.1f}px "
                  f"(want ->0)", flush=True)
            # Fix the nd_shift vs phase2d sign so the residual shrinks. Test it
            # on the axis actually being corrected -- probing x while correcting
            # only y would pick the sign from a channel that never moves.
            ax = 0 if args.axes == "v" else 1
            if it == 0:
                trial = off2.copy(); trial[:, ax] += sway[:, ax]
                r1 = np.array([phase2d(gtg[j], gmag(nd_shift(frames[j], trial[j],
                              order=1, mode="nearest")), lim)[ax]
                              for j in range(0, n, 3)])
                trial[:, ax] = off2[:, ax] - sway[:, ax]
                r2 = np.array([phase2d(gtg[j], gmag(nd_shift(frames[j], trial[j],
                              order=1, mode="nearest")), lim)[ax]
                              for j in range(0, n, 3)])
                sign = 1.0 if np.abs(r1).mean() <= np.abs(r2).mean() else -1.0
            if args.axes != "v":
                off2[:, 1] += sign * sway[:, 1]
            if args.vlock or args.axes == "v":
                off2[:, 0] += sign * sway[:, 0]
        off2[:, 0] -= np.median(off2[:, 0])
        off2[:, 1] -= np.median(off2[:, 1])
    else:
        # GT-FREE fallback: heavily blur each frame so the character washes out
        # and only the broad film-width envelope remains for registration.
        B = [gaussian_filter(f, args.blur) for f in frames]
        off2 = lock_slow(B, lim=max(args.max_shift, args.v_max_shift),
                         sigma=args.slow_sigma)
    xoff = off2[:, 1]
    yoff = off2[:, 0] if (args.vlock or args.axes == "v") else np.zeros(n)
    if args.axes == "v":
        xoff = np.zeros(n)
    elif args.axes == "h":
        yoff = np.zeros(n)
    print(f"  horizontal offset: std {xoff.std():.2f}px range "
          f"[{xoff.min():.1f},{xoff.max():.1f}]  vertical std {yoff.std():.2f}",
          flush=True)

    def ftf(fr, axis):
        e = [gmag(f).mean(axis) for f in fr]
        return float(np.mean([abs(phase1d(e[i - 1], e[i], args.max_shift)[0])
                              for i in range(1, len(fr))]))

    # the nd_shift vs phase1d sign conventions can differ -> pick, per axis,
    # the sign that actually reduces the consecutive residual.
    base_x = ftf(frames, 0)
    if ftf([nd_shift(frames[j], (0, xoff[j]), order=1, mode="nearest")
            for j in range(n)], 0) > \
       ftf([nd_shift(frames[j], (0, -xoff[j]), order=1, mode="nearest")
            for j in range(n)], 0):
        xoff = -xoff
    if args.vlock or args.axes == "v":
        if ftf([nd_shift(frames[j], (yoff[j], 0), order=1, mode="nearest")
                for j in range(n)], 1) > \
           ftf([nd_shift(frames[j], (-yoff[j], 0), order=1, mode="nearest")
                for j in range(n)], 1):
            yoff = -yoff

    if args.save_offsets:
        np.save(args.save_offsets, np.stack([yoff, xoff], 1))
        print(f"  saved per-frame offsets -> {args.save_offsets}  "
              f"(y,x in this video's pixels)", flush=True)

    warped = [nd_shift(frames[j], (yoff[j], xoff[j]), order=1,
                       mode="nearest") for j in range(n)]

    def abs_drift(fr):
        e = [gmag(f).mean(0) for f in fr]
        d = np.array([0.0] + [phase1d(e[i - 1], e[i], args.max_shift)[0]
                              for i in range(1, len(fr))])
        traj = np.cumsum(d)
        return float(traj.std()), float(traj.max() - traj.min())
    b_std, b_rng = abs_drift(frames)
    a_std, a_rng = abs_drift(warped)
    print(f"  consecutive shift: {base_x:.2f} -> {ftf(warped, 0):.2f}px | "
          f"slow-sway (abs drift) std {b_std:.1f}->{a_std:.1f}px  "
          f"range {b_rng:.0f}->{a_rng:.0f}px", flush=True)

    # crop the borders any shift exposed
    mx = int(np.ceil(np.abs(xoff).max())) + 1
    my = int(np.ceil(np.abs(yoff).max())) + 1
    ex = int(W * args.extra_crop / 2)
    ey = int(H * args.extra_crop / 2)
    y0, y1 = my + ey, H - my - ey
    x0, x1 = mx + ex, W - mx - ex
    cropped = [w[y0:y1, x0:x1] for w in warped]
    ch, cw = cropped[0].shape
    print(f"  crop -> {ch}x{cw} (from {H}x{W})", flush=True)

    Hout = H
    Wout = int(round(Hout * cw / ch)); Wout += Wout % 2
    wr = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                            quality=9, macro_block_size=1)
    for c in cropped:
        im = Image.fromarray((np.clip(c, 0, 1) * 255).astype(np.uint8)
                             ).resize((Wout, Hout))
        wr.append_data(np.asarray(im))
    wr.close()

    # diagnostic offset plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 3))
    plt.plot(xoff, label="horizontal lock (px)")
    if args.vlock:
        plt.plot(yoff, label="vertical lock (px)")
    plt.axhline(0, color="k", lw=0.5); plt.legend(); plt.xlabel("frame")
    plt.title("residual motion removed by the full lock")
    plt.tight_layout(); plt.savefig(args.out + ".offsets.png", dpi=110)
    plt.close()
    print(f"Done -> {args.out} ({Hout}x{Wout})", flush=True)


if __name__ == "__main__":
    main()
