"""Detect the PICTURE BORDER rectangle of every frame in the unrolled strip.

The border is the rectangle of unexposed film around each picture: the two frame
lines running across the film (separating one picture from the next) and the two
picture edges running along it.  In the strip it shows as a thin bright outline
-- see `frame_full` in the border probe -- and it is content-independent, so it
is a fiducial.

WHY THIS AND NOT THE PERFORATIONS.  The sprocket route failed for a reason that
does not apply here: the perf rows sit in z bands the walk never traced, so they
are rendered from extrapolated geometry and each row reports its own error (the
two rows' measured motion correlates at -0.07 across the film and -0.23 along
it, when they are punched into opposite edges of the same film and must agree).
The picture border lies INSIDE the walked picture band, so it carries the same
geometry as the content it is supposed to register.

WHAT IT GIVES.  Four numbers per frame -- the two frame lines and the two side
borders -- i.e. position and size in both axes.  Cropping every frame to its own
border and resizing to a common size removes translation AND the scale
"breathing" in one step, which is why framing and stabilisation are the same
operation here.

NOT to be confused with the frame-line handling in `perf_rectify.py`: that was a
1-D brightest-peak-in-a-window search used only to nudge the along-film cut, and
it was shown to be noise-dominated (the same frame line measured at the top and
bottom of the picture band correlated at -0.05 with itself).  This module fits
each border with a matched filter against a template learned from the reel.

    python -m unwrapping.eval.detect_border <strip.npy> --out-dir <dir>
"""

import argparse
import os

import numpy as np
from scipy.signal import find_peaks


def estimate_pitch(col, lo=250, hi=1600):
    d = np.asarray(col, np.float64)
    k = 2001 | 1
    sm = np.convolve(np.pad(d, k // 2, mode="edge"), np.ones(k) / k, "valid")[:len(d)]
    d = d - sm
    d -= d.mean()
    ac = np.correlate(d, d, "full")[len(d) - 1:]
    hi = min(hi, len(ac) - 1)
    p = lo + int(np.argmax(ac[lo:hi]))
    for k in (3, 2):
        if k * p <= hi and ac[k * p] > 0.7 * ac[p]:
            return k * p
    return p


def _bandpass(x, pitch, frac=0.45):
    x = np.asarray(x, np.float64) - np.mean(x)
    F = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x))
    f0 = 1.0 / pitch
    F[np.abs(f - f0) > frac * f0] = 0.0
    return np.fft.irfft(F, n=len(x))


def _parab(a, b, c):
    d = a - 2 * b + c
    return float(np.clip(0.5 * (a - c) / d, -1, 1)) if abs(d) > 1e-9 else 0.0


def _refine(prof, guess, tmpl, half):
    """Sub-pixel position of `tmpl` in `prof` near `guess`, by cross-correlation."""
    n = len(prof)
    a, b = int(guess) - half, int(guess) + half
    if a < 1 or b > n - 1 or b - a < len(tmpl) + 4:
        return np.nan, 0.0
    seg = prof[a:b].astype(np.float64)
    seg = seg - seg.mean()
    t = tmpl - tmpl.mean()
    cc = np.correlate(seg, t, "valid")
    if len(cc) < 3:
        return np.nan, 0.0
    j = int(np.argmax(cc))
    if not (1 <= j < len(cc) - 1):
        return np.nan, float(cc[j])
    sub = j + _parab(cc[j - 1], cc[j], cc[j + 1])
    denom = np.linalg.norm(seg) * np.linalg.norm(t) + 1e-9
    return a + sub + 0.5 * (len(tmpl) - 1), float(cc[j] / denom)


def _learn_template(prof, positions, half):
    """Median of the profile in windows centred on `positions` -- the border's
    own shape, learned from the reel instead of assumed."""
    win = []
    for p in positions:
        a, b = int(round(p)) - half, int(round(p)) + half
        if a < 0 or b > len(prof):
            continue
        w = prof[a:b].astype(np.float64)
        win.append(w - w.mean())
    if len(win) < 5:
        return None
    return np.median(np.stack(win), axis=0)


def index_and_unslip(x, pitch, verbose=True):
    """Give each detected line an integer frame index, then de-slip its RESIDUAL.

    Two distinct things have to be kept apart here, and conflating them is what
    made a first version collapse 230 detections into 129 mangled ones:
      * a MISSING detection is a genuine gap -- the index must step by 2 and the
        surviving positions must be left alone;
      * a CYCLE SLIP is a spurious jump in the residual about the k*pitch ramp,
        and only that residual may be corrected.
    Correcting the raw spacing instead forces every gap to one pitch and drags
    every later frame with it.
    """
    x = np.asarray(x, float)
    k = np.zeros(len(x), int)
    for i in range(1, len(x)):
        k[i] = k[i - 1] + max(1, int(round((x[i] - x[i - 1]) / pitch)))
    resid = x - k * pitch
    d = np.diff(resid)
    n = np.round(d / pitch)
    if verbose and np.any(n != 0):
        print(f"    de-slipped {int(np.abs(n).sum())} residual jumps")
    d = d - pitch * n
    resid = np.concatenate([[resid[0]], resid[0] + np.cumsum(d)])
    return resid + k * pitch, k


# ISO 16 mm: picture aperture 10.26 mm across the film, frame pitch 7.62 mm
# along it.  Used only as this RATIO against the measured pitch, so no absolute
# pixel calibration is assumed anywhere.
ISO_ASPECT = 10.26 / 7.62           # 1.3465


def picture_band(strip, pitch, col_step=32, verbose=True):
    """z range of the picture band, from the two border lines themselves.

    Look for the PAIR of bright lines in the reel-median row profile whose
    separation is closest to the expected aperture height, ISO_ASPECT * pitch.
    Being told what separation to expect is what makes this robust; the two
    thresholding schemes tried before both failed on content:
      * energy > threshold selected the entire strip (the perf bands carry MORE
        gradient than the picture, not less);
      * energy < threshold fragmented the picture band wherever a busy scene
        raised the energy, returning 400-800 row sub-ranges.
    """
    Z = strip.shape[0]
    p = np.asarray(strip[:, ::col_step], np.float32).mean(axis=1).astype(np.float64)
    k = 9
    ps = np.convolve(np.pad(p, k // 2, mode="edge"), np.ones(k) / k, "valid")[:Z]
    want = ISO_ASPECT * pitch
    # the border is a LINE, so score by prominence above the local surround
    surround = np.convolve(np.pad(ps, 60, mode="edge"), np.ones(121) / 121,
                           "valid")[:Z]
    r = ps - surround
    pk, _ = find_peaks(r, distance=40)
    if len(pk) < 2:
        raise SystemExit("no border-line candidates in the row profile")
    best, score = None, -np.inf
    for i in range(len(pk)):
        for j in range(i + 1, len(pk)):
            sep = pk[j] - pk[i]
            if abs(sep - want) > 0.12 * want:
                continue
            sc = r[pk[i]] + r[pk[j]] - 3.0 * abs(sep - want) / want
            if sc > score:
                score, best = sc, (int(pk[i]), int(pk[j]))
    if best is None:
        raise SystemExit(f"no line pair separated by ~{want:.0f} px "
                         f"(candidates at {pk[:20]})")
    if verbose:
        print(f"  picture band z {best[0]}-{best[1]}  height {best[1] - best[0]} px "
              f"(expected {want:.0f} from the pitch)")
    return best


def find_frame_lines(strip, pitch, z_pic, half=70, verbose=True):
    """Along-film borders: the frame line at each frame boundary."""
    z0, z1 = z_pic
    prof = np.asarray(strip[z0:z1:4, :], np.float32).mean(axis=0).astype(np.float64)
    W = len(prof)
    band = _bandpass(prof, pitch)
    pk, _ = find_peaks(band, distance=int(round(pitch * 0.7)))
    if verbose:
        print(f"  frame lines: {len(pk)} candidates (expect ~{int(W / pitch)})")
    tmpl = _learn_template(prof, pk, half)
    if tmpl is None:
        raise SystemExit("could not learn a frame-line template")
    # Search +-quarter of a pitch.  A narrow window made `_refine` return NaN
    # whenever the true line sat more than ~40 px from the band-pass estimate,
    # which threw away 100 of 230 lines on the full strip.
    search = int(0.25 * pitch) + half
    pos, score = [], []
    for p in pk:
        q, s = _refine(prof, p, tmpl, search)
        pos.append(q); score.append(s)
    pos = np.array(pos); score = np.array(score)
    ok = np.isfinite(pos)
    pos, score = pos[ok], score[ok]
    o = np.argsort(pos)
    return pos[o], score[o], prof, tmpl


def refine_frame_lines(prof, lines, k, pitch, half=70, n_local=20,
                       search_frac=0.15, verbose=True):
    """Second pass: re-fit every frame line against a LOCAL template, searching
    only near where the reel's own trend says the line should be.

    Two things go wrong in a single global pass, and both showed up on the full
    strip while being invisible on short test slabs:
      * one template pooled over the whole reel fits badly, because the frame
        line's profile changes from the inner to the outer end -- the match
        score fell from 0.74-0.79 on a slab to 0.23 over the reel;
      * with a wide search the fit then locks onto the wrong feature within the
        period, leaving 17 % of lines about a third of a pitch out, which is
        what put the bottom of one picture and the top of the next in the same
        frame.
    Predicting from a robust fit and searching +-15 % of a pitch removes the
    ambiguity, and every frame SLOT is fitted, so slots missed in pass 1 are
    recovered rather than left as gaps.
    """
    lines = np.asarray(lines, float)
    k = np.asarray(k, int)
    resid = lines - k * pitch
    # GLOBAL robust polynomial trend, not a local median.  A median filter
    # follows a phase flip instead of rejecting it: where a run of lines locks
    # onto the wrong feature within the period, the local trend simply moves
    # with them and pass 2 confirms the error.  That left part of the reel cut
    # half a frame out -- the lower half of one picture above the upper half of
    # the next -- and the frame width varying by 748 px peak to peak.  A global
    # low-order fit cannot bend to a local flip, so those lines fall outside the
    # search window and get pulled back onto the true phase.
    kk = np.arange(k.min(), k.max() + 1)
    w = np.ones(len(resid), bool)
    for _ in range(6):
        if w.sum() < 8:
            w = np.ones(len(resid), bool)
            break
        cf = np.polyfit(k[w], resid[w], 3)
        r = resid - np.polyval(cf, k)
        s = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-9
        w_new = np.abs(r) < 3.0 * s
        if (w_new == w).all():
            break
        w = w_new
    cf = np.polyfit(k[w], resid[w], 3)
    if verbose:
        print(f"  global phase fit: {w.sum()}/{len(resid)} lines on-trend, "
              f"residual spread {np.std(resid[w]):.1f} px, "
              f"{np.sum(~w)} rejected as phase-flipped or mis-fit")
    pred = kk * pitch + np.polyval(cf, kk)

    search = int(search_frac * pitch)
    out, score = np.full(len(kk), np.nan), np.zeros(len(kk))
    for i, (ki, pi) in enumerate(zip(kk, pred)):
        # template from the nearest ON-TREND lines only, so a phase-flipped run
        # cannot poison the template it would then be matched against
        near = np.argsort(np.abs(k[w] - ki))[:n_local]
        tmpl = _learn_template(prof, lines[w][near], half)
        if tmpl is None:
            continue
        q, s = _refine(prof, pi, tmpl, search + half)
        if np.isfinite(q) and abs(q - pi) < search:
            out[i], score[i] = q, s
        else:
            out[i], score[i] = pi, 0.0          # fall back to the prediction
    if verbose:
        n_fit = int((score > 0).sum())
        print(f"  refine pass 2: {n_fit}/{len(kk)} slots fitted "
              f"({len(kk) - n_fit} fell back to the trend), score med "
              f"{np.median(score[score > 0]):.2f}, spacing med "
              f"{np.median(np.diff(out)):.1f} px")
    return out, kk, score


def find_side_borders_2pass(strip, pitch, lines, z_hint, half=80, inset=0.18,
                            verbose=True):
    """Two passes: a global guess, then a per-frame one from the reel's trend.

    One fixed guess for the whole reel fails wherever the border has drifted
    past the search window, which left 18 % of the edges unmeasured and filled
    from neighbours -- and a filled edge means the crop does not follow that
    frame's own border, which is exactly what stops the border sitting still in
    the output.
    """
    from scipy.ndimage import median_filter
    out = find_side_borders(strip, pitch, lines, z_hint, half, inset,
                            verbose=False)
    pred = np.empty((len(out), 2))
    for j in (0, 1):
        y = out[:, j].copy()
        ok = np.isfinite(y)
        if ok.sum() < 10:
            pred[:, j] = z_hint[j]
            continue
        g = np.arange(len(y))
        y = np.interp(g, g[ok], y[ok])
        pred[:, j] = median_filter(y, 9, mode="nearest")
    out2 = find_side_borders(strip, pitch, lines, z_hint, half, inset,
                             verbose=False, guess=pred)
    # keep pass 2 where it succeeded, else fall back to pass 1
    better = np.isfinite(out2[:, 0]) & np.isfinite(out2[:, 1])
    res = np.where(better[:, None], out2, out)
    if verbose:
        n1 = int((np.isfinite(out[:, 0]) & np.isfinite(out[:, 1])).sum())
        n2 = int((np.isfinite(res[:, 0]) & np.isfinite(res[:, 1])).sum())
        h = res[:, 1] - res[:, 0]
        print(f"  side borders: pass1 {n1}/{len(out)} -> pass2 {n2}/{len(out)}; "
              f"height med {np.nanmedian(h):.1f} px", flush=True)
    return res


def find_side_borders(strip, pitch, lines, z_hint, half=80, inset=0.18,
                      verbose=True, guess=None):
    """Across-film borders: the two picture edges, one pair per frame.

    Measured on each frame's OWN columns (inset from the frame lines so the
    interframe region cannot contribute), against a template learned from the
    reel median, so the whole edge shape is used rather than one threshold
    crossing.
    """
    Z, W = strip.shape
    zg_lo, zg_hi = z_hint
    # first pass: the reel-median row profile, to learn the two edge templates
    mid = len(lines) // 2
    a = int(lines[mid] + inset * pitch); b = int(lines[mid + 1] - inset * pitch)
    ref = np.asarray(strip[:, a:b], np.float32).mean(axis=1).astype(np.float64)
    t_lo = _learn_template(ref, [zg_lo], half)
    t_hi = _learn_template(ref, [zg_hi], half)
    if t_lo is None or t_hi is None:
        # a single window is enough here: the template IS that window
        t_lo = ref[int(zg_lo) - half:int(zg_lo) + half] - \
            ref[int(zg_lo) - half:int(zg_lo) + half].mean()
        t_hi = ref[int(zg_hi) - half:int(zg_hi) + half] - \
            ref[int(zg_hi) - half:int(zg_hi) + half].mean()

    out = np.full((len(lines) - 1, 4), np.nan)      # z_lo, z_hi, score_lo, score_hi
    for i in range(len(lines) - 1):
        a = int(lines[i] + inset * pitch)
        b = int(lines[i + 1] - inset * pitch)
        if b - a < 50 or a < 0 or b > W:
            continue
        p = np.asarray(strip[:, a:b], np.float32).mean(axis=1).astype(np.float64)
        g_lo = guess[i, 0] if guess is not None else zg_lo
        g_hi = guess[i, 1] if guess is not None else zg_hi
        if not (np.isfinite(g_lo) and np.isfinite(g_hi)):
            continue
        zl, sl = _refine(p, g_lo, t_lo, half + 50)
        zh, sh = _refine(p, g_hi, t_hi, half + 50)
        out[i] = (zl, zh, sl, sh)
    if verbose:
        good = np.isfinite(out[:, 0]) & np.isfinite(out[:, 1])
        print(f"  side borders: {good.sum()}/{len(out)} frames measured; "
              f"height med {np.nanmedian(out[:, 1] - out[:, 0]):.1f} px")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strip")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pitch", type=float, default=0)
    ap.add_argument("--z-picture", type=int, nargs=2, default=None,
                    help="Approximate z range of the picture band (default: "
                         "auto from the strip's content energy).")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    s = np.load(args.strip, mmap_mode="r")
    Z, W = s.shape
    print(f"strip {s.shape}", flush=True)

    col = np.asarray(s[::4, :], np.float32).mean(axis=0)
    pitch = float(args.pitch) if args.pitch > 0 else float(estimate_pitch(col))
    print(f"pitch {pitch:.2f} px -> ~{W / pitch:.1f} frames", flush=True)

    if args.z_picture:
        zg_lo, zg_hi = args.z_picture
        print(f"picture band z {zg_lo}-{zg_hi} (given)", flush=True)
    else:
        zg_lo, zg_hi = picture_band(s, pitch)

    z_pic = (zg_lo + 60, zg_hi - 60)
    lines, lscore, prof_c, tmpl_c = find_frame_lines(s, pitch, z_pic)
    lines, kline = index_and_unslip(lines, pitch)
    print(f"  pass 1: {len(lines)} lines over {kline[-1] - kline[0] + 1} frame "
          f"slots, spacing med {np.median(np.diff(lines)):.1f} px", flush=True)
    lines, kline, lscore = refine_frame_lines(prof_c, lines, kline, pitch)

    sides = find_side_borders_2pass(s, pitch, lines, (zg_lo, zg_hi))

    np.savez(os.path.join(args.out_dir, "border.npz"),
             pitch=pitch, lines=lines, line_score=lscore, sides=sides,
             z_guess=np.array([zg_lo, zg_hi]), strip_shape=np.array([Z, W]),
             prof_c=prof_c, tmpl_c=tmpl_c)
    print(f"wrote {args.out_dir}/border.npz", flush=True)
    _diag(s, pitch, lines, sides, prof_c, args.out_dir)


def _diag(s, pitch, lines, sides, prof_c, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Z, W = s.shape

    fig, ax = plt.subplots(3, 1, figsize=(15, 10))
    ax[0].plot(np.diff(lines), ".-", ms=3, lw=.7)
    ax[0].axhline(pitch, color="k", ls="--", lw=1)
    ax[0].set_ylim(pitch * 0.9, pitch * 1.1)
    ax[0].set_title("frame-line spacing (px) = the along-film size of each frame")
    ax[0].grid(alpha=.3)

    ax[1].plot(sides[:, 0], lw=.8, label="low side border")
    ax[1].plot(sides[:, 1], lw=.8, label="high side border")
    ax[1].set_title("side borders, across the film (px) = position + size")
    ax[1].legend(); ax[1].grid(alpha=.3)

    ax[2].plot(sides[:, 1] - sides[:, 0], lw=.8)
    ax[2].set_title("picture HEIGHT across the film (px) -- constant on real "
                    "film, so the variation is our distortion")
    ax[2].grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "border_diag.png"), dpi=110)
    plt.close(fig)

    # check sheet: the fitted rectangle drawn on six frames spread over the reel
    n = len(lines) - 1
    idx = np.unique(np.linspace(0, n - 1, 6).astype(int))
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    for ax, i in zip(np.ravel(axs), idx):
        a, b = int(lines[i]), int(lines[i + 1])
        if a < 0 or b > W:
            continue
        crop = np.asarray(s[:, a:b], np.float32)
        lo, hi = np.percentile(crop, [1, 99])
        ax.imshow(np.rot90(1 - np.clip((crop - lo) / (hi - lo + 1e-9), 0, 1)),
                  cmap="gray", aspect="auto", extent=[0, Z, b - a, 0])
        zl, zh = sides[i, 0], sides[i, 1]
        for v in (zl, zh):
            if np.isfinite(v):
                ax.axvline(v, color="lime", lw=1.4)
        ax.axhline(0, color="red", lw=1.4)
        ax.axhline(b - a, color="red", lw=1.4)
        ax.set_title(f"frame {i}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("fitted border: red = frame lines (along the film), "
                 "green = side borders (across it)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "border_check.png"), dpi=110)
    plt.close(fig)
    print(f"wrote {out_dir}/border_diag.png + border_check.png", flush=True)


if __name__ == "__main__":
    main()
