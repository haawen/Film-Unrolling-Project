"""GT-driven frame matching: every GT optical-scan frame <-> a CONTINUOUSLY
positioned window re-cut from the unrolled whole-roll strip.

Unlike compare_videos.py (pred video cells -> GT frames), this matches in the
other direction and does NOT quantize the unroll to the video's fixed
phase-locked cells: for each GT frame the strip position is searched at ~3 px
resolution, so residual pitch drift / phase error cannot cost framing accuracy.

Pipeline:
  1. Strip: normalize (1-99 pct) + invert (CT film is a negative), crop the
     z shading margins, replicate make_film_video's global phase-lock to get
     the cell grid (init + reporting only).
  2. GT: luma, leader-trim, crop to picture area (drops sprockets/edge print).
  3. Init: linear map GT index -> reversed cell centers (play dir = --reverse).
  4. Per-frame search: coarse-to-fine over strip x; score = gradient
     correlation after gradient-domain phase-correlation registration
     (cross-modal robust, same recipe the eval pipeline validated).
  5. Robust linear re-fit of x(gt) + re-search of outliers + monotonicity fix.
  6. Output: pairs/pair_*.png (GT | ours | overlay | diff), matches.json,
     match curve/score plots; ends trimmed (bad fits at the innermost/
     outermost windings) go to pairs_trimmed/. Optional --montage / --video.

Usage:
  python -m unwrapping.eval.frame_match \
      --strip unwrapping/inr/results/walk_dense_v9/wholeroll.npy \
      --gt data/h265_1080p.mp4 \
      --out-dir unwrapping/eval/results/frame_match_v9 \
      [--trim-start 4 --trim-end 2] [--montage] [--video]
"""

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import zoom

from . import metrics as M
from .compare_videos import (load_video_luma, trim_leader, parse_crop,
                             apply_crop, grad_mag, histogram_match)
from unwrapping.inr.make_film_video import (estimate_pitch, refine_pitch_phase,
                                            periodic_component)


# --------------------------------------------------------------------------- #
# strip handling
# --------------------------------------------------------------------------- #
def load_strip(path):
    s = np.load(path, mmap_mode="r")
    samp = np.asarray(s[:, ::50], np.float32)
    lo, hi = np.percentile(samp, [1, 99])
    return s, float(lo), float(hi)


def phase_lock_bounds(col, pitch_override=0.0):
    """Replicate make_film_video --phase-lock global: fractional pitch via the
    phase-slope fit + the single best boundary phase. Returns (bounds, pitch)."""
    pitch_c = int(round(pitch_override)) if pitch_override > 0 else estimate_pitch(col)
    pitch_f = float(pitch_override) if pitch_override > 0 else refine_pitch_phase(col, pitch_c)
    W = len(col)
    p = periodic_component(col, pitch_c)
    score = [p[np.round(np.arange(o, W - pitch_f, pitch_f)).astype(int)].sum()
             for o in range(pitch_c)]
    o0 = int(np.argmax(score))
    n = int((W - o0) / pitch_f)
    bounds = o0 + np.arange(n + 1) * pitch_f
    return bounds, pitch_f


class StripWindows:
    """Cut GT-oriented pred frames from the strip at arbitrary x.

    Full-res path (render): window = strip[z0:z1, x-w/2:x+w/2] -> normalize ->
    invert -> rot90(k=1) -> fliplr  (the validated video orientation vs GT).
    Match path: the same, precomputed once at match scale so a candidate
    window is a pure array slice (fast search).
    """

    def __init__(self, strip, lo, hi, zcrop, match_h, match_w, fliplr=True):
        self.s, self.lo, self.hi = strip, lo, hi
        Z, W = strip.shape
        self.z0, self.z1 = int(zcrop[0] * Z), int(zcrop[1] * Z)
        self.W = W
        self.match_h, self.match_w = match_h, match_w
        self.fliplr = fliplr
        self.fy = self.fx = 1.0  # calibrated window extents (see calibrate_scale)
        self.sa = None  # arc scale, set by build_small (needs pitch)

    def _z_extent(self, fx=None):
        fx = self.fx if fx is None else fx
        Z = self.s.shape[0]
        zc = (self.z0 + self.z1) / 2.0
        zh = (self.z1 - self.z0) / 2.0 * fx
        return max(0, int(round(zc - zh))), min(Z, int(round(zc + zh)))

    def build_small(self, pitch, fy=1.0, fx=1.0):
        """Pre-shrink the normalized inverted strip so that one (scaled) pitch
        of arc -> match_h px and the (scaled) z extent -> match_w px."""
        self.pitch, self.fy, self.fx = float(pitch), fy, fx
        self.sa = self.match_h / (self.pitch * fy)
        za, zb = self._z_extent()
        sz = self.match_w / (zb - za)
        n = np.asarray(self.s[za:zb], np.float32)
        n = 1.0 - np.clip((n - self.lo) / (self.hi - self.lo + 1e-6), 0, 1)
        print(f"  shrinking strip {n.shape} by (z {sz:.4f}, arc {self.sa:.4f})...",
              flush=True)
        self.small = zoom(n, (sz, self.sa), order=1, prefilter=False)
        print(f"  -> small strip {self.small.shape}", flush=True)

    def window_small(self, x):
        """Match-scale GT-oriented frame centered at full-res arc position x."""
        c = int(round(x * self.sa)) - self.match_h // 2
        c = max(0, min(self.small.shape[1] - self.match_h, c))
        f = np.rot90(self.small[:, c:c + self.match_h], 1)
        if self.fliplr:
            f = f[:, ::-1]
        return np.ascontiguousarray(f)

    def window_full(self, x, height, width=None, fy=None, fx=None, zshift=0.0):
        """Full-res GT-oriented frame centered at x with the calibrated (or
        given) extents, resized to height (x width if given). zshift moves the
        z-crop window (strip px) — used by the stabilized video export."""
        fy = self.fy if fy is None else fy
        wa = max(2, int(round(self.pitch * fy)))
        a = max(0, min(self.W - wa, int(round(x - wa / 2))))
        za, zb = self._z_extent(fx)
        # FRACTIONAL z-shift (sub-pixel across-film): integer part via slicing,
        # fractional part via linear blend of the two adjacent z-crops — so a
        # smoothed z trajectory renders smoothly instead of rounding to a
        # visible staircase (the residual horizontal jitter).
        Zn = self.s.shape[0]
        zf = max(float(-za), min(float(Zn - zb), float(zshift)))
        z0 = int(np.floor(zf))
        frac = zf - z0
        za0, zb0 = za + z0, zb + z0
        f = np.asarray(self.s[za0:zb0, a:a + wa], np.float32)
        if frac > 1e-3 and zb0 + 1 <= Zn:
            f1 = np.asarray(self.s[za0 + 1:zb0 + 1, a:a + wa], np.float32)
            f = (1.0 - frac) * f + frac * f1
        f = 1.0 - np.clip((f - self.lo) / (self.hi - self.lo + 1e-6), 0, 1)
        f = np.rot90(f, 1)
        if self.fliplr:
            f = f[:, ::-1]
        if width is None:
            width = max(2, int(round(height * f.shape[1] / f.shape[0])))
        im = Image.fromarray((f * 255).astype(np.uint8)).resize((width, height))
        return np.asarray(im).astype(np.float32) / 255


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #
def phase_shift_bounded(a, b, fy=0.20, fx=0.20):
    """Phase-correlation shift with the peak restricted to |dy|<fy*H, |dx|<fx*W.

    The unbounded wrap-around shift makes the match score nearly invariant to
    the strip position on low-content (dark) frames — a mis-framed window can
    be rolled back into alignment — which let the search drift off the true
    cell phase. Bounding the shift keeps registration for the small genuine
    crop-centering offset only."""
    A = np.fft.rfft2(a - a.mean())
    B = np.fft.rfft2(b - b.mean())
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12
    r = np.fft.irfft2(R, s=a.shape)
    h, w = a.shape
    my, mx = max(1, int(fy * h)), max(1, int(fx * w))
    rows = np.zeros(h, bool)
    rows[:my + 1] = rows[-my:] = True
    cols = np.zeros(w, bool)
    cols[:mx + 1] = cols[-mx:] = True
    r = np.where(rows[:, None] & cols[None, :], r, -np.inf)
    peak = np.unravel_index(np.argmax(r), r.shape)
    dy = peak[0] if peak[0] <= h // 2 else peak[0] - h
    dx = peak[1] if peak[1] <= w // 2 else peak[1] - w
    return int(dy), int(dx)


def overlap_views(a, b, dy, dx):
    """The overlapping regions of a and b after shifting b by (dy, dx) —
    NO wrap-around, so misframed content cannot roll back into alignment."""
    h, w = a.shape
    av = a[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)]
    bv = b[max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)]
    return av, bv


def match_score(gt, gt_g, pred):
    """Gradient correlation after BOUNDED gradient-domain phase-corr
    registration, scored on the un-wrapped overlap. Returns (score, dy, dx)."""
    pg = grad_mag(pred)
    dy, dx = phase_shift_bounded(gt_g, pg)
    av, bv = overlap_views(gt, pred, dy, dx)
    if av.size < 0.3 * gt.size:
        return -1.0, dy, dx
    return M.gradient_correlation(av, bv), dy, dx


def calibrate_scale(sw, gt_m, gt_g, xs, scores, n_anchor=20):
    """Global pred<->GT ZOOM calibration: find the strip-window extents (fy
    along-film, fx across-film) that render our film at the same scale as the
    GT scan, by maximizing the registered match score on the best-scoring
    anchor frames. One fixed physical geometry -> calibrate once, coarse to
    fine. (Uncalibrated extents = the 'unroll looks more zoomed-in' artifact.)"""
    idx = np.argsort(scores)[-n_anchor:]
    Hm, Wm = sw.match_h, sw.match_w

    def mean_score(fy, fx):
        return float(np.mean([match_score(
            gt_m[j], gt_g[j],
            sw.window_full(xs[j], Hm, Wm, fy=fy, fx=fx))[0] for j in idx]))

    best = (1.0, 1.0, mean_score(1.0, 1.0))
    for step, span in ((0.05, 0.10), (0.02, 0.04), (0.01, 0.01)):
        cy, cx = best[0], best[1]
        for fy in np.arange(cy - span, cy + span + 1e-9, step):
            for fx in np.arange(cx - span, cx + span + 1e-9, step):
                s = mean_score(fy, fx)
                if s > best[2]:
                    best = (float(fy), float(fx), s)
    return best


def motion_masks(gt_m, sigma=4.0, min_amp=0.02):
    """Per-GT-frame motion weight map: where frame j differs from its temporal
    neighbours. This is exactly the region that DISAMBIGUATES adjacent film
    cells of slow animation (a global score barely sees a closing mouth).
    None where the scene is static (no motion information)."""
    from scipy.ndimage import gaussian_filter
    n = len(gt_m)
    out = []
    for j in range(n):
        m = np.zeros_like(gt_m[j])
        if j > 0:
            m = np.maximum(m, np.abs(gt_m[j] - gt_m[j - 1]))
        if j < n - 1:
            m = np.maximum(m, np.abs(gt_m[j] - gt_m[j + 1]))
        m = gaussian_filter(m, sigma)
        out.append(m if m.max() > min_amp else None)
    return out


def weighted_corr(a, b, w):
    ws = float(w.sum())
    if ws < 1e-9:
        return 0.0
    wn = w / ws
    ma, mb = float((wn * a).sum()), float((wn * b).sum())
    va = float((wn * (a - ma) ** 2).sum())
    vb = float((wn * (b - mb) ** 2).sum())
    if va <= 0 or vb <= 0:
        return 0.0
    return float((wn * (a - ma) * (b - mb)).sum() / np.sqrt(va * vb))


def match_score_dp(gt_l, gt_lg, pred, mask, w_motion=0.7):
    """DP emission score on CONTRAST-NORMALIZED frames (gt_l = _lcn(gt),
    gt_lg = grad_mag(gt_l)): global gradient correlation + motion-masked
    gradient correlation (temporal disambiguation), on the registered overlap.
    LCN matters twice: it makes washout-zone frames scoreable at all, and it
    stops the (strong, pose-independent) background/shading gradients from
    swamping the pose evidence that separates adjacent cells."""
    pl = _lcn(pred)
    dy, dx = phase_shift_bounded(gt_lg, grad_mag(pl))
    h, w = gt_l.shape
    ay, ax = (max(0, dy), h + min(0, dy)), (max(0, dx), w + min(0, dx))
    by, bx = (max(0, -dy), h + min(0, -dy)), (max(0, -dx), w + min(0, -dx))
    av, bv = gt_l[ay[0]:ay[1], ax[0]:ax[1]], pl[by[0]:by[1], bx[0]:bx[1]]
    if av.size < 0.3 * gt_l.size:
        return -1.0
    s = M.gradient_correlation(av, bv)
    if mask is not None:
        mv = mask[ay[0]:ay[1], ax[0]:ax[1]]
        s += w_motion * weighted_corr(grad_mag(av), grad_mag(bv), mv)
    return s


def search_frame(sw, gt, gt_g, x_init, radius, coarse_step):
    """Coarse-to-fine 1-D search of the strip position for one GT frame."""
    best_x, best_s = x_init, -2.0
    for x in np.arange(x_init - radius, x_init + radius + 1e-6, coarse_step):
        s = match_score(gt, gt_g, sw.window_small(x))[0]
        if s > best_s:
            best_s, best_x = s, x
    fine_step = 1.0 / sw.sa  # 1 px of the shrunk strip
    x0 = best_x
    for x in np.arange(x0 - coarse_step, x0 + coarse_step + 1e-6, fine_step):
        s = match_score(gt, gt_g, sw.window_small(x))[0]
        if s > best_s:
            best_s, best_x = s, x
    return float(best_x), float(best_s)


def robust_line(j, x, n_iter=3, k=2.5):
    """Iteratively reweighted linear fit x ~ a + b*j with outlier rejection.
    Returns (a, b, inlier_mask)."""
    j, x = np.asarray(j, float), np.asarray(x, float)
    keep = np.ones(len(j), bool)
    a = b = 0.0
    for _ in range(n_iter):
        b, a = np.polyfit(j[keep], x[keep], 1)
        r = x - (a + b * j)
        s = np.median(np.abs(r[keep])) * 1.4826 + 1e-6
        keep = np.abs(r) < k * max(s, 8.0)
    return a, b, keep


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _font(size=18):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _to_u8(f):
    return (np.clip(f, 0, 1) * 255).astype(np.uint8)


def _resize(f, h, w):
    return np.asarray(Image.fromarray(_to_u8(f)).resize((w, h))).astype(np.float32) / 255


def render_pair(gt_full, pred_full, labels, font):
    """One PNG row: GT | ours | overlay (GT green / ours magenta) | |diff|.

    gt_full/pred_full: float [0,1], same height (widths may differ). Overlay +
    diff use the pred registered (gradphase shift) + histogram-matched onto GT.
    """
    h = gt_full.shape[0]
    pg = _resize(pred_full, h, gt_full.shape[1])
    dy, dx = phase_shift_bounded(grad_mag(gt_full), grad_mag(pg))
    reg = np.zeros_like(pg)
    av, bv = overlap_views(reg, pg, dy, dx)
    av[:] = bv  # zero-fill shift, no wrap-around
    reg = histogram_match(reg, gt_full)
    over = np.stack([reg, gt_full, reg], axis=-1)  # GT green, ours magenta
    diff = np.abs(reg - gt_full)

    bar, gapw = 26, 6
    panels = [np.repeat(_to_u8(gt_full)[..., None], 3, 2),
              np.repeat(_to_u8(pred_full)[..., None], 3, 2),
              _to_u8(over),
              np.repeat(_to_u8(diff)[..., None], 3, 2)]
    gap = np.zeros((h + bar, gapw, 3), np.uint8)
    cols = []
    for p, lab in zip(panels, labels):
        c = np.zeros((h + bar, p.shape[1], 3), np.uint8)
        c[bar:] = p
        im = Image.fromarray(c)
        d = ImageDraw.Draw(im)
        d.text((4, 4), lab, fill=(255, 255, 0), font=font)
        cols.append(np.asarray(im))
        cols.append(gap)
    return np.concatenate(cols[:-1], axis=1), float(np.hypot(dy, dx))


def save_montage_grid(rows_imgs, out_path, n=16, cols=4):
    idx = np.linspace(0, len(rows_imgs) - 1, min(n, len(rows_imgs)), dtype=int)
    tiles = [rows_imgs[i] for i in idx]
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    rows = int(np.ceil(len(tiles) / cols))
    canvas = np.zeros((rows * (h + 8), cols * (w + 8), 3), np.uint8)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols)
        canvas[r * (h + 8):r * (h + 8) + t.shape[0],
               c * (w + 8):c * (w + 8) + t.shape[1]] = t
    Image.fromarray(canvas).save(out_path)


def save_curve_plot(recs, a, b, trim_mask, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    j = np.array([r["gt"] for r in recs])
    x = np.array([r["x"] for r in recs])
    s = np.array([r["score"] for r in recs])
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(j, a + b * j, "k--", lw=0.8, label=f"fit x={a:.0f}{b:+.2f}·j")
    ax[0].plot(j[~trim_mask], x[~trim_mask], ".", ms=3, label="kept")
    ax[0].plot(j[trim_mask], x[trim_mask], "x", ms=4, color="crimson", label="trimmed")
    ax[0].set_ylabel("strip x (px)")
    ax[0].legend()
    ax[1].plot(j[~trim_mask], s[~trim_mask], ".-", ms=3, lw=0.6)
    ax[1].plot(j[trim_mask], s[trim_mask], "x", ms=4, color="crimson")
    ax[1].set_xlabel("GT frame")
    ax[1].set_ylabel("grad-corr")
    fig.suptitle("GT frame -> strip position match")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _lcn(img, sigma=6.0):
    """Local contrast normalization: removes the smooth washout/shading field
    (which otherwise drowns the faint content gradients) and equalizes edge
    strength — the washout-zone frames DO contain lockable content, it is just
    ~50x lower contrast than the banding."""
    from scipy.ndimage import gaussian_filter
    m = gaussian_filter(img, sigma)
    d = img - m
    v = gaussian_filter(d * d, sigma)
    return d / (np.sqrt(v) + 1e-3)


def gt_lock_corrections(sw, gt_m, xs, dys, dxs, scores, end_exclude=(0, 0)):
    """Per-frame GT-lock: convert each pair's measured registration shift into
    a window correction (arc + z, strip px) so the content sits exactly where
    GT has it — NO smoothing (a smoothed drift lags the truth and catches up =
    the sawtooth). Returns (xc = locked centers, cz = z-shifts, good mask).

    Frames without a reliable GT lock (washout, dark, saturated shift, and the
    known-bad end cells via end_exclude) are bridged in two steps:
      1. POSITION interpolation between locked neighbours (not correction
         interpolation — xs still carries the junk drift bump there);
      2. SELF-CHAIN refinement: register consecutive PRED windows against each
         other. Frame lines + faint content are film-fixed, so the pairwise
         shift measures the true local advance even where GT sees nothing —
         this pins the VISIBLE structure (the sliding frame line the eye
         catches) — then the chain is closed onto the flanking locked anchors.
    """
    n = len(xs)
    Hm, Wm = sw.match_h, sw.match_w
    za, zb = sw._z_extent()
    zsc = (zb - za) / Wm
    cy = dys / sw.sa                 # measured shift -> strip px along arc
    # Z SIGN DEPENDS ON THE MIRROR. window_full slices z, then rot90 (displayed
    # columns = strip z), then optionally fliplr (which REVERSES that axis). So
    # the direction the z window must move to shift content by +dx flips with
    # fliplr. The + convention here was validated with fliplr=True (old scan);
    # with --no-fliplr (the new full-width scan) it must be negated, or every
    # correction is applied backwards and the lock DOUBLES the error instead of
    # removing it (measured: mean|dx| 8.8 -> 27.7 px).
    zsign = 1.0 if sw.fliplr else -1.0
    cz = dxs * zsc * zsign           # -> strip px along z
    good = (scores > 0.15) & (np.abs(dys) < 0.19 * Hm) & \
           (np.abs(dxs) < 0.19 * Wm)
    if end_exclude[0]:
        good[:end_exclude[0]] = False   # banding cells: locks are bogus
    if end_exclude[1]:
        good[-end_exclude[1]:] = False
    ends = np.zeros(n, bool)
    if end_exclude[0]:
        ends[:end_exclude[0]] = True
    if end_exclude[1]:
        ends[-end_exclude[1]:] = True

    # second-chance lock on contrast-normalized frames (washout rescue).
    # A single bounded measurement cannot recover frames whose position error
    # exceeds the ±19% registration range (they saturate at the bound), so
    # SEARCH over window offsets and keep the best LCN-correlated lock.
    n_rescued = 0
    for j in np.where(~good & ~ends)[0]:
        la = _lcn(gt_m[j])
        lg = grad_mag(la)
        best = (0.15, None)             # correlation floor for acceptance
        for off in np.arange(-0.4, 0.401, 0.05) * sw.pitch:
            lb = _lcn(sw.window_small(xs[j] + off))
            dy, dx = phase_shift_bounded(lg, grad_mag(lb))
            if abs(dy) >= 0.19 * Hm or abs(dx) >= 0.19 * Wm:
                continue
            av, bv = overlap_views(la, lb, dy, dx)
            if av.size < 0.3 * la.size:
                continue
            c = M.gradient_correlation(av, bv)
            if c > best[0]:
                best = (c, (off + dy / sw.sa, dx * zsc * zsign))
        if best[1] is not None:
            cy[j], cz[j] = best[1]
            good[j] = True
            n_rescued += 1
    if n_rescued:
        print(f"  lock: {n_rescued} frames rescued via contrast-normalized "
              f"search", flush=True)

    jj = np.arange(n)
    xc = xs + cy
    if not good.any() or good.all():
        return xc, cz, good
    g = np.where(good)[0]
    xi = np.interp(jj, g, xc[g])
    if len(g) > 1:
        step = np.median(np.diff(xc[g]) / np.diff(g))
        xi[:g[0]] = xc[g[0]] + (jj[:g[0]] - g[0]) * step
        xi[g[-1] + 1:] = xc[g[-1]] + (jj[g[-1] + 1:] - g[-1]) * step
    xc = np.where(good, xc, xi)
    cz = np.where(good, cz, np.interp(jj, g, cz[g]))

    def pair_shift(xa, xb):
        """(dy, dx) aligning window(xb) onto window(xa) on contrast-normalized
        frames; None unless the aligned windows actually correlate (adjacent
        cells are near-identical, so a valid measurement must match well —
        raw chain measurements in the low-signal zone inject jumps)."""
        wa, wb = _lcn(sw.window_small(xa)), _lcn(sw.window_small(xb))
        dy, dx = phase_shift_bounded(grad_mag(wa), grad_mag(wb))
        if abs(dy) >= 0.19 * Hm or abs(dx) >= 0.19 * Wm:
            return None
        av, bv = overlap_views(wa, wb, dy, dx)
        if av.size < 0.3 * wa.size or M.gradient_correlation(av, bv) < 0.25:
            return None
        return dy, dx

    # self-chain each maximal bad run
    j = 0
    while j < n:
        if good[j]:
            j += 1
            continue
        a = j
        while j < n and not good[j]:
            j += 1
        b = j - 1                       # bad run [a, b]
        left = a - 1 if a > 0 else None
        right = b + 1 if b + 1 < n else None
        if left is not None:            # chain forward from the left anchor
            idx = list(range(a, b + 1))
            prev = left
        elif right is not None:         # no left anchor: chain from the right
            idx = list(range(b, a - 1, -1))
            prev = right
        else:
            continue
        for k in idx:
            s = pair_shift(xc[prev], xc[k])
            if s is not None:
                xc[k] += s[0] / sw.sa
                cz[k] += s[1] * zsc
            prev = k
        if left is not None and right is not None:
            e = pair_shift(xc[right - 1], xc[right])  # closure at the anchor
            err = -(e[0] / sw.sa) if e is not None else 0.0
            span = right - left
            for k in range(a, b + 1):
                xc[k] += err * (k - left) / span

    # ── z (across-film / horizontal) stabilization: GT/self fusion ──
    # The unrolled STRIP itself has the film content wobbling in z ALONG the
    # arc (a walk/unroll residual) with a FAST per-frame component. Smoothing
    # the window position CANNOT fix that (the wobble stays in the content), so
    # the wobble must be TRACKED per frame — but the cross-modal GT dx is too
    # noisy to track cleanly. Fusion:
    #   high freq (fast wobble)  <- SELF-registration of consecutive windows
    #                               on our own high-SNR content (clean, but
    #                               integration drifts)
    #   low  freq (absolute pos) <- GT dx (noisy per frame but drift-free)
    # z = lowpass(cz_GT) + highpass(cumsum(self-shift)).
    from scipy.ndimage import gaussian_filter1d
    for j in np.where(good)[0]:      # GT z-refinement (absolute reference)
        la = _lcn(gt_m[j])
        lb = _lcn(sw.window_full(xc[j], Hm, Wm, zshift=cz[j]))
        dy, dx = phase_shift_bounded(grad_mag(la), grad_mag(lb))
        if abs(dx) < 0.19 * Wm:
            av, bv = overlap_views(la, lb, dy, dx)
            if av.size > 0.3 * la.size and M.gradient_correlation(av, bv) > 0.15:
                cz[j] += dx * zsc
    gg = np.where(good)[0]
    if len(gg) > 5:
        cz_gt = np.interp(jj, gg, cz[gg])
        # self-shift (strip px) of consecutive windows at a fixed z reference
        dz = np.zeros(n)
        prevw = None
        for j in range(n):
            w = _lcn(sw.window_full(xc[j], Hm, Wm, zshift=0.0))
            if prevw is not None:
                dy, dx = phase_shift_bounded(grad_mag(prevw), grad_mag(w))
                if abs(dx) < 0.19 * Wm:
                    av, bv = overlap_views(prevw, w, dy, dx)
                    if av.size > 0.3 * w.size and \
                            M.gradient_correlation(av, bv) > 0.2:
                        dz[j] = dx * zsc
            prevw = w
        pic = np.cumsum(dz)
        s = 4.0
        hp = pic - gaussian_filter1d(pic, s)
        lp = gaussian_filter1d(cz_gt, s)
        # sign of the self-shift vs the z axis is ambiguous (fliplr) -> pick
        # the sign whose rendered frames self-align best (min frame-to-frame
        # residual = the actual "no horizontal motion" objective)
        def ftf_resid(czt):
            r, pw = [], None
            for j in range(0, n, 2):
                w = _lcn(sw.window_full(xc[j], Hm, Wm, zshift=czt[j]))
                if pw is not None:
                    dy, dx = phase_shift_bounded(grad_mag(pw), grad_mag(w))
                    r.append(abs(dx))
                pw = w
            return float(np.mean(r))
        cz_plus, cz_minus = lp + hp, lp - hp
        cz = cz_plus if ftf_resid(cz_plus) <= ftf_resid(cz_minus) else cz_minus
    return xc, cz, good


def export_stabilized_video(sw, gt_m, gt_g, xc, cz, ks, dys, dxs, good,
                            out_path, height=720, duration=9.0):
    """GT-LOCKED film video from the locked window centers (see
    gt_lock_corrections). Never-scanned cells (GT skips, e.g. at the tear)
    are inserted between their flanks so the film stays complete."""
    import imageio.v2 as iio
    n = len(xc)
    Hm, Wm = sw.match_h, sw.match_w
    jj = np.arange(n)
    entries = []
    for j in range(n):
        if j and ks[j - 1] - ks[j] == 2:  # GT skip: insert the missing cell
            entries.append(((xc[j - 1] + xc[j]) / 2, (cz[j - 1] + cz[j]) / 2))
        entries.append((xc[j], cz[j]))
    W_out = int(round(height * Wm / Hm))
    W_out += W_out % 2

    def _write(path, ents):
        fps = max(6, int(round(len(ents) / duration)))
        wr = iio.get_writer(path, fps=fps, codec="libx264", quality=8,
                            macro_block_size=1)
        for x, z in ents:
            f = sw.window_full(x, height, W_out, zshift=z)
            wr.append_data((np.clip(f, 0, 1) * 255).astype(np.uint8))
        wr.close()
        return fps

    fps = _write(out_path, entries)
    # exact-1:1-with-GT variant for the metric eval (no inserted cells: frame
    # i of this video corresponds to GT leader-trimmed frame i exactly)
    _write(out_path.replace(".mp4", "_gtsync.mp4"),
           [(xc[j], cz[j]) for j in range(n)])
    # validation: the lock should have zeroed the residual shifts
    val = [match_score(gt_m[j], gt_g[j],
                       sw.window_full(xc[j], Hm, Wm, zshift=cz[j]))[1:]
           for j in jj[good][::max(1, int(np.count_nonzero(good)) // 25)]]
    va = np.abs(np.array(val))
    print(f"  stabilized video: {len(entries)} frames @ {fps}fps -> "
          f"{out_path}\n  residual after lock (match px): "
          f"mean|dy| {va[:, 0].mean():.1f} (was {np.abs(dys[good]).mean():.1f}),"
          f" mean|dx| {va[:, 1].mean():.1f} (was {np.abs(dxs[good]).mean():.1f})",
          flush=True)


def _save_drift_plot(drift, dys, dxs, pitch, out_path):
    """Frame-line phase drift along the film + residual per-pair registration
    shifts (dy = along-film after drift correction, dx = across-film z-wander;
    a smooth nonzero dx trend would need z-stabilization, not better cutting)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    j = np.arange(len(drift))
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax[0].plot(j, drift, "-", lw=1)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_ylabel("phase drift (strip px)")
    ax[0].set_title(f"frame-line drift vs constant pitch grid "
                    f"(cell = {pitch:.1f}px)")
    ax[1].plot(j, dys, ".-", ms=2, lw=0.6, label="reg dy (along film)")
    ax[1].plot(j, dxs, ".-", ms=2, lw=0.6, label="reg dx (across film)")
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_xlabel("GT frame")
    ax[1].set_ylabel("residual shift (match px)")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip", required=True, help="wholeroll.npy (z x arc).")
    ap.add_argument("--gt", required=True, help="GT optical-scan mp4.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gt-crop", type=parse_crop,
                    default=(0.235, 0.12, 0.81, 0.88),
                    help="GT picture-area crop x0,y0,x1,y1 (x0=0.235 verified "
                         "2026-07-16: 0.21 still included the left film-edge "
                         "band next to the sprockets).")
    ap.add_argument("--z-crop", type=parse_crop, default=None,
                    help="Strip z-margin crop lo,hi fractions (default 0.05,0.95).")
    ap.add_argument("--pitch", type=float, default=0.0,
                    help="Override the frame pitch (px); 0 = auto (phase fit).")
    ap.add_argument("--match-height", type=int, default=256)
    ap.add_argument("--pair-height", type=int, default=360)
    ap.add_argument("--search-cells", type=float, default=0.75,
                    help="Coarse search radius around the init, in cells.")
    ap.add_argument("--trim-start", type=int, default=4,
                    help="Drop pairs matched to the first N cells of the unroll "
                         "timeline (innermost banding / bad end fits).")
    ap.add_argument("--trim-end", type=int, default=2,
                    help="Drop pairs matched to the last N cells.")
    ap.add_argument("--no-fliplr", action="store_true",
                    help="Disable the mirrored pred orientation.")
    ap.add_argument("--montage", action="store_true",
                    help="Also save a 16-pair overview montage.")
    ap.add_argument("--video", action="store_true",
                    help="Also write side_by_side.mp4 (kept pairs, GT timeline).")
    ap.add_argument("--export-video", action="store_true",
                    help="Also write film_stabilized.mp4: fixed-size windows, "
                         "each LOCKED to its GT frame by the measured "
                         "registration shift (no smoothing -> no sawtooth).")
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()
    zcrop = args.z_crop if args.z_crop else (0.05, 0.95)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── strip + cell grid ──
    print("Loading strip...", flush=True)
    strip, lo, hi = load_strip(args.strip)
    col = np.asarray(strip, np.float32).mean(axis=0) \
        if strip.shape[0] < 4096 else None
    if col is None:  # (never for our 1184-z strips, kept for safety)
        col = np.asarray(strip[::4], np.float32).mean(axis=0)
    bounds, pitch = phase_lock_bounds(col, args.pitch)
    n_cells = len(bounds) - 1
    print(f"  strip {strip.shape}, pitch {pitch:.2f}, {n_cells} cells", flush=True)

    # ── GT ──
    print("Loading GT...", flush=True)
    gt = load_video_luma(args.gt)
    gt, n_drop = trim_leader(gt)
    gt = apply_crop(gt, args.gt_crop)
    n_gt = len(gt)
    Hm = args.match_height
    Wm = max(2, int(round(Hm * gt.shape[2] / gt.shape[1])))
    gt_m = [_resize(f, Hm, Wm) for f in gt]
    gt_g = [grad_mag(f) for f in gt_m]
    print(f"  {n_gt} GT frames after {n_drop}-frame leader trim; "
          f"match size {Hm}x{Wm}", flush=True)

    sw = StripWindows(strip, lo, hi, zcrop, Hm, Wm, fliplr=not args.no_fliplr)
    sw.build_small(pitch)

    # ── init: GT j -> reversed cell centers (play order is --reverse) ──
    centers = (bounds[:-1] + bounds[1:]) / 2.0
    cell_of_j = (n_cells - 1) - np.arange(n_gt) * (n_cells - 1) / max(n_gt - 1, 1)
    x_init = np.interp(cell_of_j, np.arange(n_cells), centers)

    # ── pass 1: per-frame coarse-to-fine search ──
    radius = args.search_cells * pitch
    cstep = 8.0 / sw.sa  # 8 shrunk px per coarse step
    print(f"Matching {n_gt} frames (radius ±{radius:.0f}px, coarse step "
          f"{cstep:.1f}px)...", flush=True)
    xs, scores = np.zeros(n_gt), np.zeros(n_gt)
    for j in range(n_gt):
        xs[j], scores[j] = search_frame(sw, gt_m[j], gt_g[j], x_init[j],
                                        radius, cstep)
        if j % 25 == 0:
            print(f"  frame {j}/{n_gt}  x={xs[j]:.0f}  s={scores[j]:.3f}",
                  flush=True)

    # ── global zoom calibration, then rebuild the match-scale strip with it ──
    fy, fx, cal_s = calibrate_scale(sw, gt_m, gt_g, xs, scores)
    print(f"Scale calibration: fy={fy:.3f} fx={fx:.3f} "
          f"(anchor score {cal_s:.3f})", flush=True)
    if abs(fy - 1.0) > 1e-3 or abs(fx - 1.0) > 1e-3:
        sw.build_small(pitch, fy=fy, fx=fx)

    gt_masks = motion_masks(gt_m)
    gt_lcn = [_lcn(f) for f in gt_m]
    gt_lcn_g = [grad_mag(f) for f in gt_lcn]

    # ── pass 2: cell-quantized dynamic program ──
    # Both sequences are CELL-locked: the GT scanner advances exactly one film
    # cell per frame (a few duplicated scans advance zero) and the strip's cell
    # pitch is constant (phase-lock validated, drift-free). So the true map is
    #   x(j) = x_phase + k(j)·pitch,  k monotone non-increasing, steps 0/-1.
    # Per-frame free search jitters ±0.25 cell on low-contrast frames (and can
    # land half-a-cell off on dark iris frames); quantizing to the consensus
    # cell phase + DP over k removes those errors structurally.
    jj = np.arange(n_gt)
    a, b, _ = robust_line(jj, xs)
    w = np.clip(scores, 0.0, None)
    ang = (xs - bounds[0]) / pitch * 2 * np.pi
    p_star = (np.angle((w * np.exp(1j * ang)).sum()) % (2 * np.pi)) \
        / (2 * np.pi) * pitch
    kf = (xs - bounds[0] - p_star) / pitch
    print(f"Line fit x = {a:.1f} {b:+.3f}·j ; cell phase p* = {p_star:.1f}px; "
          f"DP over cell indices...", flush=True)

    def cand_x(k):
        return bounds[0] + p_star + k * pitch

    # ±4 cells: a GLOBAL offset of a few cells in the init/pass-1 chain is
    # representable and correctable ONLY if the candidate set spans it (a ±1
    # set let a ~2-cell init error persist through everything — caught by the
    # user as "the unroll is always a few frames behind").
    cands = [[int(round(kf[j])) + d for d in range(-4, 5)]
             for j in range(n_gt)]
    # Transition penalties: a GT duplicate (advance 0) is only plausible where
    # consecutive GT frames are near-identical (the scanner re-captured a
    # cell), so gate the dup penalty on GT's own frame-to-frame difference;
    # both dup and skip are otherwise expensive so score noise (adjacent cells
    # of slow animation differ by ~0.01-0.05) cannot zigzag the path.
    gtd = np.array([1.0] + [float(np.abs(gt_m[j] - gt_m[j - 1]).mean())
                            for j in range(1, n_gt)])
    # Penalties sized to the DP score scale (gc + motion term ≈ 2x plain gc)
    # and gated on GT's own evidence BOTH ways: a dup step needs GT j ≈ j-1
    # (scanner re-scan); a skip step needs an unusually LARGE GT advance (a
    # 2-cell jump, e.g. the torn-perforation transport slip at the visible
    # tear). Ungated transitions stay possible but must beat a heavy penalty
    # (a sustained real misalignment always does; per-frame noise never).
    DUP_PEN = np.where(gtd <= np.percentile(gtd[1:], 6), 0.05, 0.8)
    SKIP_PEN = np.where(gtd >= np.percentile(gtd[1:], 85), 0.30, 0.8)

    # The true frame-line phase DRIFTS slowly along the strip (unroll arc
    # stretch + film shrinkage), so a constant-phase grid straddles cells
    # where the drift accumulates. Iterate: DP on the (drift-corrected) grid,
    # then update the smooth drift from the measured along-film registration
    # offset dy at each position — within the bounded registration range the
    # overlap score is quasi-FLAT in window position (registration compensates
    # the move), so searching x for best score is degenerate; dy is the sharp
    # signal, and the correct window is the one where dy == 0.
    from scipy.ndimage import gaussian_filter1d
    drift = np.zeros(n_gt)
    for it in range(3):
        sc = {}
        for j in range(n_gt):
            for k in cands[j]:
                sc[(j, k)] = match_score_dp(
                    gt_lcn[j], gt_lcn_g[j],
                    sw.window_small(cand_x(k) + drift[j]), gt_masks[j])
        back = [{k: (sc[(0, k)], None) for k in cands[0]}]
        for j in range(1, n_gt):
            cur = {}
            for k in cands[j]:
                opts = []
                for kp, pen in ((k, DUP_PEN[j]), (k + 1, 0.0),
                                (k + 2, SKIP_PEN[j])):
                    if kp in back[j - 1]:
                        opts.append((back[j - 1][kp][0] - pen, kp))
                if not opts:  # chain break (wild pass-1 jump): reconnect
                    opts = [(back[j - 1][kp][0] - 0.5, kp)
                            for kp in back[j - 1]]
                v, kp = max(opts)
                cur[k] = (v + sc[(j, k)], kp)
            back.append(cur)
        k_end = max(back[-1], key=lambda k: back[-1][k][0])
        ks = [k_end]
        for j in range(n_gt - 1, 0, -1):
            ks.append(back[j][ks[-1]][1])
        ks = np.array(ks[::-1])

        # drift update: move each window to zero its measured dy
        dmeas, wgt, adys = np.zeros(n_gt), np.zeros(n_gt), np.zeros(n_gt)
        for j in range(n_gt):
            s, dy, _ = match_score(gt_m[j], gt_g[j],
                                   sw.window_small(cand_x(ks[j]) + drift[j]))
            dmeas[j] = drift[j] + dy / sw.sa  # +dy rows == +dy/sa strip px
            wgt[j] = max(s, 0.0) ** 2
            if abs(dy) >= 0.19 * Hm:  # saturated at the search bound
                wgt[j] *= 0.1
            adys[j] = abs(dy)
        drift = gaussian_filter1d(dmeas * wgt, 4.0) / \
            (gaussian_filter1d(wgt, 4.0) + 1e-12)
        steps = -np.diff(ks)
        dups, skips = np.where(steps == 0)[0], np.where(steps == 2)[0]
        net_est = (n_gt - 1) - (xs[0] - xs[-1]) / pitch  # from pass-1 span
        print(f"  iter {it}: {len(dups)} GT dup steps at j={list(dups + 1)}; "
              f"{len(skips)} skips at j={list(skips + 1)} "
              f"(net {len(dups) - len(skips):+d}, span-implied "
              f"{net_est:+.2f}); drift "
              f"[{drift.min():+.0f}, {drift.max():+.0f}]px; "
              f"mean|dy| {adys.mean():.1f}px", flush=True)

    xs = cand_x(ks) + drift
    dys, dxs = np.zeros(n_gt, int), np.zeros(n_gt, int)
    for j in range(n_gt):
        scores[j], dys[j], dxs[j] = match_score(gt_m[j], gt_g[j],
                                                sw.window_small(xs[j]))
    # per-frame GT-lock: used by the pair panels, side_by_side AND the
    # stabilized video, so all outputs show the same locked windows; the
    # trim-zone end cells are excluded as anchors (banding = bogus locks)
    xc, czs, lock_good = gt_lock_corrections(
        sw, gt_m, xs, dys, dxs, scores,
        end_exclude=(args.trim_start, args.trim_end))

    # temporal-offset validation: each locked window's content should best
    # match ITS OWN GT frame (d=0); a shifted histogram = systematic
    # assignment error (the "unroll runs a few frames behind" failure mode)
    votes = []
    for j in range(4, n_gt - 4, 4):
        if gt_masks[j] is None:
            continue
        w = sw.window_full(xc[j], Hm, Wm, zshift=czs[j])
        best, bd = -9.0, 0
        for d in range(-3, 4):
            s = match_score_dp(gt_lcn[j + d], gt_lcn_g[j + d], w,
                               gt_masks[j + d])
            if s > best:
                best, bd = s, d
        votes.append(bd)
    votes = np.array(votes)
    print(f"  temporal-offset validation: median {np.median(votes):+.1f} "
          f"(want 0), hist "
          f"{ {d: int((votes == d).sum()) for d in range(-3, 4)} }",
          flush=True)

    # ── export GT-validated cell boundaries for the standalone video ──
    # (make_film_video --bounds-npy: cuts exactly at this verified geometry
    #  instead of re-detecting from the frame-line signal)
    cell_centers = {}
    for j in range(n_gt):
        cell_centers.setdefault(int(ks[j]), []).append(xc[j])
    centers = np.array(sorted(np.mean(v) for v in cell_centers.values()))
    edges = np.sort(np.concatenate([centers - pitch / 2, centers + pitch / 2]))
    merged = [edges[0]]
    for e in edges[1:]:
        if e - merged[-1] < 0.5 * pitch:
            merged[-1] = (merged[-1] + e) / 2
        else:
            merged.append(float(e))
    vb = np.array(merged)
    while vb[0] > pitch * 0.6:  # extend to the strip ends at local spacing
        vb = np.concatenate([[vb[0] - (vb[1] - vb[0])], vb])
    while vb[-1] < sw.W - pitch * 0.6:
        vb = np.concatenate([vb, [vb[-1] + (vb[-1] - vb[-2])]])
    vb = np.clip(vb, 0, sw.W)
    np.save(os.path.join(args.out_dir, "video_bounds.npy"), vb)
    print(f"  exported {len(vb) - 1} GT-validated cells -> video_bounds.npy",
          flush=True)

    if args.export_video:
        print("Exporting GT-locked stabilized video...", flush=True)
        export_stabilized_video(
            sw, gt_m, gt_g, xc, czs, ks, dys, dxs, lock_good,
            os.path.join(args.out_dir, "film_stabilized.mp4"))

    # monotonicity (b<0: x must be non-increasing along GT time)
    viol = int((np.diff(xs) > 0.5 * abs(b)).sum()) if b < 0 else \
        int((np.diff(xs) < -0.5 * abs(b)).sum())
    if viol:
        print(f"  WARNING: {viol} monotonicity violations remain", flush=True)

    # ── trim: drop pairs on the unroll's first/last cells (timeline ends) ──
    cell_idx = (xs - bounds[0]) / pitch  # fractional cell of each match
    tl_cell = (n_cells - 1) - cell_idx   # timeline cell (0 = video start)
    trim = (tl_cell < args.trim_start) | (tl_cell > n_cells - 1 - args.trim_end)
    print(f"Trim: {int(trim.sum())} pairs dropped "
          f"(first {args.trim_start} / last {args.trim_end} cells)", flush=True)

    # ── render pairs ──
    print("Rendering pair PNGs...", flush=True)
    for sub in ("pairs", "pairs_trimmed"):
        d = os.path.join(args.out_dir, sub)
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):  # stale pairs from a previous run's trim split
            os.remove(os.path.join(d, f))
    font = _font()
    Hp = args.pair_height
    recs, kept_rows = [], []
    Wp = int(round(Hp * gt.shape[2] / gt.shape[1]))
    for j in range(n_gt):
        gt_p = _resize(gt[j], Hp, Wp)
        # same size as the GT panel (calibrated zoom) AND GT-locked position:
        # the raw 'ours' panel is exactly the stabilized video's frame
        pred_p = sw.window_full(xc[j], Hp, Wp, zshift=czs[j])
        labels = [f"GT #{j}", f"unroll x={xc[j]:.0f} (cell {tl_cell[j]:.1f})",
                  f"overlay  gc={scores[j]:.3f}", "abs diff"]
        row, shift_p = render_pair(gt_p, pred_p, labels, font)
        sub = "pairs_trimmed" if trim[j] else "pairs"
        Image.fromarray(row).save(
            os.path.join(args.out_dir, sub, f"pair_g{j:03d}.png"))
        if not trim[j]:
            kept_rows.append(row)
        recs.append({"gt": j, "x": float(xs[j]), "x_locked": float(xc[j]),
                     "z_shift": float(czs[j]),
                     "cell_timeline": float(tl_cell[j]),
                     "score": float(scores[j]), "shift_px": shift_p,
                     "drift_px": float(drift[j]), "reg_dy": int(dys[j]),
                     "reg_dx": int(dxs[j]), "trimmed": bool(trim[j])})
        if j % 40 == 0:
            print(f"  pair {j}/{n_gt}", flush=True)

    save_curve_plot(recs, a, b, trim, os.path.join(args.out_dir, "match_curve.png"))
    _save_drift_plot(drift, dys, dxs, pitch,
                     os.path.join(args.out_dir, "drift_diag.png"))
    if args.montage and kept_rows:
        save_montage_grid(kept_rows, os.path.join(args.out_dir, "montage.png"))
    if args.video and kept_rows:
        import imageio.v2 as imageio
        h = max(r.shape[0] for r in kept_rows)
        w = max(r.shape[1] for r in kept_rows)
        w += w % 2
        h += h % 2
        wr = imageio.get_writer(os.path.join(args.out_dir, "side_by_side.mp4"),
                                fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=1)
        for r in kept_rows:
            c = np.zeros((h, w, 3), np.uint8)
            c[:r.shape[0], :r.shape[1]] = r
            wr.append_data(c)
        wr.close()

    kept = [r for r in recs if not r["trimmed"]]
    result = {
        "strip": os.path.abspath(args.strip),
        "gt": os.path.abspath(args.gt),
        "pitch": pitch, "n_cells": n_cells, "n_gt": n_gt,
        "gt_leader_dropped": n_drop,
        "line_fit": {"a": float(a), "b": float(b)},
        "cell_phase_px": float(p_star),
        "scale_fy": float(fy), "scale_fx": float(fx),
        "gt_duplicate_steps": [int(d + 1) for d in dups],
        "gt_skip_steps": [int(s + 1) for s in skips],
        "drift_px_range": [float(drift.min()), float(drift.max())],
        "trim_start_cells": args.trim_start, "trim_end_cells": args.trim_end,
        "n_kept": len(kept), "n_trimmed": int(trim.sum()),
        "score_mean_kept": float(np.mean([r["score"] for r in kept])),
        "score_median_kept": float(np.median([r["score"] for r in kept])),
        "pairs": recs,
    }
    with open(os.path.join(args.out_dir, "matches.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nDone: {len(kept)} kept / {int(trim.sum())} trimmed; "
          f"grad-corr mean {result['score_mean_kept']:.3f} "
          f"-> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
