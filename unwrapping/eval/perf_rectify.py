"""Perforation-locked framing + rectification of the full-width unrolled strip.

GT-FREE.  Everything here is driven by the two perforation rows found by
`detect_perfs.py`; the GT scan is never read.  That matters because every earlier
stabilizer in this project (`stabilize_horizontal.py --gt`, `stabilize_affine.py`,
`stabilize_flow.py`) registers to GT, so the stillest of them are viewing copies
rather than measurements.  Perfs are a physical ruler punched into the stock, so
locking to them is what a real film scanner does.

THE MODEL -- "two rails".
------------------------
The two perf rows are two curves running the length of the film at fixed,
known across-width positions.  Let

    u        nominal along-film coordinate (px).  Hole k of BOTH rows is DEFINED
             to sit at u = k * P*, which is what makes the output framing
             constant and removes the along-film scale error.
    ZL*, ZH* nominal across-film positions of the two rows (reel medians)
    t        = (Z - ZL*) / (ZH* - ZL*)   -- 0 at the low row, 1 at the high row

Then rectified position (Z, u) is sampled from strip position

    z = (1-t) * zL(u) + t * zH(u)
    c = (1-t) * cL(u) + t * cH(u)

with zL, zH, cL, cH smooth interpolants through the per-hole measurements.
This is a ruled (bilinear) warp: within a frame it is a full affine -- all six
DOF, translation / rotation / independent across and along scale / shear -- but
it is CONTINUOUS along the whole reel rather than fitted per frame, so no seam
appears at the cuts and one bad hole cannot throw a whole frame.

Conditioning is why this beats per-frame corner fitting: across-scale and shear
are measured over the ~1730 px baseline BETWEEN the rows, not across one 271 px
hole.  The picture band lies between the rows, so it is always interpolated.

WHAT IT REMOVES.  The film is static inside the roll, so there is no projector
gate wobble in this data.  What the perfs see is (a) how the film actually lies
in the roll -- lateral drift / telescoping, smooth over a winding -- and (b) our
unrolling's own geometric error, once-per-turn plus per-anchor noise.  Neither
belongs to the movie, and holding the perfs still removes both.

    python -m unwrapping.eval.perf_rectify <strip.npy> --perfs <perfs.npz> \
        --out-dir <dir> --reverse --invert
"""

import argparse
import os

import numpy as np

# ISO 16 mm double-perf, in mm.
ISO_APERTURE_ACROSS = 10.26     # picture width across the film
ISO_PERF_ALONG = 1.27           # hole size along the film
ISO_PERF_ACROSS = 1.83          # hole size across the film
ISO_FRAME_PITCH = 7.62          # one frame

# SCALE.  The strip is ISOTROPIC at pitch / 7.62 px per mm, and the crop is
# derived from that rather than from the perf-row separation.
#
# Getting this wrong costs an 11 % error, so the evidence: the along scale is
# anchored to the frame pitch, which is 7.62 mm by definition of the format.
# The measured hole size ACROSS the film is 269-275 px, and 1.83 mm at the along
# scale (148.2 px/mm) predicts 271 -- so across and along agree, i.e. isotropic.
# The measured perf-row SEPARATION, 1743 px, would instead imply 131.6 px/mm if
# the rows really were the textbook 13.25 mm apart.  The direct hole measurement
# wins over the assumed separation, so the rows are ~11.76 mm apart on this
# stock and 13.25 was the wrong constant.  Deriving the crop from the separation
# made the frame 11 % too narrow AND squashed its aspect to 1.20 where the ISO
# aperture is 1.35.


# ─────────────────────────────── rails ────────────────────────────────────

def _index_holes(c, pitch):
    """Assign each hole an integer frame index.

    Incremental (k_i = k_{i-1} + round(dc / pitch)) rather than
    round((c - c0) / pitch): the local pitch varies by a few percent along the
    reel, so an absolute index accumulates that drift and eventually slips a
    whole frame, whereas the incremental one only has to resolve a single gap.
    """
    k = np.zeros(len(c), int)
    for i in range(1, len(c)):
        step = int(round((c[i] - c[i - 1]) / pitch))
        k[i] = k[i - 1] + max(1, step)
    return k


def _robust_smooth(u, y, sigma_u, n_iter=3, k_mad=3.5):
    """Gaussian-weighted local-LINEAR smoother with MAD reweighting.

    Local-linear, not local-mean, because the sway genuinely ramps along the
    reel and a local mean would flatten the ramp and leave its own residual.
    """
    u = np.asarray(u, float)
    y = np.asarray(y, float)
    w = np.ones_like(y)
    out = y.copy()
    if sigma_u <= 0:
        return out, np.ones(len(y), bool)
    for _ in range(n_iter):
        for i in range(len(u)):
            d = u - u[i]
            g = np.exp(-0.5 * (d / sigma_u) ** 2) * w
            gs = g.sum()
            if gs < 1e-9:
                out[i] = y[i]
                continue
            md = (g * d).sum() / gs
            my = (g * y).sum() / gs
            vd = (g * (d - md) ** 2).sum() / gs
            cv = (g * (d - md) * (y - my)).sum() / gs
            out[i] = my + (cv / vd if vd > 1e-9 else 0.0) * (0.0 - md)
        r = y - out
        mad = np.median(np.abs(r - np.median(r))) + 1e-9
        w = (np.abs(r) < k_mad * 1.4826 * mad).astype(float)
    return out, w > 0


def _per_frame_noise(y):
    """Per-sample noise from the second difference of a smooth series.

    For a signal that is smooth frame to frame, the second difference is
    dominated by noise, and white noise of sd s gives a second difference of
    sd s*sqrt(6).  Robust (MAD) so the gross outliers do not set the scale.
    """
    d2 = y[2:] - 2 * y[1:-1] + y[:-2]
    d2 = d2[np.isfinite(d2)]
    if len(d2) < 8:
        return np.nan
    return float(np.median(np.abs(d2 - np.median(d2))) * 1.4826 / np.sqrt(6))


def _unslip(off, pitch):
    """Remove whole-frame CYCLE SLIPS from an along-film offset series.

    A missed or spurious detection makes the running frame index jump by one,
    which shows up as a ~+-pitch step in the offset and would move the framing
    by an entire frame -- visible as a skipped or repeated frame.  The genuine
    frame-to-frame change is a few tens of px at most, so any step near a whole
    pitch is a slip.  Same guard `make_film_video.tracked_bounds` applies to the
    analytic phase, in px instead of radians.
    """
    off = np.asarray(off, float)
    d = np.diff(off)
    n = np.round(d / pitch)
    if np.any(n != 0):
        print(f"  removed {int(np.abs(n).sum())} whole-frame cycle slips "
              f"from the along-film rail")
    d = d - pitch * n
    return np.concatenate([[off[0]], off[0] + np.cumsum(d)])


def _regular(k, y):
    """Put samples indexed by integer frame k onto a gap-free frame grid."""
    g = np.arange(int(k.min()), int(k.max()) + 1)
    return g, np.interp(g, k, y)


def _reject_outliers(y, n_sigma=4.0, window=9):
    """Flag samples far from a running median -- gross detector failures.

    These are NOT smoothed away: a single hole displaced by 300 px would drag a
    Gaussian-weighted mean for several frames around it, so it has to be removed
    before smoothing rather than by it.
    """
    from scipy.ndimage import median_filter
    med = median_filter(y, size=window, mode="nearest")
    r = y - med
    s = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-9
    return np.abs(r) < n_sigma * s


def build_rails(perfs, smooth_frames=1.0, along="perf", min_score=None,
                fix_scale=False, fix_shear=False, verbose=True):
    """Build the warp rails from the perforations and the frame lines.

    Fiducial choice is driven by MEASURED per-frame noise, not by preference:

      * ACROSS-film (the sway, and the across-film scale) can only come from the
        perforations -- the frame line runs along the film and says nothing
        about lateral position.
      * ALONG-film (framing) comes from the LOW perf row.  The frame lines look
        better on raw per-frame noise (6.9 px against 13.8 px) and they sit in
        the walked picture band rather than the extrapolated perf bands, so they
        were the first choice -- but their series carries gross non-slip errors
        that survive both cycle-slip removal and outlier rejection, leaving a
        2764 px (2.45 frame) drift and a local pitch dipping to 851 px.  The
        perf row, once its 4 cycle slips are removed, drifts 362 px (0.32
        frames) with the local pitch inside 1088-1173.  `along="frameline"` is
        kept for comparison but needs a better frame-line detector to be useful.
    """
    pitch = float(perfs["pitch"])
    lo = {k[3:]: perfs[k].copy() for k in perfs.files if k.startswith("lo_")}
    hi = {k[3:]: perfs[k].copy() for k in perfs.files if k.startswith("hi_")}

    for d, nm in ((lo, "low"), (hi, "high")):
        thr = min_score if min_score is not None else 0.5 * np.median(d["score"])
        keep = d["score"] >= thr
        if verbose:
            print(f"  {nm} perf row: {keep.sum()}/{len(keep)} holes with "
                  f"score >= {thr:.3f}")
        for k in list(d):
            d[k] = d[k][keep]

    # ── across-film rails, from the perforations ──
    kL = _index_holes(lo["c"], pitch)
    kH = _index_holes(hi["c"], pitch)
    shift = int(round(np.median(np.interp(lo["c"], hi["c"], kH.astype(float)) - kL)))
    kL = kL + shift
    k0 = min(kL.min(), kH.min())
    kL -= k0; kH -= k0

    gL, zL_raw = _regular(kL, lo["z"])
    gH, zH_raw = _regular(kH, hi["z"])
    g0 = max(gL.min(), gH.min()); g1 = min(gL.max(), gH.max())
    g = np.arange(g0, g1 + 1)
    zL_raw = np.interp(g, gL, zL_raw)
    zH_raw = np.interp(g, gH, zH_raw)

    okL = _reject_outliers(zL_raw)
    okH = _reject_outliers(zH_raw)
    if verbose:
        print(f"  sway per-frame noise: low {_per_frame_noise(zL_raw):.2f} px, "
              f"high {_per_frame_noise(zH_raw):.2f} px  "
              f"(rejected {(~okL).sum()} / {(~okH).sum()} gross outliers)")
    zL, _ = _robust_smooth(g[okL], zL_raw[okL], smooth_frames)
    zH, _ = _robust_smooth(g[okH], zH_raw[okH], smooth_frames)
    zL = np.interp(g, g[okL], zL)
    zH = np.interp(g, g[okH], zH)

    ZL = float(np.median(lo["z"]))
    ZH = float(np.median(hi["z"]))

    # SWAY: noise-weighted mean of the two rows.  The weights come out ~0.96 /
    # 0.04, i.e. this is essentially the low row, because the high row's 17.5 px
    # per-frame scatter is nearly as big as the sway itself.
    nl, nh = _per_frame_noise(zL_raw), _per_frame_noise(zH_raw)
    wl, wh = 1.0 / max(nl, 1e-3) ** 2, 1.0 / max(nh, 1e-3) ** 2
    wl, wh = wl / (wl + wh), wh / (wl + wh)
    sway = wl * (zL - ZL) + wh * (zH - ZH)
    if verbose:
        print(f"  sway = {wl:.2f}*low + {wh:.2f}*high (weights from the noise); "
              f"ptp {np.ptp(sway):.1f} px")

    # ACROSS-FILM SCALE needs BOTH rows to be trustworthy and the high row is
    # not: taking it at face value asks for an 11.3 % scale swing, which would
    # visibly pump the picture.  Off unless asked for.
    if fix_scale:
        half = 0.5 * ((zH - zL) - (ZH - ZL))
    else:
        half = np.zeros_like(sway)
    zL_rail = ZL + sway - half
    zH_rail = ZH + sway + half

    # ── along-film rails ──
    n_sub = sum(1 for k in perfs.files
                if k.startswith("fl_") and k[3:].isdigit())
    use_fl = along == "frameline" and n_sub >= 1
    if use_fl:
        fl_z = perfs["fl_z"]
        subs, noise = [], []
        for i in range(n_sub):
            x = np.asarray(perfs[f"fl_{i}"], float)
            ki = _index_holes(x, pitch)
            gi, xi = _regular(ki, x - ki * pitch)
            noise.append(_per_frame_noise(xi))
            subs.append((gi, xi))
            if verbose:
                print(f"  frame line sub-band z~{fl_z[i]:.0f}: per-frame noise "
                      f"{noise[-1]:.2f} px")
        best = int(np.nanargmin(noise))
        if verbose:
            print(f"  using frame-line sub-band {best} (z~{fl_z[best]:.0f}, "
                  f"noise {noise[best]:.2f} px) for the along-film rail")
        gi, xi = subs[best]
        # Align to the perf frame numbering by cross-correlating the two column
        # sequences.  Aligning by frame INDEX instead let a one-frame offset
        # between sub-bands masquerade as a 1129 px shear, which is how an
        # earlier version produced a nonsensical 16 deg median shear.
        c_along_raw = np.interp(g, gi + _align_offset(gi, xi, kL, lo["c"], pitch),
                                xi)
    else:
        _, craw = _regular(kL, lo["c"] - kL * pitch)
        c_along_raw = np.interp(g, np.arange(len(craw)) + kL.min(), craw)
        if verbose:
            print(f"  along-film from the LOW perf row, per-frame noise "
                  f"{_per_frame_noise(c_along_raw):.2f} px")

    c_along_raw = _unslip(c_along_raw, pitch)
    okc = _reject_outliers(c_along_raw)
    c_s, _ = _robust_smooth(g[okc], c_along_raw[okc], smooth_frames)
    c_along = np.interp(g, g[okc], c_s) + g * pitch
    if verbose:
        print(f"  along-film rail: {(~okc).sum()} outliers rejected, "
              f"drift ptp {np.ptp(c_along - g * pitch):.0f} px "
              f"({np.ptp(c_along - g * pitch) / pitch:.2f} frames)")

    # SHEAR would come from the frame line's tilt across the z sub-bands, but at
    # 7-18 px of noise over a 1085 px z baseline a single frame's shear is only
    # measurable to ~1 deg, which is the size of the effect.  Off unless asked.
    if fix_shear and use_fl and n_sub >= 2:
        A = np.full((n_sub, len(g)), np.nan)
        for i in range(n_sub):
            gi, xi = subs[i]
            A[i] = np.interp(g, gi + _align_offset(gi, xi, kL, lo["c"], pitch), xi)
        zs = np.asarray(perfs["fl_z"], float)
        Zc = zs - zs.mean()
        slope = (Zc[:, None] * (A - A.mean(axis=0))).sum(axis=0) / (Zc ** 2).sum()
        slope, _ = _robust_smooth(g, slope, smooth_frames)
        if verbose:
            print(f"  shear {np.degrees(np.arctan(np.median(slope))):+.2f} deg "
                  f"median, {np.degrees(np.arctan(np.ptp(slope))):.2f} deg swing")
    else:
        slope = np.zeros_like(c_along)

    cL = c_along + slope * (ZL - 0.5 * (ZL + ZH))
    cH = c_along + slope * (ZH - 0.5 * (ZL + ZH))
    cL = np.maximum.accumulate(cL)
    cH = np.maximum.accumulate(cH)

    u = g * pitch
    if verbose:
        print(f"  nominal rows ZL={ZL:.1f} ZH={ZH:.1f}  separation "
              f"{ZH - ZL:.1f} px = {(ZH - ZL) / (pitch / 7.62):.2f} mm "
              f"-- the textbook 13.25 is wrong for this stock, see the scale "
              f"note at the top of this module")
        print(f"  {len(g)} frames on the rails")
    return dict(pitch=pitch, ZL=ZL, ZH=ZH, uL=u, uH=u, frames=g,
                zL=zL_rail, cL=cL, zH=zH_rail, cH=cH, sway=sway,
                raw_lo=lo, raw_hi=hi, zL_raw=zL_raw, zH_raw=zH_raw)


def _align_offset(gi, xi, k_ref, c_ref, pitch):
    """Frame-index offset that puts series (gi, xi) on the reference numbering.

    Both series are periodic at the pitch, so alignment is done on the COLUMN
    positions: the offset is the median frame difference between each reference
    feature and the nearest feature of the other series.
    """
    ci = xi + gi * pitch
    j = np.clip(np.searchsorted(ci, c_ref), 1, len(ci) - 1)
    pick = np.where(np.abs(ci[j] - c_ref) < np.abs(ci[j - 1] - c_ref), j, j - 1)
    good = np.abs(ci[pick] - c_ref) < 0.45 * pitch
    if good.sum() < 10:
        return 0.0
    return float(np.median(k_ref[good] - gi[pick][good]))


# ──────────────────────────── phase calibration ───────────────────────────

def calibrate_perf_phase(strip, rails, band_rows, n_fold=None):
    """Where does the frame line sit relative to the perf centre?

    This is the ONE number that cannot come from the perfs: it is set by how the
    camera placed the picture relative to the perforation.  It is a CONSTANT of
    the stock, so measuring it once over the whole reel costs nothing in
    stillness -- the per-frame corrections stay entirely perf-driven.

    Method: resample the picture-band column mean into u (where frame lines are
    exactly periodic by construction), fold modulo the pitch, and take the
    darkest phase -- the frame line is the unexposed gap between pictures.
    """
    pitch = rails["pitch"]
    z0, z1 = band_rows
    prof_c = np.asarray(strip[z0:z1:4, :], np.float32).mean(axis=0)

    u = np.arange(rails["uL"].min(), rails["uL"].max(), 1.0)
    c_mid = 0.5 * (np.interp(u, rails["uL"], rails["cL"]) +
                   np.interp(u, rails["uH"], rails["cH"]))
    prof_u = np.interp(c_mid, np.arange(len(prof_c)), prof_c)

    n = int(pitch)
    m = (len(prof_u) // n) * n
    fold = prof_u[:m].reshape(-1, n).mean(axis=0)
    fold = fold - fold.mean()
    contrast = float(np.ptp(fold) / (np.std(prof_u) + 1e-9))
    print(f"  folded picture profile: darkest at {np.argmin(fold)} px, "
          f"brightest at {np.argmax(fold)} px, fold contrast {contrast:.2f}")
    return float(np.argmin(fold)), fold


# ───────────────────────────── rectification ──────────────────────────────

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


def rectify_frames(strip, rails, phase, crop_mm=ISO_APERTURE_ACROSS, full=False,
                   progress=40):
    """Yield (k, frame) with frame indexed [across, along] in nominal px."""
    Z, W = strip.shape
    pitch = rails["pitch"]
    ZL, ZH = rails["ZL"], rails["ZH"]
    px_mm = pitch / ISO_FRAME_PITCH          # isotropic; see the module header

    if full:
        Z0, Z1 = 0.0, float(Z)
    else:
        mid = 0.5 * (ZL + ZH)
        half = 0.5 * crop_mm * px_mm
        Z0, Z1 = mid - half, mid + half
    nZ = int(round(Z1 - Z0))
    nC = int(round(pitch))
    print(f"  scale {px_mm:.1f} px/mm; crop {crop_mm:.2f} x "
          f"{nC / px_mm:.2f} mm -> {nZ} x {nC} px (aspect {nZ / nC:.3f})",
          flush=True)

    Zg = (Z0 + np.arange(nZ, dtype=np.float64))[:, None]
    t = (Zg - ZL) / (ZH - ZL)

    kmin = int(np.ceil(max(rails["uL"].min(), rails["uH"].min()) / pitch))
    kmax = int(min(rails["uL"].max(), rails["uH"].max()) / pitch)
    print(f"  output frame {nZ} x {nC} px, cells {kmin}..{kmax}", flush=True)

    n_out = 0
    for k in range(kmin, kmax):
        u = k * pitch + phase + np.arange(nC, dtype=np.float64)[None, :]
        zl = np.interp(u, rails["uL"], rails["zL"])
        zh = np.interp(u, rails["uH"], rails["zH"])
        cl = np.interp(u, rails["uL"], rails["cL"])
        ch = np.interp(u, rails["uH"], rails["cH"])
        z = (1 - t) * zl + t * zh
        c = (1 - t) * cl + t * ch
        c_lo = int(np.floor(c.min())) - 2
        c_hi = int(np.ceil(c.max())) + 3
        if c_lo < 0 or c_hi > W:
            continue
        slab = np.asarray(strip[:, c_lo:c_hi], np.float32)
        fr = _bilinear(slab, z, c, c_lo)
        if progress and n_out % progress == 0:
            print(f"    cell {k}", flush=True)
        n_out += 1
        yield k, fr


# ─────────────────────────────── driver ───────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strip")
    ap.add_argument("--perfs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--video", default="film_perflock.mp4")
    ap.add_argument("--duration", type=float, default=9.0)
    ap.add_argument("--smooth-frames", type=float, default=1.0,
                    help="Rail smoothing sigma in FRAMES. The hole-finding "
                         "itself is sub-pixel (0.3 px against known shifts), but "
                         "the perf bands are rendered from extrapolated geometry "
                         "so the measured positions carry 4-18 px of real "
                         "per-frame scatter; this damps that. One winding is "
                         "~6.7 frames, so going much above ~1.5 starts eating "
                         "the once-per-turn distortion we are trying to remove.")
    ap.add_argument("--along", choices=["frameline", "perf"], default="perf",
                    help="Fiducial for the along-film rail. perf (default) "
                         "drifts only 0.32 frames over the reel once cycle slips "
                         "are removed; frameline is quieter per frame but its "
                         "series has gross errors that leave a 2.45-frame drift.")
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--fix-scale", action="store_true",
                    help="Also correct the across-film scale from the row "
                         "separation. OFF by default: the high perf row carries "
                         "17.5 px of per-frame scatter, nearly the size of the "
                         "sway itself, and taking it at face value asks for an "
                         "11 %% scale swing that would visibly pump the picture.")
    ap.add_argument("--fix-shear", action="store_true",
                    help="Also correct the shear from the frame-line tilt. OFF "
                         "by default and currently NOT trustworthy -- the "
                         "sub-band fit returns tens of degrees where the real "
                         "effect is under one.")
    ap.add_argument("--crop-mm", type=float, default=ISO_APERTURE_ACROSS,
                    help="Output width across the film in mm (ISO aperture is "
                         "10.26). Converted at the isotropic pitch/7.62 px per "
                         "mm -- see the scale note at the top of this module.")
    ap.add_argument("--full-width", action="store_true",
                    help="Output the whole film width including the perf rows.")
    ap.add_argument("--perf-phase", default="auto")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--no-rotate", action="store_true")
    ap.add_argument("--save-frames", action="store_true",
                    help="Also save the rectified frames as a .npy stack.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    strip = np.load(args.strip, mmap_mode="r")
    perfs = np.load(args.perfs)
    print(f"strip {strip.shape}  perfs {args.perfs}", flush=True)

    print("building rails:", flush=True)
    rails = build_rails(perfs, smooth_frames=args.smooth_frames,
                        along=args.along, min_score=args.min_score,
                        fix_scale=args.fix_scale, fix_shear=args.fix_shear)

    band_lo = perfs["band_lo"]; band_hi = perfs["band_hi"]
    pic = (int(band_lo[1]) + 60, int(band_hi[0]) - 60)
    # The cut goes on the DARK interframe gap.  Note the frame-line detector
    # locks to the BRIGHT bar, which is a stronger feature to measure position
    # with but sits half a pitch away from the boundary -- cutting at u=0 put
    # every cell half a frame out, the same half-pitch trap that makes
    # make_film_video need --invert-phase on this scan.  Taking the phase from
    # the folded profile's MINIMUM handles both rail modes without a flag.
    _, fold = calibrate_perf_phase(strip, rails, pic)
    phase = float(np.argmin(fold)) if args.perf_phase == "auto" \
        else float(args.perf_phase)
    print(f"  cut phase = {phase:.1f} px", flush=True)

    import imageio.v2 as imageio
    from PIL import Image

    samp = np.asarray(strip[::8, ::64], np.float32)
    lo_i, hi_i = np.percentile(samp, [1, 99])

    out, ks = [], []
    for k, fr in rectify_frames(strip, rails, phase, crop_mm=args.crop_mm,
                                full=args.full_width):
        ks.append(k); out.append(fr.astype(np.float32))
    print(f"rectified {len(out)} frames", flush=True)
    if not out:
        raise SystemExit("no frames produced")

    n = len(out)
    fps = max(6, int(round(n / args.duration)))
    order = list(range(n - 1, -1, -1)) if args.reverse else list(range(n))
    h0, w0 = out[0].shape
    fh, fw = (w0, h0) if not args.no_rotate else (h0, w0)
    H = args.height
    Wf = int(round(H * fw / fh)); Wf += Wf % 2

    path = os.path.join(args.out_dir, args.video)
    wr = imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                            macro_block_size=1)
    tiles, tile_idx = [], set(order[::max(1, n // 16)][:16])
    for i in order:
        fr = np.clip((out[i] - lo_i) / (hi_i - lo_i + 1e-6), 0, 1)
        if args.invert:
            fr = 1.0 - fr
        if not args.no_rotate:
            fr = np.rot90(fr)
        im = Image.fromarray((fr * 255).astype(np.uint8)).resize((Wf, H))
        wr.append_data(np.asarray(im))
        if i in tile_idx:
            tiles.append(np.asarray(im.resize((240, 180))))
    wr.close()
    print(f"wrote {path}  ({H}x{Wf}, {n / fps:.1f}s)", flush=True)

    cols = 8
    rows_n = int(np.ceil(len(tiles) / cols))
    mon = np.zeros((rows_n * 180, cols * 240), np.uint8)
    for j, t in enumerate(tiles):
        mon[(j // cols) * 180:(j // cols) * 180 + 180,
            (j % cols) * 240:(j % cols) * 240 + 240] = t
    Image.fromarray(mon).save(path + ".frames.png")

    if args.save_frames:
        np.save(os.path.join(args.out_dir, "frames.npy"), np.stack(out))

    np.savez(os.path.join(args.out_dir, "rect_geom.npz"),
             phase=phase, cells=np.array(ks), pitch=rails["pitch"],
             ZL=rails["ZL"], ZH=rails["ZH"],
             **{k: v for k, v in rails.items() if isinstance(v, np.ndarray)})
    _rail_diag(rails, args.out_dir, fold)


def _rail_diag(rails, out_dir, fold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fr = rails["frames"]
    fig, ax = plt.subplots(4, 1, figsize=(14, 13))
    ax[0].plot(fr, rails["zL_raw"] - rails["ZL"], ".", ms=3, alpha=.45,
               label="low row, raw")
    ax[0].plot(fr, rails["zH_raw"] - rails["ZH"], ".", ms=3, alpha=.45,
               label="high row, raw")
    ax[0].plot(fr, rails["sway"], "-", lw=1.4, color="k",
               label="sway applied (noise-weighted, outliers rejected)")
    ax[0].set_ylim(-120, 120)
    ax[0].set_title("across-film position of each perf row  =  THE SWAY (px)")
    ax[0].set_xlabel("frame"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    ax[1].plot(fr, rails["zH"] - rails["zL"] - (rails["ZH"] - rails["ZL"]))
    ax[1].set_title("across-film SCALE correction applied (px); flat = --fix-scale off")
    ax[1].set_xlabel("frame"); ax[1].grid(alpha=.3)

    ax[2].plot(fr, rails["cL"] - fr * rails["pitch"])
    ax[2].set_title("along-film rail: how far each frame sits from a constant-"
                    "pitch cut (px). This is the framing drift being removed.")
    ax[2].set_xlabel("frame"); ax[2].grid(alpha=.3)

    if fold is not None:
        ax[3].plot(fold)
        ax[3].axvline(np.argmin(fold), color="r", ls="--",
                      label="frame line (cut phase)")
        ax[3].set_title("picture-band column mean folded on the pitch, in u "
                        "-- the frame-line phase")
        ax[3].legend(); ax[3].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rails.png"), dpi=110)
    plt.close(fig)
    print(f"wrote {out_dir}/rails.png", flush=True)


if __name__ == "__main__":
    main()
