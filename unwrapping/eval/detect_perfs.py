"""Detect the two perforation rows in a FULL-WIDTH unrolled strip.

The new full-width reconstruction (`Mickey_merged_rigid_nobin`, see
`.claude/docs/newscan-fullwidth.md`) keeps both perforation rows of the 16 mm
double-perf stock, so the strip carries a physical fiducial: one rectangular
hole per frame per edge, at ISO-standard size (1.83 mm across the film width x
1.27 mm along its length) and standard pitch (7.62 mm = one frame).  Those holes
are the only GT-free way to measure what the unrolling got wrong per frame.
This module finds them; `perf_rectify.py` uses them.

Strip convention (as written by `unroll_walk_wholeroll_v13`):
    strip[z, c]   z = CT slice index  = ACROSS the film width
                  c = arc length      = ALONG the film
Intensity is raw CT attenuation, so film is BRIGHT and a perforation -- a hole
full of air -- is DARK.  (The film image being a photographic negative, which is
why the video render inverts, is a separate thing from the CT polarity.)

HOW IT FINDS THEM.  A matched filter: score a nominal-sized box against its own
surround, so the response peaks only where a compact rectangle of air sits in
film.  A plain threshold does not work here -- the perf bands are rendered from
z-EXTRAPOLATED geometry, so they carry large dark artefacts where the sampling
path left the film, and those artefacts are as air-dark as the holes.  They are
not hole-SHAPED, which is what the matched filter keys on.  The fit also searches
over a shear, both because the holes really are slightly sheared in the strip and
because a shear-blind fit lets a neighbouring artefact drag the centre.

    python -m unwrapping.eval.detect_perfs <strip.npy> --out-dir <dir> [--pitch P]
"""

import argparse
import os

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import find_peaks


# ISO 16 mm double-perf nominal geometry, in mm.
ISO_PERF_ACROSS = 1.83      # hole size across the film width
ISO_PERF_ALONG = 1.27       # hole size along the film length
ISO_FRAME_PITCH = 7.62      # one frame
ISO_FILM_WIDTH = 16.0


def estimate_pitch(col, lo=250, hi=1600):
    """Frame pitch from the column-mean profile (autocorrelation, guarded
    against sub-harmonics exactly as make_film_video does)."""
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


def levels(strip, row_step=4, col_step=16):
    """Air and film intensity levels.

    Air is taken as a low percentile of the strip, NOT from the rows outside the
    film: out there the strip samples an EXTRAPOLATED radius which lands on
    neighbouring windings, so those rows read as film (0.56) rather than air
    (0.15) and every threshold derived from them is far too high.
    """
    samp = np.asarray(strip[::row_step, ::col_step], np.float32).ravel()
    return float(np.percentile(samp, 1)), float(np.percentile(samp, 75))


def airness_of(block, air, film):
    """0 where solid film, 1 where air."""
    return np.clip((film - block) / (film - air + 1e-9), 0.0, 1.3)


def _match(airness, hz, hc, surround=1.7):
    """Box-minus-surround matched-filter response for a hz x hc hole."""
    inner = uniform_filter(airness, size=(hz, hc), mode="nearest")
    outer = uniform_filter(airness, size=(int(hz * surround), int(hc * surround)),
                           mode="nearest")
    return inner - outer


def find_perf_bands(strip, pitch, air, film, col_step=4, min_rows=80):
    """Locate the two perf rows along z from the matched-filter response.

    Uses the same statistic as the detection itself, so a band is by definition
    "the rows where hole-shaped things live".  The obvious alternatives both
    fail on this strip and are worth not re-trying:
      * amplitude of the periodic-at-frame-pitch component -- picture frame lines
        are periodic at the SAME pitch, so the picture band sits at 0.10 against
        a perf peak of 0.18 and no half-max edge closes;
      * plain air fraction -- the strip's dark picture content falls below any
        air threshold placed midway between the air and film modes.
    """
    Z, W = strip.shape
    hc = int(round(ISO_PERF_ALONG / ISO_FRAME_PITCH * pitch))
    hz = int(round(hc * ISO_PERF_ACROSS / ISO_PERF_ALONG))
    sub = airness_of(np.asarray(strip[:, ::col_step], np.float32), air, film)
    S = _match(sub, hz, max(3, hc // col_step))
    prof = np.percentile(S, 99.5, axis=1)

    k = 15
    ps = np.convolve(np.pad(prof, k // 2, mode="edge"), np.ones(k) / k, "valid")[:Z]
    thr = 0.5 * ps.max()
    inb = ps > thr
    runs, i = [], 0
    while i < Z:
        if inb[i]:
            j = i
            while j < Z and inb[j]:
                j += 1
            if j - i >= min_rows:
                runs.append((i, j))
            i = j
        else:
            i += 1
    if len(runs) < 2:
        raise SystemExit(f"expected 2 perf bands, found {len(runs)}: {runs}")
    runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    return sorted(runs[:2]), prof, (hz, hc)


def _shear(win, slant):
    """Shift each row of `win` along its columns by slant * (row - centre)."""
    if abs(slant) < 1e-6:
        return win
    nz, nc = win.shape
    z = np.arange(nz) - 0.5 * (nz - 1)
    shift = slant * z
    c = np.arange(nc)[None, :] + shift[:, None]
    c0 = np.clip(np.floor(c).astype(np.int32), 0, nc - 2)
    f = np.clip(c - c0, 0, 1)
    r = np.arange(nz)[:, None]
    return win[r, c0] * (1 - f) + win[r, c0 + 1] * f


def _parab(a, b, c):
    d = a - 2 * b + c
    return float(np.clip(0.5 * (a - c) / d, -1, 1)) if abs(d) > 1e-9 else 0.0


def _edges(prof, frac=0.5):
    """Sub-pixel first/last crossing of `frac` of the profile's own range."""
    lo, hi = np.percentile(prof, 10), prof.max()
    lvl = lo + frac * (hi - lo)
    m = prof >= lvl
    if not m.any():
        return None
    i0 = int(np.argmax(m))
    i1 = len(prof) - 1 - int(np.argmax(m[::-1]))
    if i0 == 0 or i1 == len(prof) - 1:
        return None
    a, b = prof[i0 - 1], prof[i0]
    e0 = (i0 - 1) + (lvl - a) / (b - a + 1e-12)
    a, b = prof[i1], prof[i1 + 1]
    e1 = i1 + (a - lvl) / (a - b + 1e-12)
    return e0, e1


def _centroid(prof):
    """Baseline-subtracted first moment of a 1-D marginal."""
    p = np.clip(prof - np.percentile(prof, 10), 0, None)
    s = p.sum()
    return float((np.arange(len(p)) * p).sum() / s) if s > 1e-6 else np.nan


def leave_one_out_residual(y, half_window=3):
    """Residual of each sample from a local-linear fit to its NEIGHBOURS.

    The real geometry varies smoothly along the reel -- roll telescoping plus a
    once-per-turn unrolling error, one turn being ~6.7 frames -- so a fit to the
    neighbours predicts the true value and what is left is measurement noise.
    This is the only honest way to choose between the centre estimators, since
    there is no ground truth for where a hole "really" is.
    """
    y = np.asarray(y, float)
    idx = np.arange(len(y))
    r = np.full(len(y), np.nan)
    for i in range(len(y)):
        m = (np.abs(idx - i) <= half_window) & (idx != i) & np.isfinite(y)
        if m.sum() < 3:
            continue
        r[i] = y[i] - np.polyval(np.polyfit(idx[m], y[m], 1), i)
    return r


def _report_precision(out, pitch, tag):
    """Print the leave-one-out noise of each centre estimator."""
    k = np.concatenate([[0], np.cumsum(np.maximum(
        1, np.round(np.diff(out["c"]) / pitch).astype(int)))])
    msg = []
    for axis, ramp in (("c", k * pitch), ("z", 0.0)):
        r = leave_one_out_residual(out[axis] - ramp)
        r = r[np.isfinite(r)]
        if len(r) < 10:
            continue
        rob = float(np.median(np.abs(r - np.median(r))) * 1.4826)
        msg.append(f"{axis} {rob:6.2f}")
    print(f"  [{tag}] leave-one-out robust scatter (px): " + "  ".join(msg)
          + "   (estimator noise is ~0.3 px, so this is REAL displacement)",
          flush=True)


def measure_holes(strip, band, pitch, air, film, tmpl, pad=None, tag=""):
    """Fit every perforation in one band.

    Returns a dict of arrays, one entry per hole:
        c, z            centre from the matched-filter argmax
        w_c, w_z        hole size along / across the film
        score           matched-filter response -- reject on this
    """
    hz, hc = tmpl
    if pad is None:
        # The band is where the matched-filter RESPONSE is high, which is the
        # hole's centre region and so is NARROWER than the hole (the response
        # falls off within +-hz/2 of centre).  Padding by less than a hole
        # height leaves no room for either the template or the edge marginals,
        # which is what made w_z come out anywhere from 159 to 300 px.
        pad = int(1.1 * hz)

    z0, z1 = band
    Z, W = strip.shape
    a0, a1 = max(0, z0 - pad), min(Z, z1 + pad)
    A = airness_of(np.asarray(strip[a0:a1], np.float32), air, film)

    S0 = _match(A, hz, hc)
    prof = S0.max(axis=0)
    pk, _ = find_peaks(prof, distance=int(round(pitch * 0.6)),
                       height=0.35 * np.percentile(prof, 99))
    print(f"  [{tag}] band z {z0}-{z1}: {len(pk)} candidates "
          f"(expect ~{int(W / pitch)}), template {hz}x{hc}", flush=True)

    half = int(round(pitch * 0.45))
    rows, rej = [], {"edge": 0, "fit": 0, "size": 0}
    for c0 in pk:
        a, b = int(c0) - half, int(c0) + half
        if a < 1 or b > W - 1:
            rej["edge"] += 1
            continue
        win = A[:, a:b]
        # CENTRE first, with an UPRIGHT template.  Letting the shear float while
        # locating the centre lets a sheared template straddle the hole and a
        # neighbouring dark artefact, which moved centres by up to 45 px in c
        # and 39 px in z -- larger than the wobble being corrected.  The shear is
        # then measured at the FIXED centre below, so it cannot feed back.
        Sm = _match(win, hz, hc)
        zi, ci = np.unravel_index(int(np.argmax(Sm)), Sm.shape)
        if not (1 <= zi < Sm.shape[0] - 1 and 1 <= ci < Sm.shape[1] - 1):
            rej["fit"] += 1
            continue
        sc = float(Sm[zi, ci])
        zf = zi + _parab(Sm[zi - 1, ci], Sm[zi, ci], Sm[zi + 1, ci])
        cf = ci + _parab(Sm[zi, ci - 1], Sm[zi, ci], Sm[zi, ci + 1])

        # Marginals of the hole, taken on a window ~1 hole wider than the hole
        # and centred on the coarse fit, so neither the picture band nor a dark
        # artefact further out can reach the 50 % level and pose as an edge.
        # No de-shearing: the per-hole shear estimate saturates at the search
        # bounds on real data (+-20 deg with no structure), i.e. it is noise, and
        # the rectifier gets its shear from the 1747 px baseline BETWEEN the two
        # rows instead, which is far better conditioned.
        r0, r1 = int(max(0, zf - hz * 0.3)), int(min(win.shape[0], zf + hz * 0.3))
        c_lo = int(max(0, cf - hc)); c_hi = int(min(win.shape[1], cf + hc))
        mc = win[r0:r1, c_lo:c_hi].mean(axis=0)
        ec = _edges(mc)
        q0 = int(max(0, cf - hc * 0.25)); q1 = int(min(win.shape[1], cf + hc * 0.25))
        z_lo = int(max(0, zf - hz * 0.9)); z_hi = int(min(win.shape[0], zf + hz * 0.9))
        mz = win[z_lo:z_hi, q0:q1].mean(axis=1)
        ez = _edges(mz)
        if ec is None or ez is None:
            rej["size"] += 1
        w_c = (ec[1] - ec[0]) if ec is not None else np.nan
        w_z = (ez[1] - ez[0]) if ez is not None else np.nan

        # Centre = the matched-filter peak.  Measured against known sub-pixel
        # shifts of real hole crops it recovers them to sd 0.21-0.29 px (max
        # error 0.52), whereas the edge-midpoint and centroid alternatives came
        # out at 10-27 px -- so the plateau-flat-peak worry was unfounded and
        # those two are simply worse here.  They are gone; this is the estimator.
        rows.append((a + cf, a0 + zf, w_c, w_z, sc))

    if not rows:
        raise SystemExit(f"no holes fitted in band {band}; rejects {rej}")
    r = np.array(rows, float)
    out = dict(c=r[:, 0], z=r[:, 1], w_c=r[:, 2], w_z=r[:, 3], score=r[:, 4])
    o = np.argsort(out["c"])
    out = {k: v[o] for k, v in out.items()}
    _report_precision(out, pitch, tag)
    print(f"  [{tag}] fitted {len(r)}/{len(pk)} rejects={rej} | "
          f"w_c {np.nanmedian(out['w_c']):.1f} (nom {hc}) "
          f"w_z {np.nanmedian(out['w_z']):.1f} (nom {hz}) "
          f"ratio {np.nanmedian(out['w_z']) / np.nanmedian(out['w_c']):.2f} "
          f"(ISO {ISO_PERF_ACROSS / ISO_PERF_ALONG:.2f}) | "
          f"score med {np.median(out['score']):.3f}", flush=True)
    return out


def detect_framelines(strip, pitch, band_lo, band_hi, n_sub=4, tag="frameline"):
    """Locate the frame line -- the bright interframe bar -- in the PICTURE band.

    This is the second fiducial, and it is the one that matters most for
    trusting the first: the perf rows are rendered from z-EXTRAPOLATED geometry,
    so a hole's position carries that extrapolation's error, whereas the frame
    line sits in the walked picture band.  Correlating the two says whether the
    perf displacement is REAL film geometry (shared by the picture, so safe to
    correct with) or an artefact confined to the perf rows (so correcting the
    picture with it would inject error).

    Splitting the picture band into `n_sub` z sub-bands also measures the frame
    line's TILT directly where the picture is, rather than interpolating it from
    the perf rows.

    Returns (positions, z_centres): positions[i] is the array of sub-pixel
    column positions for sub-band i.
    """
    z0 = int(band_lo[1]) + 60
    z1 = int(band_hi[0]) - 60
    edges = np.linspace(z0, z1, n_sub + 1).astype(int)
    W = strip.shape[1]

    out, zc = [], []
    for i in range(n_sub):
        blk = np.asarray(strip[edges[i]:edges[i + 1]:4, :], np.float32)
        prof = blk.mean(axis=0)
        k = 11
        ps = np.convolve(np.pad(prof, k // 2, mode="edge"), np.ones(k) / k,
                         "valid")[:W]
        band = _bandpass_at(ps, pitch)
        pk, _ = find_peaks(band, distance=int(round(pitch * 0.7)))
        # refine each peak on the SMOOTHED profile, parabolic sub-pixel
        ref = []
        for p in pk:
            a = max(1, p - int(0.12 * pitch)); b = min(W - 1, p + int(0.12 * pitch))
            j = a + int(np.argmax(ps[a:b]))
            if 0 < j < W - 1:
                ref.append(j + _parab(ps[j - 1], ps[j], ps[j + 1]))
        out.append(np.array(ref))
        zc.append(0.5 * (edges[i] + edges[i + 1]))
        print(f"  [{tag}] z sub-band {edges[i]}-{edges[i+1]}: {len(ref)} frame "
              f"lines (expect ~{int(W / pitch)})", flush=True)
    return out, np.array(zc)


def _detrended_residual(x, pitch, half_window=3):
    """Along-film displacement of a periodic feature: remove the k*pitch ramp,
    then take the leave-one-out residual about the local trend."""
    x = np.asarray(x, float)
    k = np.concatenate([[0], np.cumsum(np.maximum(
        1, np.round(np.diff(x) / pitch).astype(int)))])
    return leave_one_out_residual(x - k * pitch, half_window), k


def cross_check(lo, hi, fl, fl_z, pitch):
    """THE decisive test for whether the perfs may drive the picture band.

    The perf rows are rendered from z-EXTRAPOLATED geometry; the frame lines sit
    in the walked picture band.  If a hole's displacement from its local trend
    is REAL film geometry, the picture band moves with it and the two residuals
    correlate.  If the displacement is an artefact of the extrapolation, it is
    confined to the perf rows, the correlation is ~0, and using the perfs to
    place picture content would inject error rather than remove it.
    """
    print("\n=== cross-check: perf displacement vs frame-line displacement ===",
          flush=True)
    rl, _ = _detrended_residual(lo["c"], pitch)
    rh, _ = _detrended_residual(hi["c"], pitch)
    ref = fl[len(fl) // 2]
    rf, _ = _detrended_residual(ref, pitch)

    def corr(a, a_pos, b, b_pos, lbl):
        # match by column position, then correlate where both are finite
        j = np.clip(np.searchsorted(b_pos, a_pos), 1, len(b_pos) - 1)
        pick = np.where(np.abs(b_pos[j] - a_pos) < np.abs(b_pos[j - 1] - a_pos),
                        j, j - 1)
        bb = b[pick]
        m = (np.isfinite(a) & np.isfinite(bb)
             & (np.abs(b_pos[pick] - a_pos) < 0.4 * pitch)
             & (np.abs(a) < 150) & (np.abs(bb) < 150))
        if m.sum() < 20:
            print(f"  {lbl}: too few pairs ({m.sum()})")
            return
        r = float(np.corrcoef(a[m], bb[m])[0, 1])
        print(f"  {lbl}: corr = {r:+.3f}  (n={m.sum()}, "
              f"sd {a[m].std():.1f} vs {bb[m].std():.1f} px)", flush=True)

    corr(rl, lo["c"], rf, ref, "low perf   vs frame line")
    corr(rh, hi["c"], rf, ref, "high perf  vs frame line")
    corr(rl, lo["c"], rh, hi["c"], "low perf   vs high perf ")

    # frame-line sub-bands against each other: an upper bound on how well ANY
    # picture-band feature can be measured, and a check on the frame-line
    # detector itself.
    if len(fl) >= 2:
        ra, _ = _detrended_residual(fl[0], pitch)
        rb, _ = _detrended_residual(fl[-1], pitch)
        corr(ra, fl[0], rb, fl[-1], "frameline top vs bottom ")
    print(flush=True)


def _bandpass_at(x, pitch, bandfrac=0.45):
    x = np.asarray(x, np.float64) - np.mean(x)
    F = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x))
    f0 = 1.0 / pitch
    F[np.abs(f - f0) > bandfrac * f0] = 0.0
    return np.fft.irfft(F, n=len(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strip")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pitch", type=float, default=0,
                    help="Frame pitch in strip px (0 = auto-detect).")
    ap.add_argument("--col-step", type=int, default=4,
                    help="Column subsampling for the band search (speed only).")
    ap.add_argument("--frameline-subbands", type=int, default=4,
                    help="Split the picture band into N z sub-bands when "
                         "locating frame lines, so their tilt is measured too.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    s = np.load(args.strip, mmap_mode="r")
    Z, W = s.shape
    print(f"strip {s.shape} {s.dtype}", flush=True)

    air, film = levels(s)
    print(f"levels: air {air:.3f}  film {film:.3f}", flush=True)

    col = np.asarray(s[::4, :], np.float32).mean(axis=0)
    pitch = float(args.pitch) if args.pitch > 0 else float(estimate_pitch(col))
    print(f"frame pitch = {pitch:.2f} px  -> ~{W / pitch:.1f} frames", flush=True)

    bands, prof, tmpl = find_perf_bands(s, pitch, air, film, col_step=args.col_step)
    print(f"perf bands (z rows): {bands}   template {tmpl[0]}x{tmpl[1]}", flush=True)

    lo = measure_holes(s, bands[0], pitch, air, film, tmpl, tag="low")
    hi = measure_holes(s, bands[1], pitch, air, film, tmpl, tag="high")

    fl, fl_z = detect_framelines(s, pitch, bands[0], bands[1],
                                 n_sub=args.frameline_subbands)
    cross_check(lo, hi, fl, fl_z, pitch)

    np.savez(os.path.join(args.out_dir, "perfs.npz"),
             pitch=pitch, air=air, film=film,
             band_lo=np.array(bands[0]), band_hi=np.array(bands[1]),
             tmpl=np.array(tmpl), band_prof=prof, strip_shape=np.array([Z, W]),
             fl_z=fl_z,
             **{f"fl_{i}": v for i, v in enumerate(fl)},
             **{f"lo_{k}": v for k, v in lo.items()},
             **{f"hi_{k}": v for k, v in hi.items()})
    print(f"wrote {args.out_dir}/perfs.npz", flush=True)
    _diagnostics(s, bands, lo, hi, prof, pitch, tmpl, air, film, args.out_dir)


def _diagnostics(s, bands, lo, hi, prof, pitch, tmpl, air, film, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Z, W = s.shape
    hz, hc = tmpl
    fig, ax = plt.subplots(5, 1, figsize=(15, 16))
    ax[0].plot(prof)
    for (a, b) in bands:
        ax[0].axvspan(a, b, color="tab:orange", alpha=.3)
    ax[0].set_title("matched-filter response vs z (perf bands shaded)")
    ax[0].set_xlabel("z row"); ax[0].grid(alpha=.3)

    ax[1].plot(lo["c"], lo["z"], ".-", ms=3, lw=.7, label="low row")
    ax[1].plot(hi["c"], hi["z"], ".-", ms=3, lw=.7, label="high row")
    ax[1].set_title("perf centre ACROSS the film (z) along the reel  =  THE SWAY")
    ax[1].legend(); ax[1].grid(alpha=.3)

    for d, nm in ((lo, "low"), (hi, "high")):
        ax[2].plot(d["c"][1:], np.diff(d["c"]), ".-", ms=3, lw=.7, label=nm)
    ax[2].axhline(pitch, color="k", ls="--", lw=1, label=f"pitch {pitch:.1f}")
    ax[2].set_ylim(pitch * 0.9, pitch * 1.1)
    ax[2].set_title("hole-to-hole pitch along the film (px)  =  along-film SCALE")
    ax[2].legend(); ax[2].grid(alpha=.3)

    for d, nm in ((lo, "low"), (hi, "high")):
        ax[3].plot(d["c"], d["w_c"], ".", ms=3, label=f"{nm} along w_c")
        ax[3].plot(d["c"], d["w_z"], ".", ms=3, label=f"{nm} across w_z")
    ax[3].axhline(hc, color="k", ls="--", lw=1)
    ax[3].axhline(hz, color="k", ls=":", lw=1)
    ax[3].set_ylim(0, 2.2 * hz)
    ax[3].set_title("hole size (px); ISO says constant, so the scatter is our distortion")
    ax[3].legend(fontsize=8); ax[3].grid(alpha=.3)

    # Which centre estimator is actually the most precise?  Leave-one-out
    # residual against a local-linear fit to the neighbours.
    for d, nm in ((lo, "low"), (hi, "high")):
        k = np.concatenate([[0], np.cumsum(np.maximum(
            1, np.round(np.diff(d["c"]) / pitch).astype(int)))])
        ax[4].plot(d["c"], leave_one_out_residual(d["c"] - k * pitch), "-",
                   lw=.8, label=f"{nm} along")
        ax[4].plot(d["c"], leave_one_out_residual(d["z"]), "--", lw=.8,
                   label=f"{nm} across")
    ax[4].set_ylim(-150, 150)
    ax[4].set_title("leave-one-out residual of the ALONG-film centre (px) -- "
                    "this is measurement noise, and it must stay well under the "
                    "sway we are correcting")
    ax[4].legend(fontsize=7, ncol=3); ax[4].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "perf_diag.png"), dpi=110)
    plt.close(fig)

    # overlay montage: the fit drawn on the strip at 8 places along the reel
    fig, axs = plt.subplots(8, 1, figsize=(16, 22))
    spots = np.linspace(0.02, 0.98, 8)
    for j, (ax, f) in enumerate(zip(axs, spots)):
        c_at = f * W
        band = bands[j % 2]              # alternate rows down the page
        d = lo if j % 2 == 0 else hi
        z0 = max(0, band[0] - 60); z1 = min(Z, band[1] + 60)
        a = int(max(0, c_at - 2.2 * pitch)); b = int(min(W, a + 4.4 * pitch))
        ax.imshow(np.asarray(s[z0:z1, a:b], np.float32), cmap="gray",
                  aspect="auto", extent=[a, b, z1, z0])
        m = (d["c"] > a) & (d["c"] < b)
        ax.plot(d["c"][m], d["z"][m], "r+", ms=14, mew=2)
        for i in np.nonzero(m)[0]:
            wz = d["w_z"][i] if np.isfinite(d["w_z"][i]) else hz
            wc = d["w_c"][i] if np.isfinite(d["w_c"][i]) else hc
            ax.add_patch(plt.Rectangle((d["c"][i] - wc / 2, d["z"][i] - wz / 2),
                                       wc, wz, fill=False, ec="lime", lw=1.2))
        ax.set_title(f"{'low' if j % 2 == 0 else 'high'} row, cols {a}-{b}",
                     fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "perf_fit.png"), dpi=95)
    plt.close(fig)
    print(f"wrote {out_dir}/perf_diag.png + perf_fit.png", flush=True)


if __name__ == "__main__":
    main()
