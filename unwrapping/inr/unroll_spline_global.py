"""Whole-roll unroll via ONE global spline through the emulsion-pixel cloud.

New direction (2026-06-29). Both prior global-spline attempts and the per-winding
poly have a known failure each; this fixes all three:

  * `spline_base_unwrap.py` (abandoned) fed on RAYCAST-COUNTED centerlines (dashed
    emulsion -> miscounts -> wriggle) AND folded the once-per-turn eccentricity
    sinusoid into a single over-flexible r(phi) (re-learnt 35x -> inter-winding
    wriggle).
  * `unroll_wholeroll_simple.py` (current winner) fits each winding INDEPENDENTLY,
    so a winding missed at the seed ray is never fit (skips) and nothing enforces
    inner->outer ordering (artifacts).

Idea here: fit ALL emulsion (class 2) pixels at once as ONE continuous spiral,
decoupling the two physical effects:

    r(theta, k) = base(phi) + ecc(theta),   phi = (theta - seam) + 2*pi*k

  - base(phi): smooth global spline = the "middle of the emulsion bands" curve.
    Low DOF (eccentricity is removed into ecc), so it cannot wriggle and spans all
    windings as one curve -> no skipped windings, ordered inner->outer by design.
  - ecc(theta): a few Fourier harmonics in azimuth, SHARED across all windings
    (the off-center / elliptical spool). --ecc-harmonics 0 disables it (the naive
    single-spline, for comparison).

Winding assignment is by FIT not COUNT (ICP/EM): seed base from the continuous
film-band cache, then iterate {assign each pixel to the nearest base(phi_k)+ecc,
refit base + ecc robustly}.

Reads the cached per-anchor emulsion cloud (wholeroll_cache.npz: PHI, RR, centers,
seam) + the film-band cache (seeds), so NO 40-min seg reload.

Modes:
  --mode fit     (default) fit on selected anchors, write diagnostics, exit.
  --mode render  fit every anchor, interpolate across z, render all raw-CT slices.

Usage (cheap fit/diagnostic, from cache):
  python -m unwrapping.inr.unroll_spline_global --mode fit \
      --cache unwrapping/inr/results/arc_simple/wholeroll_v2/wholeroll_cache.npz \
      --filmband-cache unwrapping/inr/results/ring_track/anchor_filmband_r2880.npz \
      --ct-dir 01_Mickey_hdf --out-dir <out> --ecc-harmonics 4
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.inr.sanity_winding_strip import bilinear_sample
from unwrapping.inr.unroll_continuous import (
    load_ct, norm_slice, discover_ct_slices,
)
from unwrapping.inr.unroll_wholeroll_simple import filmband_seeds
from unwrapping.inr.fit_arc_simple import robust_polyfit

TWO_PI = 2.0 * math.pi


class PolyBase:
    """Low-degree polynomial base(phi): spline-compatible (call + derivative).

    A global low-order poly cannot wriggle and is monotone over the roll's phi
    range -> windings are strictly ordered and pitch-separated by construction
    (no overlap, no skip). The 'sharply tuned-down sensitivity' base."""

    def __init__(self, coef):
        self.coef = coef

    def __call__(self, phi):
        return np.polyval(self.coef, np.asarray(phi, float))

    def derivative(self):
        d = np.polyder(self.coef)
        return lambda phi: np.polyval(d, np.asarray(phi, float))


# ───────────────────────── model pieces ─────────────────────────
def robust_fourier(theta, resid, harmonics, iters=3):
    """IRLS Fourier fit ecc(theta) = sum_{n=1..H} a_n cos(n th) + b_n sin(n th).

    No constant term (absorbed by base). Shared across all windings (function of
    absolute azimuth only). Returns coef vector [a1,b1,a2,b2,...] (len 2H)."""
    if harmonics <= 0 or theta.size == 0:
        return np.zeros(0)
    cols = []
    for n in range(1, harmonics + 1):
        cols.append(np.cos(n * theta)); cols.append(np.sin(n * theta))
    A = np.stack(cols, axis=1)
    w = np.ones_like(resid)
    coef = np.zeros(A.shape[1])
    for _ in range(iters):
        WA = A * w[:, None]
        coef, *_ = np.linalg.lstsq(WA, resid * w, rcond=None)
        r = resid - A @ coef
        s = 1.4826 * np.median(np.abs(r)) + 1e-6
        w = 1.0 / (1.0 + (r / (2.0 * s)) ** 2)
    return coef


def eval_fourier(theta, coef):
    if coef.size == 0:
        return np.zeros_like(np.asarray(theta, dtype=float))
    H = coef.size // 2
    out = np.zeros_like(np.asarray(theta, dtype=float))
    for n in range(1, H + 1):
        out += coef[2 * (n - 1)] * np.cos(n * theta)
        out += coef[2 * (n - 1) + 1] * np.sin(n * theta)
    return out


def fit_base_spline(phi, r, smooth_px, k=3, grid_step=0.002):
    """Smoothing spline base(phi). Duplicate/near-duplicate phi averaged onto a
    fine grid (UnivariateSpline needs strictly increasing x). s scales as
    (smooth_px^2)*N so smooth_px is the target per-point RMS deviation (px)."""
    key = np.round(phi / grid_step).astype(np.int64)
    order = np.argsort(key)
    key, phi, r = key[order], phi[order], r[order]
    uk, idx = np.unique(key, return_inverse=True)
    rsum = np.zeros(uk.size); cnt = np.zeros(uk.size)
    np.add.at(rsum, idx, r); np.add.at(cnt, idx, 1.0)
    phi_u = uk * grid_step
    r_u = rsum / np.maximum(cnt, 1.0)
    s = (smooth_px ** 2) * phi_u.size
    return UnivariateSpline(phi_u, r_u, k=k, s=s)


def _fit_base(phi, r, base_type, smooth_px, degree):
    if base_type == "poly":
        return PolyBase(robust_polyfit(phi, r, degree=degree, iters=3))
    return fit_base_spline(phi, r, smooth_px)


def fit_global_spiral(phi_pix, r_pix, seam, seeds, smooth_px, ecc_harmonics,
                      gate_px=11.0, icp_iters=5, base_type="poly", base_degree=4):
    """ICP fit of base(phi) + ecc(theta) to the emulsion cloud.

    phi_pix: wrapped phase in [0, 2pi); r_pix: radius; theta_abs = seam + phi_pix.
    seeds: per-winding seed radii (sorted, at azimuth opposite the seam, phi=pi).

    base(phi) is a SMOOTH monotone trend (low-degree poly, or heavily-smoothed
    spline) -> windings strictly ordered & pitch-separated (no overlap/skip).
    ecc(theta) is a SHARED Fourier eccentricity applied identically to every
    winding -> keeps the constant pitch separation while landing on the bands.
    Returns (base, ecc_coef, phi_assigned, keep_mask, kstar, info).
    """
    nw = len(seeds)
    theta = seam + phi_pix
    # seed base: linear (Archimedean) through (pi + 2pi k, seed_k)
    seed_phi = math.pi + TWO_PI * np.arange(nw)
    A = np.vstack([np.ones(nw), seed_phi]).T
    (c0, c1), *_ = np.linalg.lstsq(A, np.asarray(seeds, float), rcond=None)
    base = lambda p: c0 + c1 * np.asarray(p, float)        # noqa: E731 (seed only)
    ecc_coef = np.zeros(0)

    ks = np.arange(nw)
    last = None
    for it in range(icp_iters):
        # assign each pixel to the winding k minimizing |r - base(phi_k) - ecc|
        phi_cand = phi_pix[:, None] + TWO_PI * ks[None, :]    # (N, nw)
        pred = base(phi_cand) + eval_fourier(theta, ecc_coef)[:, None]
        resid_all = np.abs(r_pix[:, None] - pred)
        kstar = np.argmin(resid_all, axis=1)
        best = resid_all[np.arange(r_pix.size), kstar]
        keep = best < gate_px
        phi_asg = phi_pix + TWO_PI * kstar
        # refit base on (phi, r - ecc); then ecc on residual (r - base)
        e = eval_fourier(theta[keep], ecc_coef)
        base = _fit_base(phi_asg[keep], r_pix[keep] - e, base_type, smooth_px,
                         base_degree)
        if ecc_harmonics > 0:
            res = r_pix[keep] - base(phi_asg[keep])
            ecc_coef = robust_fourier(theta[keep], res, ecc_harmonics)
        last = (phi_asg, keep, kstar)

    phi_asg, keep, kstar = last
    pred_keep = base(phi_asg[keep]) + eval_fourier(theta[keep], ecc_coef)
    rms = float(np.sqrt(np.mean((r_pix[keep] - pred_keep) ** 2)))
    nw_used = int(np.unique(kstar[keep]).size)
    info = dict(rms=rms, n_kept=int(keep.sum()), n_total=int(keep.size),
                nw=nw, nw_used=nw_used,
                ecc_amp=float(np.sqrt(np.mean(eval_fourier(theta, ecc_coef) ** 2)))
                if ecc_harmonics > 0 else 0.0)
    return base, ecc_coef, phi_asg, keep, kstar, info


def fit_winding_curve(ph_m, r_m, eps, skel_fn, fit_type="spline", degree=6,
                      smooth_px=2.5, clip_px=11.0, n_bins=180):
    """One winding's curve r(phi) over the turn, returned as a callable r=f(phi).

    Hugs its own emulsion band but is HARD-CLIPPED to the smooth global skeleton
    +-clip_px, so it physically cannot swoop into a neighbour band where the
    assignment is contaminated / the emulsion is dashed.

    fit_type:
      'poly'   — robust IRLS polynomial (stiff; whole-turn degree-`degree`).
      'spline' — TIGHT hug: robust median emulsion radius per fine angular bin
                 (centerline); bins far from the skeleton or off the robust trend
                 are rejected; low-smoothing UnivariateSpline through the rest.
    """
    if fit_type == "poly":
        coef = robust_polyfit(ph_m, r_m, degree=degree, iters=3)
        lo, hi = eps, TWO_PI - eps

        def f(p, c=coef):
            pp = np.clip(np.mod(p, TWO_PI), lo, hi)
            sk = skel_fn(pp)
            return np.clip(np.polyval(c, pp), sk - clip_px, sk + clip_px)
        return f

    edges = np.linspace(eps, TWO_PI - eps, n_bins + 1)
    cen = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(ph_m, edges) - 1, 0, n_bins - 1)
    rb = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = idx == b
        if int(sel.sum()) >= 3:
            rb[b] = np.median(r_m[sel])
    good = ~np.isnan(rb)
    # reject bins that disagree with the smooth skeleton by > clip (contamination)
    good &= np.abs(rb - skel_fn(cen)) <= clip_px
    if int(good.sum()) < 10:
        return None
    xb, yb = cen[good], rb[good]
    s = (smooth_px ** 2) * xb.size
    spl = UnivariateSpline(xb, yb, k=3, s=s)
    # one robust pass: drop bins far from the spline (outliers -> swoops)
    res = yb - spl(xb)
    mad = 1.4826 * np.median(np.abs(res)) + 1e-6
    ok = np.abs(res) <= 3.0 * mad
    if ok.sum() >= 10 and ok.sum() < xb.size:
        xb, yb = xb[ok], yb[ok]
        spl = UnivariateSpline(xb, yb, k=3, s=(smooth_px ** 2) * xb.size)
    x0, x1 = float(xb[0]), float(xb[-1])

    def f(p, spl=spl, x0=x0, x1=x1):
        pp = np.clip(np.mod(p, TWO_PI), x0, x1)
        sk = skel_fn(pp)
        return np.clip(spl(pp), sk - clip_px, sk + clip_px)
    return f


def refine_per_winding(phi_pix, r_pix, kstar, keep, nw, eps, base, ecc_coef, seam,
                       degree=6, fit_type="spline", smooth_px=2.5, clip_px=11.0):
    """Per-winding independent curve r_k(phi) on the GLOBAL skeleton's assignment.

    The global base+ecc fit assigns each emulsion pixel a winding kstar (robust,
    ordered, no skips). Each winding is then fit on ITS OWN pixels, hugging that
    band's shape, but hard-clipped to the skeleton so it can't cross neighbours.
    Returns a list of callables r=f(phi) (or None) indexed by winding.
    """
    ph = np.mod(phi_pix, TWO_PI)
    curves = [None] * nw
    for k in range(nw):
        def skel_fn(p, k=k):
            pp = np.mod(p, TWO_PI)
            return base(pp + TWO_PI * k) + eval_fourier(seam + pp, ecc_coef)
        m = keep & (kstar == k) & (ph > eps) & (ph < TWO_PI - eps)
        if int(m.sum()) >= degree + 10:
            curves[k] = fit_winding_curve(ph[m], r_pix[m], eps, skel_fn,
                                          fit_type=fit_type, degree=degree,
                                          smooth_px=smooth_px, clip_px=clip_px)
    return curves


def eval_winding(curve, phi):
    """Per-winding curve r(phi); `curve` is a callable f(phi) or a poly-coef array."""
    if callable(curve):
        return curve(phi)
    return np.polyval(curve, np.mod(phi, TWO_PI))


# ───────────────────────── diagnostics ─────────────────────────
def diag_anchor(phi_pix, r_pix, seam, base, ecc_coef, phi_asg, keep, kstar, info,
                eps, out_prefix, ct_img=None, cx=None, cy=None, tag=""):
    theta = seam + phi_pix
    phimax = float(phi_asg[keep].max())
    phig = np.linspace(eps, phimax - eps, 4000)

    # (1) radius-vs-phi cloud + base curve (eccentricity removed) ----------------
    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.scatter(phi_asg[keep] / TWO_PI, (r_pix[keep] - eval_fourier(theta[keep], ecc_coef)),
               s=1, alpha=0.12, color="gray", label="emulsion px (ecc removed)")
    ax.plot(phig / TWO_PI, base(phig), "r-", lw=1.4, label="base(phi) spline")
    ax.set_xlabel("winding number (phi / 2pi)"); ax.set_ylabel("radius (px)")
    ax.set_title(f"{tag}: base spline through emulsion cloud — "
                 f"{info['nw_used']}/{info['nw']} windings, RMS {info['rms']:.2f}px, "
                 f"ecc_amp {info['ecc_amp']:.1f}px")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout(); plt.savefig(out_prefix + "_radius_fit.png", dpi=140); plt.close()

    # (2) full spiral curve overlaid on the CT slice -----------------------------
    if ct_img is not None:
        # dense curve so a single turn draws smoothly (4000 pts over 33 turns = ~120/turn)
        phidense = np.linspace(eps, phimax - eps, int(phimax / TWO_PI) * 400)
        r_curve = base(phidense) + eval_fourier(seam + phidense, ecc_coef)
        th = seam + phidense
        xs = cx + r_curve * np.cos(th); ys = cy + r_curve * np.sin(th)
        disp = norm_slice(ct_img)
        fig, ax = plt.subplots(1, 1, figsize=(11, 11))
        ax.imshow(disp, cmap="gray")
        ax.plot(xs, ys, "-", color="red", lw=0.5)
        ax.plot(cx, cy, "c+", ms=12)
        ax.set_title(f"{tag}: global spiral (all windings, one curve)")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout(); plt.savefig(out_prefix + "_overlay.png", dpi=130); plt.close()

        # (3) THREE individual windings, each a continuous 1-turn line in its own
        #     colour (no masks-with-gaps -> no spurious radial connectors; only 3
        #     rings -> no aliasing). Decisive: does each follow one CT band cleanly?
        nwind = int(phimax / TWO_PI)
        fig, ax = plt.subplots(1, 1, figsize=(11, 11))
        ax.imshow(disp, cmap="gray")
        for kk, col in zip([max(2, nwind // 4), nwind // 2, 3 * nwind // 4],
                           ["red", "yellow", "lime"]):
            pp = np.linspace(kk * TWO_PI, (kk + 1) * TWO_PI, 2000)
            pp = pp[(pp >= eps) & (pp <= phimax - eps)]
            rr = base(pp) + eval_fourier(seam + pp, ecc_coef)
            tt = seam + pp
            ax.plot(cx + rr * np.cos(tt), cy + rr * np.sin(tt), "-", color=col,
                    lw=1.3, label=f"winding {kk}")
        ax.plot(cx, cy, "c+", ms=12); ax.legend(loc="upper right", fontsize=10)
        ax.set_title(f"{tag}: 3 single windings, each one continuous turn")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout(); plt.savefig(out_prefix + "_3wind.png", dpi=150)
        plt.close()

        # (4) HIGH-RES ZOOM: curve (red line) vs emulsion px (cyan) on a top wedge.
        wmid = math.radians(-90.0)
        wband = math.radians(16.0)
        sel = np.abs(np.mod(theta - wmid + math.pi, TWO_PI) - math.pi) < wband
        if sel.sum() > 50:
            exs = cx + r_pix[sel] * np.cos(theta[sel])
            eys = cy + r_pix[sel] * np.sin(theta[sel])
            x0, x1 = max(0, int(exs.min()) - 20), int(exs.max()) + 20
            y0, y1 = max(0, int(eys.min()) - 20), int(eys.max()) + 20
            fig, ax = plt.subplots(1, 1, figsize=(13, 13 * (y1 - y0) / max(1, x1 - x0)))
            ax.imshow(disp[y0:y1, x0:x1], cmap="gray")
            ax.scatter(exs - x0, eys - y0, s=2, c="cyan", alpha=0.4, label="emulsion px")
            # plot the curve per winding (no cross-winding connector lines)
            phic = np.mod(wmid - seam, TWO_PI)        # wedge centre in phi_pix
            for kk in range(int(phimax / TWO_PI) + 1):
                pp = phic + kk * TWO_PI + np.linspace(-wband, wband, 120)
                pp = pp[(pp >= eps) & (pp <= phimax - eps)]
                if pp.size < 2:
                    continue
                rr = base(pp) + eval_fourier(seam + pp, ecc_coef)
                tt = seam + pp
                ax.plot(cx + rr * np.cos(tt) - x0, cy + rr * np.sin(tt) - y0,
                        "-", color="red", lw=0.9)
            ax.plot([], [], "-", color="red", lw=0.9, label="fitted spline")
            ax.set_xlim(0, x1 - x0); ax.set_ylim(y1 - y0, 0)
            ax.legend(loc="upper right", fontsize=10)
            ax.set_title(f"{tag}: ZOOM curve(red) vs emulsion(cyan)")
            ax.set_xticks([]); ax.set_yticks([])
            plt.tight_layout(); plt.savefig(out_prefix + "_zoom.png", dpi=150)
            plt.close()

    return info


def ring_check(base, ecc_coef, seam, FB_anchor, n_rays, phimax, eps, out_path,
               tag=""):
    """Predicted ring radii vs TRUE film bands at several azimuths.

    Directly measures skips (a film band with NO predicted ring nearby) and
    overlaps (two predicted rings on ONE band). For 4 azimuths: predicted
    r_k = base(phic+2pi k)+ecc; true = sorted film bands (FB) at that ray.
    Returns dict(skips, overlaps, match_rms)."""
    nw = int(phimax / TWO_PI)
    azis = np.array([-90.0, -20.0, 110.0, 200.0])          # deg, avoid seam ~53.5
    fig, axes = plt.subplots(1, len(azis), figsize=(4 * len(azis), 7), sharey=True)
    tot_skip = tot_ovl = 0; resids = []
    for ax, ad in zip(axes, azis):
        th0 = math.radians(ad)
        ray = int(round(np.mod(th0, TWO_PI) / TWO_PI * n_rays)) % n_rays
        bands = np.sort(FB_anchor[ray][~np.isnan(FB_anchor[ray])])
        phic = np.mod(th0 - seam, TWO_PI)
        kk = np.arange(nw)
        pred = base(phic + TWO_PI * kk) + eval_fourier(np.full(nw, th0), ecc_coef)
        pred = pred[(pred > 0)]
        # match each true band to nearest predicted; pitch ~ median band spacing
        pitch = float(np.median(np.diff(bands))) if bands.size > 1 else 22.0
        # only score INTERIOR bands the prediction is meant to cover (exclude the
        # degenerate inner tongue below pred range + the thin outer tail above it)
        if pred.size:
            lo_cut, hi_cut = pred.min() - 0.5 * pitch, pred.max() + 0.5 * pitch
            score_bands = bands[(bands >= lo_cut) & (bands <= hi_cut)]
        else:
            score_bands = bands
        # remove the ~constant emulsion-vs-filmband offset before thresholding
        if pred.size and score_bands.size:
            near = np.array([pred[np.argmin(np.abs(pred - b))] - b for b in score_bands])
            off = float(np.median(near))
        else:
            off = 0.0
        skip = 0
        for b in score_bands:
            d = np.min(np.abs(pred - (b + off))) if pred.size else 1e9
            if d > 0.5 * pitch:
                skip += 1
            else:
                resids.append(d)
        # overlap: predicted rings closer than 0.4*pitch to each other
        ovl = int(np.sum(np.diff(np.sort(pred)) < 0.4 * pitch)) if pred.size > 1 else 0
        tot_skip += skip; tot_ovl += ovl
        ax.plot(np.zeros_like(bands), bands, "o", mfc="none", mec="k", ms=9,
                label="film band (true)")
        ax.plot(np.ones_like(pred) * 0.3, pred, "rx", ms=8, label="predicted ring")
        for b in bands:
            ax.hlines(b, -0.1, 0.4, color="0.8", lw=0.6, zorder=0)
        ax.set_xlim(-0.3, 0.7); ax.set_xticks([])
        ax.set_title(f"{ad:.0f}deg\nskip {skip} ovl {ovl}", fontsize=9)
    axes[0].set_ylabel("radius (px)"); axes[0].legend(fontsize=8, loc="upper left")
    mrms = float(np.sqrt(np.mean(np.square(resids)))) if resids else 0.0
    fig.suptitle(f"{tag}: rings vs true bands — total skips {tot_skip}, "
                 f"overlaps {tot_ovl}, match RMS {mrms:.1f}px", fontsize=11)
    plt.tight_layout(); plt.savefig(out_path, dpi=130); plt.close()
    return dict(skips=tot_skip, overlaps=tot_ovl, match_rms=mrms)


# ───────────────────────── arc-length / render ─────────────────────────
def arclen_phi(base, ecc_coef, seam, eps, phimax):
    phid = np.linspace(eps, phimax - eps, 40000)
    r = base(phid) + eval_fourier(seam + phid, ecc_coef)
    dr = base.derivative()(phid) + np.gradient(eval_fourier(seam + phid, ecc_coef), phid)
    ds = np.sqrt(r ** 2 + dr ** 2)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(phid))])
    return phid, s, float(s[-1])


def load_anchor_cloud(cache):
    d = np.load(cache, allow_pickle=True)
    return (np.asarray(d["z_anchor"], float), list(d["cx_a"]), list(d["cy_a"]),
            list(d["phi"]), list(d["r"]), float(d["seam"]))


def load_filmband(path):
    fb = np.load(path)
    return fb["FB"], int(fb["n_rays"])


def raw_max_seeds(FB, n_rays, pct=99.5):
    """True winding count + seed radii from the most-complete RAW film-band ray.

    `filmband_seeds` INFILLS missing windings, which over-adds a spurious winding
    when a legitimately wide spacing (~1.5x median from tension variation) looks
    like a gap -> one curve too many. Instead: scan every ray of every anchor and
    take the ray with the most RAW bands (no infill). The ray where the real
    windings are maximally air-separated gives the exact count. A high percentile
    (not the absolute max) guards against a noise-split ray over-counting.
    Returns (sorted seed radii, count).
    """
    counts = np.array([int(np.sum(~np.isnan(FB[a, r])))
                       for a in range(FB.shape[0]) for r in range(n_rays)])
    target = int(np.floor(np.percentile(counts, pct)))
    best = None
    for a in range(FB.shape[0]):
        for r in range(n_rays):
            bands = FB[a, r][~np.isnan(FB[a, r])]
            if bands.size == target:
                best = np.sort(bands); break
        if best is not None:
            break
    return best, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="wholeroll_cache.npz (emul cloud)")
    ap.add_argument("--filmband-cache", required=True, help="anchor_filmband_*.npz")
    ap.add_argument("--ct-dir", default=None, help="raw-CT dir (overlays / render)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mode", choices=["fit", "render"], default="fit")
    ap.add_argument("--ecc-harmonics", type=int, default=6,
                    help="Fourier harmonics for ecc(theta); 0 = naive single spline.")
    ap.add_argument("--base-type", choices=["poly", "spline"], default="poly",
                    help="poly = global low-degree (monotone, no wriggle); "
                         "spline = heavily-smoothed UnivariateSpline.")
    ap.add_argument("--base-degree", type=int, default=4,
                    help="Degree of the global poly base (base-type poly).")
    ap.add_argument("--base-smooth-px", type=float, default=20.0,
                    help="Target per-point RMS (px) of base spline (base-type spline).")
    ap.add_argument("--gate-px", type=float, default=11.0)
    ap.add_argument("--icp-iters", type=int, default=5)
    ap.add_argument("--refine-degree", type=int, default=6,
                    help="Per-winding refine poly degree (refine-type poly).")
    ap.add_argument("--refine-type", choices=["spline", "poly"], default="spline",
                    help="Per-winding curve: 'spline' (tight hug via median-bin + "
                         "low-smoothing spline) or 'poly'. The global fit is only "
                         "the skeleton; each winding hugs its own emulsion band.")
    ap.add_argument("--refine-smooth-px", type=float, default=2.0,
                    help="Per-winding spline smoothing (target per-bin RMS px); "
                         "smaller = tighter hug.")
    ap.add_argument("--count-pct", type=float, default=99.5,
                    help="Percentile of per-ray raw band counts used as the true "
                         "winding count (guards against a noise-split ray).")
    ap.add_argument("--refine-clip-px", type=float, default=8.0,
                    help="Per-winding curve hard-clip to skeleton +-this (px). "
                         "Decoupled from the ICP assignment gate.")
    ap.add_argument("--seam-exclude-deg", type=float, default=8.0)
    ap.add_argument("--seed-ray-halfdeg", type=float, default=6.0)
    ap.add_argument("--fit-anchors", default="",
                    help="Comma z-values to diagnose in fit mode (default: clean+noisy).")
    ap.add_argument("--z-step", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    eps = math.radians(args.seam_exclude_deg)

    z_anchor, cx_a, cy_a, PHI, RR, seam = load_anchor_cloud(args.cache)
    FB, n_rays = load_filmband(args.filmband_cache)
    n = len(z_anchor)
    rayi = int(round((np.mod(seam + math.pi, TWO_PI)) / TWO_PI * n_rays)) % n_rays
    print(f"{n} anchors; seam={math.degrees(seam):.1f}deg; FB {FB.shape}; "
          f"ecc_harmonics={args.ecc_harmonics} base_smooth={args.base_smooth_px}px")
    if FB.shape[0] != n:
        print(f"  WARNING: FB anchors {FB.shape[0]} != cloud anchors {n}")

    def fit_one(i):
        seeds = filmband_seeds(FB[i], rayi, n_rays, half_deg=args.seed_ray_halfdeg)
        return fit_global_spiral(
            np.asarray(PHI[i], float), np.asarray(RR[i], float), seam, seeds,
            args.base_smooth_px, args.ecc_harmonics,
            gate_px=args.gate_px, icp_iters=args.icp_iters,
            base_type=args.base_type, base_degree=args.base_degree)

    if args.mode == "fit" and args.fit_anchors == "all":
        # Audit EVERY anchor: skips/overlaps (ring-check, numbers + image), no
        # heavy CT overlays. Prints a summary table so we can confirm zero skips
        # across the whole roll before rendering.
        print("  SCAN ALL anchors (skips/overlaps audit):")
        rows = []
        for i in range(n):
            base, ecc, phi_asg, keep, kstar, info = fit_one(i)
            zi = int(z_anchor[i]); phimax = float(phi_asg[keep].max())
            rc = ring_check(base, ecc, seam, FB[i], n_rays, phimax, eps,
                            os.path.join(args.out_dir, f"a{i:02d}_z{zi}_ringcheck.png"),
                            tag=f"z{zi}")
            rows.append((zi, info["nw_used"], info["nw"], rc["skips"],
                         rc["overlaps"], rc["match_rms"], info["rms"]))
            print(f"    z{zi:>4} a{i:02d}: {info['nw_used']}/{info['nw']}w  "
                  f"skips {rc['skips']}  overlaps {rc['overlaps']}  "
                  f"matchRMS {rc['match_rms']:.1f}  fitRMS {info['rms']:.1f}", flush=True)
        ts = sum(r[3] for r in rows); to = sum(r[4] for r in rows)
        worst = sorted(rows, key=lambda r: -r[3])[:5]
        print(f"  TOTAL across {n} anchors: skips {ts}, overlaps {to}")
        print(f"  worst-skip anchors: {[(r[0], r[3]) for r in worst]}")
        print(f"Done (scan-all). Ring-checks in {args.out_dir}")
        return

    if args.mode == "fit":
        if args.fit_anchors:
            want = [float(x) for x in args.fit_anchors.split(",")]
            idxs = [int(np.argmin(np.abs(z_anchor - w))) for w in want]
        else:
            idxs = sorted({int(np.argmin(np.abs(z_anchor - 846))),
                           int(np.argmin(np.abs(z_anchor - 1605)))})
        z_to_path = discover_ct_slices(args.ct_dir) if args.ct_dir else {}
        for i in idxs:
            base, ecc, phi_asg, keep, kstar, info = fit_one(i)
            zi = int(z_anchor[i])
            print(f"  anchor z{zi} (a{i}): {info['nw_used']}/{info['nw']} windings, "
                  f"kept {info['n_kept']}/{info['n_total']}, RMS {info['rms']:.2f}px, "
                  f"ecc_amp {info['ecc_amp']:.1f}px", flush=True)
            ct_img = cx = cy = None
            if z_to_path:
                za = min(z_to_path, key=lambda zz: abs(zz - zi))
                ct_img = load_ct(z_to_path[za]); cx, cy = cx_a[i], cy_a[i]
            diag_anchor(np.asarray(PHI[i], float), np.asarray(RR[i], float), seam,
                        base, ecc, phi_asg, keep, kstar, info, eps,
                        os.path.join(args.out_dir, f"a{i:02d}_z{zi}"),
                        ct_img=ct_img, cx=cx, cy=cy, tag=f"z{zi}")
            phimax = float(phi_asg[keep].max())
            rc = ring_check(base, ecc, seam, FB[i], n_rays, phimax, eps,
                            os.path.join(args.out_dir, f"a{i:02d}_z{zi}_ringcheck.png"),
                            tag=f"z{zi}")
            print(f"    ring-check: skips {rc['skips']}, overlaps {rc['overlaps']}, "
                  f"match RMS {rc['match_rms']:.1f}px", flush=True)
        print(f"Done (fit). Diagnostics in {args.out_dir}")
        return

    # ── render mode ──
    # FIXED winding count from the most-complete anchor (one curve per emulsion
    # band everywhere); per anchor: robust global skeleton (assignment, no skips)
    # -> per-winding refine (each winding hugs its own band). Render each winding
    # over its own turn (arc-length cols) and CONCATENATE inner->outer.
    seeds_g, nw = raw_max_seeds(FB, n_rays, pct=args.count_pct)
    print(f"  fixed nw={nw} (raw most-complete ray, pct={args.count_pct}); "
          f"refine {args.refine_type} smooth {args.refine_smooth_px}")

    COEFS = [[None] * n for _ in range(nw)]               # COEFS[k][i]
    for i in range(n):
        base, ecc, phi_asg, keep, kstar, info = fit_global_spiral(
            np.asarray(PHI[i], float), np.asarray(RR[i], float), seam, seeds_g,
            args.base_smooth_px, args.ecc_harmonics, gate_px=args.gate_px,
            icp_iters=args.icp_iters, base_type=args.base_type,
            base_degree=args.base_degree)
        ci = refine_per_winding(np.asarray(PHI[i], float), np.asarray(RR[i], float),
                                kstar, keep, nw, eps, base, ecc, seam,
                                degree=args.refine_degree, fit_type=args.refine_type,
                                smooth_px=args.refine_smooth_px, clip_px=args.refine_clip_px)
        for k in range(nw):
            COEFS[k][i] = ci[k]
        nfit = sum(c is not None for c in ci)
        print(f"  fit a{i} z{int(z_anchor[i])}: {nfit}/{nw} windings refined",
              flush=True)

    fcx = interp1d(z_anchor, cx_a, bounds_error=False, fill_value=(cx_a[0], cx_a[-1]))
    fcy = interp1d(z_anchor, cy_a, bounds_error=False, fill_value=(cy_a[0], cy_a[-1]))

    def arclen_winding(curve):
        phid = np.linspace(eps, TWO_PI - eps, 12000)
        r = eval_winding(curve, phid); dr = np.gradient(r, phid)
        ds = np.sqrt(r ** 2 + dr ** 2)
        s = np.concatenate([[0.0], np.cumsum(0.5 * (ds[1:] + ds[:-1]) * np.diff(phid))])
        ncol = max(2, int(round(s[-1])))
        return np.interp(np.linspace(0, s[-1], ncol), s, phid)

    # per winding: fixed arc-length phi grid from a reference anchor's coef, then
    # interpolate the coef-eval'd radius across z. Fill anchors missing a winding
    # with the nearest fitted coef so the curve is continuous in z.
    win = []; total_cols = 0
    for k in range(nw):
        present = [i for i in range(n) if COEFS[k][i] is not None]
        if len(present) < 2:
            continue
        kref = min(present, key=lambda i: abs(i - n // 2))
        phig = arclen_winding(COEFS[k][kref])
        R = np.empty((n, phig.size), np.float32)
        for i in range(n):
            ci = COEFS[k][i] if COEFS[k][i] is not None else \
                COEFS[k][min(present, key=lambda p: abs(p - i))]
            R[i] = eval_winding(ci, phig)
        fRk = interp1d(z_anchor, R, axis=0, bounds_error=False,
                       fill_value=(R[0], R[-1]))
        th = seam + phig
        win.append((np.cos(th), np.sin(th), fRk, phig.size))
        total_cols += phig.size
    print(f"  whole roll: {len(win)} windings, {total_cols} cols")

    z_to_path = discover_ct_slices(args.ct_dir)
    z_list = sorted(z_to_path)[:: args.z_step]
    strip = np.empty((len(z_list), total_cols), dtype=np.float32)
    for j, z in enumerate(z_list):
        img = norm_slice(load_ct(z_to_path[z]))
        cx, cy = float(fcx(z)), float(fcy(z))
        c = 0
        for cos_t, sin_t, fRk, w in win:
            r = fRk(z)
            strip[j, c:c + w] = bilinear_sample(img, cx + r * cos_t, cy + r * sin_t)
            c += w
        if j % 200 == 0 or j == len(z_list) - 1:
            print(f"  [{j + 1}/{len(z_list)}] z{z}", flush=True)

    np.save(os.path.join(args.out_dir, "wholeroll.npy"), strip)
    np.save(os.path.join(args.out_dir, "z_index.npy"), np.array(z_list))
    from PIL import Image as _Im
    step = max(1, total_cols // 6000)
    sm = strip[:, ::step]; lo, hi = np.percentile(sm, [1, 99])
    sm = np.clip((sm - lo) / (hi - lo + 1e-6), 0, 1)
    _Im.fromarray((sm * 255).astype(np.uint8)).save(
        os.path.join(args.out_dir, "wholeroll_overview.png"))
    print(f"Done (render). whole roll {strip.shape}. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
