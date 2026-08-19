"""Register the OLD stitch frame (where all delivered geometry lives) to the NEW
Mickey_merged_rigid_nobin reconstruction: in-plane scale, rotation, translation.

    p_zip = s * R(theta) @ (p_old - c_old) + c_zip

Needed because the two are NOT the same sampling: old slices are 3063x3062, the
zip's 3738x3769 and ~1.245x finer in-plane. Every walked path, threshold and the
whole segmentation live in OLD pixels, so the frames must be related before
anything is rendered from the new data.

ACCURACY TARGET (perf bands only -- the picture band keeps its existing validated
strip): the sampling line must stay inside the 18px FILM layer, not the 4px
emulsion, so ~0.3% in scale and ~0.15deg in rotation (both = ~4px at r=1350).
The z map needs only ~1%: the winding curves move just 3.5-7px over 350 slices
(measured, diag_z_extrap_cv), so a 50-slice z error costs <1px of geometry.

METHOD -- geometric primitives, not image correlation. The first attempt at this
correlated high-passed, 4x-DECIMATED slices and failed completely (ncc 0.06-0.09,
rotation pinned at the search-grid edge, scale scattered +-1.3%, z residuals +-90):
decimating by 4 with no prefilter aliases the 21px/26px ring texture down to ~5px
and destroys the very structure the match depends on, leaving a multi-modal ring
-aliasing surface. So instead, per slice pair:
  1. film mask (per-slice Otsu), polar-resample about a provisional centre
  2. r_out(phi), r_in(phi) = outer/inner film boundary; algebraic circle fit to the
     outer boundary -> CENTRE (iterated twice)
  3. THETA from the circular cross-correlation of the mean-removed, mean-NORMALISED
     (hence scale-free) eccentricity profile e(phi) = r_out(phi)/mean - 1. The roll
     is eccentric ~80px ptp AND its outermost winding TERMINATES (a step in r_out),
     so the profile is strongly asymmetric = a sharp rotation signal.
  4. SCALE from the median over phi of r_out_zip(phi+theta)/r_out_old(phi) and the
     same for r_in, cross-checked against a 1D radial-coverage-profile match.
Each parameter is estimated from a signal that does not depend on the others, at
FULL resolution, so there is no aliasing and no coupled search grid.

Usage:
  python Scripts/diag_zip_match.py --zip data/Mickey_merged_rigid_nobin-tif.zip \
      --old-dir data/01_Mickey_hdf_subset --out-dir <dir> [--n-slices 8]
"""
import argparse
import glob
import io
import json
import os
import re
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
import tifffile as tf
from scipy.ndimage import map_coordinates

NPHI = 1440                                   # 0.25 deg azimuth bins


def otsu(a, nbins=512):
    h, e = np.histogram(a, bins=nbins)
    c = 0.5 * (e[1:] + e[:-1])
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    s0 = np.cumsum(h * c); s1 = s0[-1] - s0
    ok = (w0 > 0) & (w1 > 0)
    var = np.zeros(nbins)
    var[ok] = w0[ok] * w1[ok] * (s0[ok] / w0[ok] - s1[ok] / w1[ok]) ** 2
    return float(c[int(np.argmax(var))])


def film_mask(img):
    """Film/air mask, robust in BOTH frames.

    A single Otsu works on the zip (air ~10091, film ~14500, no background spike:
    p1 = 8810) but FAILS on the old stitch: 35% of an old slice is a CONSTANT 12
    (outside the reconstruction circle), so plain Otsu returns 5579 = "inside the
    FOV" vs "outside" and reports film coverage 1.000 out to r=1400 -- air gaps and
    the central hole counted as film.

    So the background stage must be CONDITIONAL, not blanket: applying it
    unconditionally over-thresholds the zip onto the bright emulsion and the
    annulus disappears entirely. Background is detected as a dominant modal bin
    (>15% of pixels) lying below the median; only then is it excluded before Otsu.
    """
    h, e = np.histogram(img, bins=512)
    c = 0.5 * (e[1:] + e[:-1])
    i = int(np.argmax(h))
    pop = img.ravel()
    if h[i] / h.sum() > 0.15 and c[i] < np.median(img):
        pop = img[img > c[i] + (e[1] - e[0])]
    return img > otsu(pop if pop.size > 1000 else img.ravel())


def coverage(mask, cy, cx, rmax):
    """Azimuthal film coverage vs radius (the robust boundary signal)."""
    phi = np.linspace(0, 2 * np.pi, NPHI, endpoint=False)
    r = np.arange(20, rmax)
    R, P = np.meshgrid(r, phi, indexing="ij")
    s = map_coordinates(mask.astype(np.float32),
                        [cy + R * np.sin(P), cx + R * np.cos(P)], order=0)
    return r, s.mean(1), s > 0.5


def boundaries(mask, cy, cx, rmax, run=3):
    """r_in/r_out per azimuth: first/last radius with `run` consecutive film px."""
    phi = np.linspace(0, 2 * np.pi, NPHI, endpoint=False)
    r = np.arange(20, rmax)
    R, P = np.meshgrid(r, phi, indexing="ij")
    m = map_coordinates(mask.astype(np.float32),
                        [cy + R * np.sin(P), cx + R * np.cos(P)],
                        order=0) > 0.5                        # (nr, nphi)
    # require `run` consecutive radial samples to reject speckle
    k = np.ones(run, bool)
    solid = np.zeros_like(m)
    for i in range(run):
        solid[i:m.shape[0] - run + 1 + i] |= np.all(
            np.stack([m[j:m.shape[0] - run + 1 + j] for j in range(run)]), axis=0)
    r_in = np.full(NPHI, np.nan); r_out = np.full(NPHI, np.nan)
    for j in range(NPHI):
        idx = np.nonzero(solid[:, j])[0]
        if idx.size:
            r_in[j] = r[idx[0]]; r_out[j] = r[idx[-1]]
    return phi, r_in, r_out


def circle_fit(x, y):
    """Kasa algebraic circle fit -> (cy, cx, R)."""
    A = np.stack([x, y, np.ones_like(x)], 1)
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = sol[0] / 2, sol[1] / 2
    R = np.sqrt(sol[2] + cx ** 2 + cy ** 2)
    return cy, cx, R


def analyse(img, rmax, iters=2, cov_thresh=0.5):
    """mask -> centre (outer-boundary circle fit) + boundary profiles + annulus.

    The ANNULUS is located from the coverage profile (first/last radius with
    coverage > cov_thresh); per-azimuth boundaries are then taken only inside it,
    so a bright reconstruction-circle rim outside the roll or a central artefact
    cannot masquerade as the film edge (both did in the first attempt: r_out came
    out a z-invariant 1391 old / 1690 zip, and old r_in pinned to the search floor).
    """
    m = film_mask(img)
    ys, xs = np.nonzero(m)
    cy, cx = float(ys.mean()), float(xs.mean())
    r_lo = r_hi = None
    for _ in range(iters):
        r, cov, _ = coverage(m, cy, cx, rmax)
        hit = np.nonzero(cov > cov_thresh)[0]
        if hit.size < 10:
            raise RuntimeError("no annulus found -- check the mask")
        r_lo, r_hi = float(r[hit[0]]), float(r[hit[-1]])
        phi, r_in, r_out = boundaries(m, cy, cx, int(r_hi * 1.05))
        ok = np.isfinite(r_out) & (r_out > 0.8 * r_lo)
        x = cx + r_out[ok] * np.cos(phi[ok]); y = cy + r_out[ok] * np.sin(phi[ok])
        cy, cx, _R = circle_fit(x, y)
    phi, r_in, r_out = boundaries(m, cy, cx, int(r_hi * 1.05))
    bad = ~np.isfinite(r_out) | (r_out < 0.8 * r_lo)
    r_out = np.where(bad, np.nan, r_out)
    r_in = np.where(~np.isfinite(r_in) | (r_in > 1.2 * r_lo), np.nan, r_in)
    return dict(cy=cy, cx=cx, phi=phi, r_in=r_in, r_out=r_out,
                r_lo=r_lo, r_hi=r_hi, mask=m)


def polar_mask(mask, cy, cx, r_phys, scale, nphi=3600):
    """Sample the film mask on a COMMON PHYSICAL polar grid.

    r_phys is in OLD-frame px; the zip is sampled at r_phys*scale, so both maps
    live on the same physical (r, phi) grid and can be compared directly.
    """
    phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    R, P = np.meshgrid(np.asarray(r_phys, float) * scale, phi, indexing="ij")
    v = map_coordinates(mask.astype(np.float32),
                        [cy + R * np.sin(P), cx + R * np.cos(P)], order=1)
    v = v - v.mean(axis=1, keepdims=True)          # per-radius mean removal
    return v


def rotation_from_polar(Mo, Mz):
    """Best rotation between two polar maps, using EVERY radius at once.

    C(theta) = sum_r sum_phi Mo(r,phi) * Mz(r,phi+theta), evaluated for all theta
    in one shot via an FFT along phi. Unlike the outer-boundary profile (which is
    dominated by the roll's 2-cycle elliptical component and therefore ambiguous by
    180deg), this uses all ~35 windings, their air gaps and the film end, so the
    peak is unique. Returns (theta_deg, normalised peak, C).
    """
    F = np.fft.rfft(Mo, axis=1) * np.conj(np.fft.rfft(Mz, axis=1))
    C = np.fft.irfft(F.sum(axis=0), n=Mo.shape[1])
    C = C / (np.linalg.norm(Mo) * np.linalg.norm(Mz))
    k = int(np.argmax(C))
    y0, y1, y2 = C[(k - 1) % len(C)], C[k], C[(k + 1) % len(C)]
    den = y0 - 2 * y1 + y2
    kk = k + (0.5 * (y0 - y2) / den if den != 0 else 0.0)
    if kk > len(C) / 2:
        kk -= len(C)
    return float(kk * 360.0 / len(C)), float(y1), C


def circ_xcorr_shift(a, b):
    """Circular cross-correlation peak of two 1-D profiles, sub-bin (parabolic).

    Returns the shift k (in bins) such that a(phi) ~ b(phi + k), and the peak
    correlation. NaNs are filled with 0 after mean removal.
    """
    def prep(v):
        v = np.asarray(v, float).copy()
        good = np.isfinite(v)
        v[~good] = np.nan
        v = v / np.nanmean(v) - 1.0            # scale-free
        v[~np.isfinite(v)] = 0.0
        v -= v.mean()
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    A, B = prep(a), prep(b)
    cc = np.fft.irfft(np.fft.rfft(A) * np.conj(np.fft.rfft(B)), n=len(A))
    k = int(np.argmax(cc))
    y0, y1, y2 = cc[(k - 1) % len(cc)], cc[k], cc[(k + 1) % len(cc)]
    denom = (y0 - 2 * y1 + y2)
    dk = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
    kk = k + dk
    if kk > len(cc) / 2:
        kk -= len(cc)
    return float(kk), float(y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--old-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-slices", type=int, default=8)
    ap.add_argument("--pic-band", type=float, nargs=2, default=[470, 1975],
                    help="zip z of the picture band (diag_zip_zsurvey)")
    ap.add_argument("--old-band", type=float, nargs=2, default=[832, 2015])
    ap.add_argument("--rmax-old", type=int, default=1500)
    ap.add_argument("--rmax-zip", type=int, default=1900)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    zf = zipfile.ZipFile(args.zip)
    pat = re.compile(r"_(\d+)\.tif$", re.I)
    zmap = {int(pat.search(n).group(1)): n for n in zf.namelist() if pat.search(n)}
    olds = sorted(p for p in glob.glob(os.path.join(args.old_dir, "*.h5"))
                  if "Probabilities" not in p)
    sel = [olds[i] for i in np.unique(np.linspace(0, len(olds) - 1,
                                                  args.n_slices).astype(int))]
    a_z = (args.pic_band[1] - args.pic_band[0]) / (args.old_band[1] - args.old_band[0])
    b_z = args.pic_band[0] - a_z * args.old_band[0]
    print(f"{len(olds)} old slices ({len(sel)} used); z guess "
          f"z_zip = {a_z:.4f}*z_old {b_z:+.1f}\n"
          f"(z only needs ~1%: geometry moves 3.5-7px per 350 slices)")

    rows = []
    for p in sel:
        zo = int(re.search(r"_(\d+)\.h5$", p).group(1))
        zz = int(round(a_z * zo + b_z))
        if zz not in zmap:
            continue
        with h5py.File(p, "r") as f:
            O = analyse(np.asarray(f["image"], np.float32), args.rmax_old)
        Z = analyse(tf.imread(io.BytesIO(zf.read(zmap[zz]))).astype(np.float32),
                    args.rmax_zip)

        dk, pk = circ_xcorr_shift(Z["r_out"], O["r_out"])
        theta = dk * 360.0 / NPHI                       # deg, zip = old rotated by
        # scale: compare radii at MATCHED azimuth (shift the old profile by dk)
        idx = (np.arange(NPHI) + dk) % NPHI
        def shifted(v):
            return np.interp(idx, np.arange(NPHI), v, period=NPHI)
        ro_s, ri_s = shifted(O["r_out"]), shifted(O["r_in"])
        with np.errstate(invalid="ignore"):
            s_out = np.nanmedian(Z["r_out"] / ro_s)
            s_in = np.nanmedian(Z["r_in"] / ri_s)
        s_span = (Z["r_hi"] - Z["r_lo"]) / (O["r_hi"] - O["r_lo"])
        rows.append(dict(z_old=zo, z_zip=zz, theta_deg=theta, peak=pk,
                         s_out=float(s_out), s_in=float(s_in),
                         s_span=float(s_span),
                         r_lo_old=O["r_lo"], r_hi_old=O["r_hi"],
                         r_lo_zip=Z["r_lo"], r_hi_zip=Z["r_hi"],
                         c_old=[O["cy"], O["cx"]], c_zip=[Z["cy"], Z["cx"]],
                         rout_old=float(np.nanmedian(O["r_out"])),
                         rout_zip=float(np.nanmedian(Z["r_out"])),
                         rin_old=float(np.nanmedian(O["r_in"])),
                         rin_zip=float(np.nanmedian(Z["r_in"]))))
        r = rows[-1]
        print(f"  old z{zo} <-> zip z{zz}: theta {theta:+.3f}deg (peak {pk:.3f}) | "
              f"s_out {s_out:.4f} s_in {s_in:.4f} s_span {s_span:.4f} | "
              f"annulus old {O['r_lo']:.0f}-{O['r_hi']:.0f} "
              f"zip {Z['r_lo']:.0f}-{Z['r_hi']:.0f}", flush=True)

    th = np.array([r["theta_deg"] for r in rows])
    so = np.array([r["s_out"] for r in rows]); si = np.array([r["s_in"] for r in rows])
    pk = np.array([r["peak"] for r in rows])
    print(f"\n  theta  {th.mean():+.3f} +- {th.std():.3f} deg   "
          f"(xcorr peak {pk.mean():.3f}; target accuracy 0.15deg)")
    print(f"  s_out  {so.mean():.4f} +- {so.std():.4f}  "
          f"({100 * so.std() / so.mean():.2f}%; target 0.3%)")
    print(f"  s_in   {si.mean():.4f} +- {si.std():.4f}  "
          f"({100 * si.std() / si.mean():.2f}%)")
    sp = np.array([r["s_span"] for r in rows])
    print(f"  s_span {sp.mean():.4f} +- {sp.std():.4f}  "
          f"({100 * sp.std() / sp.mean():.2f}%)  [annulus width ratio, "
          f"translation-free]")
    print(f"  s_out vs s_in differ by {100 * abs(so.mean() - si.mean()) / so.mean():.2f}%"
          f"  (a real similarity => they must AGREE; disagreement = a scale that "
          f"is not uniform, i.e. the model is wrong)")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot([r["z_old"] for r in rows], th, "o-"); ax[0].set_title("theta (deg)")
    ax[1].plot([r["z_old"] for r in rows], so, "o-", label="s_out")
    ax[1].plot([r["z_old"] for r in rows], si, "s-", label="s_in")
    ax[1].legend(); ax[1].set_title("in-plane scale")
    ax[2].plot([r["z_old"] for r in rows], pk, "o-")
    ax[2].set_title("eccentricity-profile xcorr peak")
    for a_ in ax:
        a_.set_xlabel("old z"); a_.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, "zip_match.png"), dpi=130)

    with open(os.path.join(args.out_dir, "zip_match.json"), "w") as f:
        json.dump(dict(rows=rows, theta_mean=float(th.mean()),
                       theta_std=float(th.std()), s_out_mean=float(so.mean()),
                       s_out_std=float(so.std()), s_in_mean=float(si.mean()),
                       z_map=[a_z, b_z]), f, indent=1)
    print(f"Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
