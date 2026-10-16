"""Locate the roll's SEAM from the OUTER-BOUNDARY profile r_out(phi).

Why a new detector.  `_detect_seam_angle` counts film->air transitions along
radial rays and takes the argmin (the azimuth where one air gap is missing).  On
the NEW scan that estimator is unusable: on consecutive slices with identical
centres it returned -59.5, -68.5, -102.0, -71.0, +89.0, +136.0 deg -- the finer
sampling resolves more air gaps and segmentation pinholes add spurious
transitions, so its argmin wanders.  Pooling the raw gap-count profiles gives
133.5 deg but with a dip of only 30.12 against a 30.1-36.2 baseline, i.e. not
separated from noise.

The physical signal used here is much sharper.  The film's OUTER END terminates
at the seam, so going around the roll the outermost radius r_out(phi) drops by
one full film pitch (~21 px old scan, ~26 px new) at that single azimuth, on an
otherwise smooth (eccentricity-dominated, once-per-turn) curve.  A step of a
whole pitch on a curve that is smooth everywhere else is easy to find, and --
unlike the gap counter -- it does not need the segmentation, only a per-slice
threshold, and it is insensitive to the centre (a centre error adds a SMOOTH
cos(phi) term, it cannot manufacture a step).

The same measurement is reported for the INNER boundary r_in(phi): the film's
inner end (the tongue) also terminates, giving a step of the opposite sign.
The two are independent estimates of the same physical ray.

VALIDATE ON THE OLD SCAN FIRST (`--ct-dir 01_Mickey_hdf`), where the answer is
independently known to be 60 deg, before believing anything it says about the
new scan.

Usage:
  python Scripts/diag_seam_outer.py --ct-dir 01_Mickey_hdf \
      --z 900,1100,1300,1500,1700 --center-xy 1500,1500 --expect-deg 60
"""
import argparse
import glob
import math
import os
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

TWO_PI = 2.0 * math.pi


def otsu(a, nbins=512):
    """Otsu threshold, computed PER SLICE.

    Per slice is load-bearing: overall brightness varies a lot with z on this
    scan, and a fixed threshold tuned mid-stack reports ~0 film in the perf rows
    (which are visibly full of film).  Local implementation -- skimage is NOT in
    the thesis env.  (Same routine as Scripts/diag_zip_zsurvey.py; do not
    re-derive it, an earlier hand-rolled variant here dropped the N factor from
    the between-class variance and thresholded ~all film away.)
    """
    h, e = np.histogram(a, bins=nbins)
    c = 0.5 * (e[1:] + e[:-1])
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    s0 = np.cumsum(h * c); s1 = s0[-1] - s0
    ok = (w0 > 0) & (w1 > 0)
    var = np.zeros(nbins)
    var[ok] = w0[ok] * w1[ok] * (s0[ok] / w0[ok] - s1[ok] / w1[ok]) ** 2
    return float(c[int(np.argmax(var))])


def boundary_profiles(mask, cx, cy, n_az, r0, r1, min_run=3):
    """Outermost / innermost radius carrying a film run of >= min_run px.

    Returns (r_out, r_in), each (n_az,) float with NaN where the ray is empty.
    """
    H, W = mask.shape
    az = np.linspace(0.0, TWO_PI, n_az, endpoint=False)
    rr = np.arange(r0, r1 + 1.0, 1.0)
    xs = np.round(cx + np.outer(np.cos(az), rr)).astype(np.int32)
    ys = np.round(cy + np.outer(np.sin(az), rr)).astype(np.int32)
    inside = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    A = np.zeros(xs.shape, dtype=bool)
    A[inside] = mask[ys[inside], xs[inside]]

    run = A[:, : -(min_run - 1)].copy()
    for k in range(1, min_run):
        run &= A[:, k: A.shape[1] - (min_run - 1) + k]

    any_run = run.any(axis=1)
    first = np.argmax(run, axis=1).astype(np.float64)
    last = (run.shape[1] - 1 - np.argmax(run[:, ::-1], axis=1)).astype(np.float64)
    r_in = np.where(any_run, r0 + first, np.nan)
    r_out = np.where(any_run, r0 + last + (min_run - 1), np.nan)
    return r_out, r_in, az


def step_response(prof, az, half_deg):
    """S(phi) = median(prof just BEFORE phi) - median(prof just AFTER phi).

    Medians (not means) so a couple of bad azimuth bins cannot invent a step,
    and a plain difference of two one-sided windows (not a derivative) so the
    response is a plateau-to-plateau jump, which is what a film end is.
    """
    n = len(prof)
    w = max(2, int(round(math.radians(half_deg) / (TWO_PI / n))))
    idx = np.arange(n)
    before = np.stack([prof[(idx - k) % n] for k in range(1, w + 1)])
    after = np.stack([prof[(idx + k) % n] for k in range(0, w)])
    with np.errstate(invalid="ignore"):
        return np.nanmedian(before, axis=0) - np.nanmedian(after, axis=0)


def top_peaks(S, az, k=4, min_sep_deg=15.0):
    """The k strongest |steps|, greedily separated in azimuth."""
    order = np.argsort(-np.abs(np.nan_to_num(S)))
    picks = []
    for i in order:
        if all(abs((az[i] - az[j] + math.pi) % TWO_PI - math.pi)
               > math.radians(min_sep_deg) for j in picks):
            picks.append(i)
        if len(picks) == k:
            break
    return picks


def ascii_profile(prof, az, width=100, label=""):
    n = width
    binned = np.array([np.nanmedian(prof[i * len(prof) // n:(i + 1) * len(prof) // n])
                       for i in range(n)])
    lo, hi = np.nanmin(binned), np.nanmax(binned)
    rows = 14
    grid = [[" "] * n for _ in range(rows)]
    for i, v in enumerate(binned):
        if not np.isfinite(v):
            continue
        r = int(round((hi - v) / (hi - lo + 1e-9) * (rows - 1)))
        grid[r][i] = "*"
    print(f"  {label}  r range {lo:.0f}..{hi:.0f} px  (phi 0..360 deg left->right)")
    for r, row in enumerate(grid):
        val = hi - (hi - lo) * r / (rows - 1)
        print(f"    {val:7.1f} |" + "".join(row))
    print("            +" + "-" * n)
    print("             0" + " " * (n // 2 - 4) + "180" + " " * (n // 2 - 6) + "360")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-dir", required=True)
    ap.add_argument("--z", default=None,
                    help="Comma list of z indices. Default: --n-slices spread "
                         "over the whole discovered range.")
    ap.add_argument("--n-slices", type=int, default=15)
    ap.add_argument("--center-xy", default=None,
                    help="cx,cy. Default = film-mask centroid (adequate: a "
                         "centre error is a smooth cos(phi) term, not a step).")
    ap.add_argument("--center-from-cache", default=None,
                    help="Walk cache dir; take the median cx_a/cy_a from its "
                         "matched_walks.npz. Preferred over the centroid, which "
                         "is pulled off-centre by the roll's eccentricity.")
    ap.add_argument("--n-az", type=int, default=1440)
    ap.add_argument("--half-deg", type=float, default=8.0,
                    help="One-sided window of the step detector.")
    ap.add_argument("--pitch-px", type=float, default=None,
                    help="Expected winding pitch; the step should be ~1 pitch.")
    ap.add_argument("--expect-deg", type=float, default=None,
                    help="Known answer, for validating the estimator.")
    ap.add_argument("--r-max", type=float, default=None,
                    help="Cap the radial search (keeps packaging out).")
    ap.add_argument("--save-npz", default=None,
                    help="Write the pooled profiles here so the step detector "
                         "can be re-tuned without re-reading the CT.")
    args = ap.parse_args()

    z_to_path = {}
    for p in glob.glob(os.path.join(args.ct_dir, "*.h5")):
        m = re.search(r"_(\d+)\.h5$", os.path.basename(p))
        if m:
            z_to_path[int(m.group(1))] = p
    if not z_to_path:
        raise SystemExit(f"no *_NNNN.h5 in {args.ct_dir}")
    zs_all = sorted(z_to_path)
    if args.z:
        zs = [int(x) for x in args.z.split(",")]
    else:
        zs = [zs_all[i] for i in np.linspace(0, len(zs_all) - 1,
                                            args.n_slices).astype(int)]
    print(f"{args.ct_dir}: {len(zs_all)} slices z{zs_all[0]}-{zs_all[-1]}; "
          f"using {len(zs)} -> {zs}")

    fixed_c = None
    if args.center_from_cache:
        cd = np.load(os.path.join(args.center_from_cache, "matched_walks.npz"),
                     allow_pickle=True)
        fixed_c = (float(np.median(cd["cx_a"])), float(np.median(cd["cy_a"])))
        print(f"centre from {args.center_from_cache}: "
              f"{fixed_c[0]:.1f},{fixed_c[1]:.1f}")
    elif args.center_xy:
        fixed_c = tuple(float(v) for v in args.center_xy.split(","))

    outs, ins = [], []
    az = None
    for z in zs:
        with h5py.File(z_to_path[z], "r") as f:
            img = np.asarray(f["image"], dtype=np.float32)
        thr = otsu(img)
        mask = img > thr
        if fixed_c:
            cx, cy = fixed_c
        else:
            ys, xs = np.nonzero(mask)
            cx, cy = float(xs.mean()), float(ys.mean())
        r1 = args.r_max or (min(img.shape) / 2.0 - 2.0)
        ro, ri, az = boundary_profiles(mask, cx, cy, args.n_az, 5.0, r1)
        frac = float(np.mean(np.isfinite(ro)))
        flag = ""
        if not 0.10 < mask.mean() < 0.65:
            flag = "  <-- SUSPECT: the roll should cover ~20-50% of the frame"
        print(f"  z{z}: thr {thr:.1f}  film {mask.mean()*100:.1f}%  "
              f"centre {cx:.1f},{cy:.1f}  r_out med {np.nanmedian(ro):.0f}  "
              f"r_in med {np.nanmedian(ri):.0f}  rays with film {frac*100:.0f}%"
              f"{flag}")
        outs.append(ro)
        ins.append(ri)

    R_out = np.nanmedian(np.stack(outs), axis=0)
    R_in = np.nanmedian(np.stack(ins), axis=0)
    if args.save_npz:
        np.savez(args.save_npz, R_out=R_out, R_in=R_in, az=az,
                 per_slice_out=np.stack(outs), per_slice_in=np.stack(ins),
                 z=np.array(zs))
        print(f"wrote {args.save_npz}")
    print(f"\npooled over {len(zs)} slices; azimuth resolution "
          f"{360.0/args.n_az:.2f} deg; step window +-{args.half_deg} deg")

    for name, prof, sign in (("r_out (film OUTER end)", R_out, +1),
                             ("r_in  (film INNER end / tongue)", R_in, -1)):
        print(f"\n=== {name} ===")
        ascii_profile(prof, az, label=name)
        S = step_response(prof, az, args.half_deg)
        noise = float(np.nanmedian(np.abs(S - np.nanmedian(S))))
        picks = top_peaks(S, az, k=4)
        print(f"  step noise (MAD over all azimuths) {noise:.2f} px"
              + (f"; expected step ~1 pitch = {args.pitch_px:.0f} px"
                 if args.pitch_px else ""))
        for rank, i in enumerate(picks):
            deg = math.degrees(az[i]) % 360.0
            snr = abs(S[i]) / (noise + 1e-9)
            drop = "DROP going CCW" if S[i] > 0 else "RISE going CCW"
            tag = ""
            if args.expect_deg is not None:
                d = abs((deg - args.expect_deg + 180.0) % 360.0 - 180.0)
                tag = f"   [{d:.1f} deg from the known {args.expect_deg:.1f}]"
            print(f"  #{rank+1}  az {deg:7.2f} deg   step {S[i]:+7.1f} px   "
                  f"SNR {snr:5.1f}x   {drop}{tag}")
        # the seam is where the OUTER end terminates: r_out drops as phi grows
        cand = [i for i in picks if np.sign(S[i]) == sign]
        if cand:
            print(f"  --> seam candidate from this profile: "
                  f"{math.degrees(az[cand[0]]) % 360.0:.2f} deg")


if __name__ == "__main__":
    main()
