"""Measure the SEAM azimuth in both reconstructions -> the rotation between them.

Why this and not another image-alignment objective: a roll of concentric bands is
nearly ROTATIONALLY SYMMETRIC, so band geometry cannot determine rotation. Measured
the hard way -- curve-template alignment over 5 slices returned theta = -104, +116,
-170, -132, +124 deg (std 126!) while the SCALE from the same fits was stable to
0.15%, because radial spacing is rotation-invariant. Only two features break the
symmetry: the eccentricity (2-cycle => ambiguous by 180deg) and the film ends/SEAM.

The seam is the spiral's radial discontinuity: at that azimuth one air gap is
missing (two windings merge), so a radial ray crosses fewer film->air transitions.
The old frame's value is independently validated at ~60deg (a physical constant of
this roll, confirmed across v9/v12/v3nohp), so

    theta = seam_zip - seam_old

Single-slice detection is known unreliable on this data (it once returned 27.5deg
vs the true 60deg and cost a full re-walk), so this samples many slices and takes
the CIRCULAR MEDIAN, and reports the spread so the estimate can be trusted or not.
The old frame is measured the same way as a CONTROL: it must come out near 60deg,
otherwise the estimator -- not the data -- is at fault.

Usage:
  python Scripts/diag_seam_zip.py --zip-dir 01_Mickey_sprockets \
      --old-dir 01_Mickey_hdf --out-dir <dir> \
      --zip-z 470 480 ... --old-z 840 860 ...
"""
import argparse
import glob
import json
import math
import os
import re

import numpy as np
import h5py

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unwrapping.inr.geodesic_unwrap import _detect_seam_angle


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
    """Conditional background removal, then Otsu (see diag_zip_match.film_mask):
    the old stitch has 35% of pixels at a constant 12 outside the FOV, which drags a
    plain Otsu onto the wrong split; the zip has no such spike."""
    h, e = np.histogram(img, bins=512)
    c = 0.5 * (e[1:] + e[:-1])
    i = int(np.argmax(h))
    pop = img.ravel()
    if h[i] / h.sum() > 0.15 and c[i] < np.median(img):
        pop = img[img > c[i] + (e[1] - e[0])]
    return img > otsu(pop if pop.size > 1000 else img.ravel())


def seam_of(img):
    m = film_mask(img)
    ys, xs = np.nonzero(m)
    cy, cx = float(ys.mean()), float(xs.mean())
    r = np.hypot(ys - cy, xs - cx)
    r_min, r_max = np.percentile(r, [2, 98])
    return math.degrees(_detect_seam_angle(m, (cy, cx), r_min, r_max))


def circ_stats(deg):
    a = np.deg2rad(np.asarray(deg, float))
    z = np.mean(np.exp(1j * a))
    mean = math.degrees(np.angle(z))
    # circular spread: mean absolute deviation from the circular mean
    d = np.degrees(np.angle(np.exp(1j * (a - np.deg2rad(mean)))))
    return mean, float(np.median(np.abs(d))), float(np.std(d)), np.round(
        np.degrees(a), 1).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip-dir", required=True)
    ap.add_argument("--old-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--zip-z", type=int, nargs="+", required=True)
    ap.add_argument("--old-z", type=int, nargs="+", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    zs = []
    for z in args.zip_z:
        p = os.path.join(args.zip_dir, f"Mickey_merged_{z:04d}.h5")
        if not os.path.exists(p):
            print(f"  zip z{z}: missing"); continue
        with h5py.File(p, "r") as f:
            v = seam_of(np.asarray(f["image"], np.float32))
        zs.append(v); print(f"  zip z{z}: seam {v:+.2f} deg", flush=True)

    os_ = []
    old_map = {int(m.group(1)): p for p in glob.glob(os.path.join(args.old_dir, "*.h5"))
               for m in [re.search(r"_(\d+)\.h5$", os.path.basename(p))] if m}
    for z in args.old_z:
        if z not in old_map:
            print(f"  old z{z}: missing"); continue
        with h5py.File(old_map[z], "r") as f:
            v = seam_of(np.asarray(f["image"], np.float32))
        os_.append(v); print(f"  old z{z}: seam {v:+.2f} deg", flush=True)

    mz, madz, sdz, allz = circ_stats(zs)
    mo, mado, sdo, allo = circ_stats(os_)
    print(f"\n  ZIP seam  {mz:+.2f} deg  (MAD {madz:.2f}, sd {sdz:.2f}) {allz}")
    print(f"  OLD seam  {mo:+.2f} deg  (MAD {mado:.2f}, sd {sdo:.2f}) {allo}")
    print(f"  CONTROL: old seam must be ~60 deg (validated physical constant) -> "
          f"{'OK' if abs(((mo - 60 + 180) % 360) - 180) < 10 else 'FAILED: the '
           'estimator is at fault, not the data'}")
    theta = ((mz - mo + 180) % 360) - 180
    print(f"  => theta (zip - old) = {theta:+.2f} deg")
    with open(os.path.join(args.out_dir, "seam.json"), "w") as f:
        json.dump(dict(zip_seam=mz, old_seam=mo, theta=theta, zip_all=allz,
                       old_all=allo, zip_mad=madz, old_mad=mado), f, indent=1)
    print(f"Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
