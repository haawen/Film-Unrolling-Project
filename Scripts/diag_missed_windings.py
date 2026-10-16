"""Does the walk track EVERY emulsion band? — a CT-referenced miss census.

Reads the CT directly (not the segmentation) so it is independent of the mask the
walk was driven by: along radial rays it finds the bright emulsion peaks, then
asks whether each one has a tracked winding within a few px. A peak with no track
is film the render will silently drop; a track with no peak is a phantom winding.

This is the check the fit overlays only imply. "Bare band = MISS" is easy to miss
by eye over 34 windings x 360 deg, and a winding that is walked but then dropped
by --min-coverage looks identical in a full-ring overlay to one that was never
found.

Reports misses grouped by radius so an OUTERMOST-wrap miss (the last, thinnest
wrap) is not averaged together with the inner tongue.

Usage:
  python Scripts/diag_missed_windings.py --npz .../matched_walks.npz \
      --ct-dir 01_Mickey_sprockets --anchors 0,10,19
"""
import argparse
import glob
import math
import os
import re

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

TWO_PI = 2.0 * math.pi


def emulsion_peaks(nrm, cx, cy, az, r0, r1, min_prom, smooth=3.0):
    """Radii of bright-line maxima along one ray, selected by PROMINENCE.

    Prominence, not an absolute intensity threshold: this scan's brightness
    varies strongly with z and with radius, so a fixed cut (0.55 of the 1-99
    percentile range) found only ~a third of the emulsion lines on some slices
    and none on others -- which silently under-reports misses and reports every
    correctly-tracked winding as a 'phantom'. A peak's height above the higher of
    its two flanking valleys is what an emulsion line actually has and film base
    does not.
    """
    rr = np.arange(r0, r1, 0.5)
    prof = map_coordinates(nrm, [cy + rr * math.sin(az), cx + rr * math.cos(az)],
                           order=1)
    ps = gaussian_filter1d(prof, smooth)
    ismax = (ps[1:-1] >= ps[:-2]) & (ps[1:-1] > ps[2:])
    ismin = (ps[1:-1] <= ps[:-2]) & (ps[1:-1] < ps[2:])
    maxi = np.nonzero(ismax)[0] + 1
    mini = np.nonzero(ismin)[0] + 1
    if maxi.size == 0 or mini.size == 0:
        return np.empty(0)
    out = []
    for i in maxi:
        left = mini[mini < i]
        right = mini[mini > i]
        if left.size == 0 or right.size == 0:
            continue                      # edge peak: no valley on one side
        prom = ps[i] - max(ps[left[-1]], ps[right[0]])
        if prom >= min_prom:
            out.append(rr[i])
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="matched_walks.npz")
    ap.add_argument("--ct-dir", required=True)
    ap.add_argument("--anchors", default="all",
                    help="Comma indices or 'all'.")
    ap.add_argument("--az-step-deg", type=float, default=5.0)
    ap.add_argument("--tol-px", type=float, default=6.0,
                    help="A peak is 'tracked' if a winding is within this.")
    ap.add_argument("--min-prom", type=float, default=0.06,
                    help="Minimum peak prominence (normalised units) for a "
                         "bright line to count as an emulsion band.")
    ap.add_argument("--r-pad", type=float, default=40.0,
                    help="Search this far beyond the tracked radial range, so a "
                         "MISSED outermost/innermost wrap is still seen.")
    ap.add_argument("--split-r", type=float, default=None,
                    help="Radius separating the 'inner' and 'outer' report "
                         "(default: midpoint of the tracked range).")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    P_all = d["paths"]
    cx_a = np.asarray(d["cx_a"], float)
    cy_a = np.asarray(d["cy_a"], float)
    z_anchor = np.asarray(d["z_anchor"], float)
    idxs = (list(range(len(P_all))) if args.anchors.strip() == "all"
            else [int(x) for x in args.anchors.split(",")])

    z_to_path = {}
    for p in glob.glob(os.path.join(args.ct_dir, "*.h5")):
        m = re.search(r"_(\d+)\.h5$", os.path.basename(p))
        if m:
            z_to_path[int(m.group(1))] = p

    azs = np.arange(0.0, 360.0, args.az_step_deg)
    print(f"{args.npz}\n  {len(idxs)} anchors, {len(azs)} azimuths each, "
          f"tol {args.tol_px} px")
    print(f"\n  {'anchor':>6} {'z':>6} {'wnd':>4} {'inner miss':>11} "
          f"{'outer miss':>11} {'phantom':>8}   worst-miss radii")
    tot_in = tot_out = tot_ph = tot_cells = tot_tr = 0
    for a in idxs:
        za = min(z_to_path, key=lambda zz: abs(zz - z_anchor[a]))
        with h5py.File(z_to_path[za], "r") as f:
            img = np.asarray(f["image"], dtype=np.float32)
        lo, hi = np.percentile(img, [1, 99])
        nrm = np.clip((img - lo) / (hi - lo + 1e-6), 0.0, 1.0)

        cx, cy = cx_a[a], cy_a[a]
        P = [np.asarray(p) for p in P_all[a]]
        TH = [np.arctan2(p[:, 1] - cy, p[:, 0] - cx) for p in P]
        R = [np.hypot(p[:, 0] - cx, p[:, 1] - cy) for p in P]
        # Window from PERCENTILES of the per-path extremes, not the absolute
        # min/max: the tongue winding dives ~180 px further in than any other
        # (581 vs 760 on z500), and opening the window that far lets the ray hit
        # the concentric rings inside the BORE, which are not film -- they made
        # up most of the "inner misses" in the first version of this census.
        # Asymmetric on purpose: only the INNER side has non-film structure to
        # run into (the bore rings), so it uses the robust percentile; the outer
        # side is bounded by air, and clipping it with a percentile would hide
        # exactly the thing this census exists to find -- a missed OUTERMOST wrap.
        rmin = float(np.percentile([r.min() for r in R], 10)) - args.r_pad
        rmax = max(float(r.max()) for r in R) + args.r_pad
        split = args.split_r or 0.5 * (rmin + rmax)

        n_in = n_out = n_ph = 0
        n_pk = n_tr = 0
        miss_r = []
        for azd in azs:
            az = math.radians(azd)
            pk = emulsion_peaks(nrm, cx, cy, az, rmin, rmax, args.min_prom)
            tr = []
            for p, th, r in zip(P, TH, R):
                dif = np.abs((th - az + math.pi) % TWO_PI - math.pi)
                j = int(np.argmin(dif))
                if dif[j] < math.radians(args.az_step_deg * 0.15):
                    tr.append(float(r[j]))
            tr = np.array(sorted(tr))
            # Only judge tracks that lie INSIDE the search window; a track
            # outside it (the tongue diving toward the bore, the outermost wrap)
            # has no peak to match simply because no peak was looked for there,
            # and counting it as a phantom is an artefact of the window.
            tr = tr[(tr >= rmin) & (tr <= rmax)] if tr.size else tr
            n_pk += pk.size; n_tr += tr.size
            for q in pk:
                if tr.size == 0 or np.min(np.abs(tr - q)) > args.tol_px:
                    miss_r.append(q)
                    if q >= split:
                        n_out += 1
                    else:
                        n_in += 1
            for t in tr:
                if pk.size == 0 or np.min(np.abs(pk - t)) > args.tol_px:
                    n_ph += 1
        # denominators are (ray, band) CELLS, not rays: a ray carries ~34 bands,
        # so a per-ray percentage reads >100% and means nothing.
        tot_in += n_in; tot_out += n_out; tot_ph += n_ph
        tot_cells += n_pk
        tot_tr += n_tr
        worst = ", ".join(f"{v:.0f}" for v in sorted(set(np.round(miss_r)))[:6])
        print(f"  {a:>6} {za:>6} {len(P):>4} {n_in:>5}/{n_pk:<5} "
              f"{n_out:>5}/{n_pk:<5} {n_ph:>4}/{n_tr:<5}  r{rmin:.0f}-{rmax:.0f}  {worst}")

    print(f"\n  TOTAL over {len(idxs)} anchors x {len(azs)} azimuths:")
    print(f"    CT emulsion bands seen   {tot_cells}")
    print(f"    inner-region misses {tot_in} "
          f"({100.0*tot_in/max(1,tot_cells):.1f}% of bands)")
    print(f"    outer-region misses {tot_out} "
          f"({100.0*tot_out/max(1,tot_cells):.1f}% of bands)")
    print(f"    phantom tracks      {tot_ph} "
          f"({100.0*tot_ph/max(1,tot_tr):.1f}% of tracked "
          f"points) -- a high number here means the PEAK DETECTOR is failing, "
          f"check it before reading the miss columns")
    print("\n  NOTE a 'miss' concentrated in a contiguous azimuth arc with a "
          "SMOOTHLY VARYING radius is a real winding the tracker dropped; "
          "scattered single-ray misses are usually peak-detector noise.")


if __name__ == "__main__":
    main()
