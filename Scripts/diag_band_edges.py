"""Find the TRUE z limits of the picture band on the new full-width scan.

CLAUDE.md records the picture band as z470-1975 and the dense walk was launched
with Z_START=470 on that basis. The z470 walk overlay shows that is wrong: the
film bands there are dashed by perforation holes and the walk tears. Everything
downstream inherits it -- the first chunk's anchors are garbage, and the batch's
first find_spool_center sample is taken on a perf slice (b0's came back cx 1812
against ~1904 for its neighbours, a ~90 px outlier).

Measurement, CT only, no segmentation needed: sample the film annulus in polar
coordinates and report

  cover  -- fraction of the annulus above an Otsu threshold. Perforations are
            holes THROUGH the film, so inside a perf band the film's own area
            drops.
  dash   -- mean number of air runs longer than --dash-deg encountered going
            around each radius ring. In the picture band a ring crosses the
            inter-winding air a fixed number of times; a perf band adds one
            extra hole per frame, so this rises sharply. This is the sensitive
            one: it counts holes rather than measuring their area.

Print both against z and read the band edge off where they plateau. The centre
only has to be good to a few tens of px -- both statistics are annulus-wide.

Usage:
  python Scripts/diag_band_edges.py --ct-dir 01_Mickey_sprockets \
      --z 400 700 --step 5
"""
import argparse
import glob
import math
import os

import h5py
import numpy as np


def otsu(a, nbins=512):
    h, e = np.histogram(a, bins=nbins)
    c = 0.5 * (e[1:] + e[:-1])
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    s0 = np.cumsum(h * c); s1 = s0[-1] - s0
    ok = (w0 > 0) & (w1 > 0)
    var = np.zeros(nbins)
    var[ok] = w0[ok] * w1[ok] * (s0[ok] / w0[ok] - s1[ok] / w1[ok]) ** 2
    return float(c[int(np.argmax(var))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct-dir", required=True)
    ap.add_argument("--z", nargs=2, type=int, required=True)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--cx", type=float, default=1890.0)
    ap.add_argument("--cy", type=float, default=1885.0)
    ap.add_argument("--r", nargs=2, type=float, default=(820.0, 1620.0))
    ap.add_argument("--dr", type=float, default=8.0)
    ap.add_argument("--dphi-deg", type=float, default=0.25)
    ap.add_argument("--dash-deg", type=float, default=1.5,
                    help="An air run longer than this counts as a gap.")
    ap.add_argument("--ref-z", type=int, default=1200,
                    help="Slice the fixed threshold is taken from; must be well "
                         "inside the picture band.")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.ct_dir, "*.h5")))
    byz = {}
    for f in files:
        # STRIP THE EXTENSION FIRST. "Mickey_merged_0042.h5" otherwise yields the
        # digits "00425" -- the 5 comes from ".h5" -- whose last four read as
        # 0425, so slice z42 silently answers to z425. That made an entire band
        # sweep return the same few packaging slices and look perfectly flat.
        d = "".join(ch for ch in os.path.splitext(os.path.basename(f))[0]
                    if ch.isdigit())
        if d:
            byz[int(d[-4:])] = f

    rr = np.arange(args.r[0], args.r[1], args.dr)
    pp = np.radians(np.arange(0.0, 360.0, args.dphi_deg))
    R, P = np.meshgrid(rr, pp, indexing="ij")
    XX = np.rint(args.cx + R * np.cos(P)).astype(np.int32)
    YY = np.rint(args.cy + R * np.sin(P)).astype(np.int32)
    minrun = max(1, int(round(args.dash_deg / args.dphi_deg)))

    def sample(z):
        f = byz.get(z)
        if f is None:
            return None, None
        with h5py.File(f, "r") as h:
            img = h["image"][()]
        H, W = img.shape
        ok = (XX >= 0) & (XX < W) & (YY >= 0) & (YY < H)
        pol = np.zeros(XX.shape, np.float32)
        pol[ok] = img[YY[ok], XX[ok]]
        return pol, ok

    # ONE threshold for the whole sweep, taken on a slice known to be picture.
    # A per-slice Otsu is self-defeating here: it re-balances to whatever film
    # fraction the slice happens to have, so it partly cancels the very drop in
    # film area that a perforation band is being detected by.
    ref, refok = sample(args.ref_z)
    if ref is None:
        raise SystemExit(f"reference slice z{args.ref_z} not found")
    thr = otsu(ref[refok])
    print(f"annulus r{args.r[0]:.0f}-{args.r[1]:.0f} about ({args.cx:.0f},"
          f"{args.cy:.0f}), {len(rr)} rings x {len(pp)} azimuths, "
          f"gap = air run > {args.dash_deg} deg")
    print(f"fixed threshold {thr:.1f} from reference slice z{args.ref_z} "
          f"(cover there {float((ref > thr)[refok].mean()):.4f})")
    print(f"\n  {'z':>6} {'cover':>8} {'gaps/ring':>11}")
    for z in range(args.z[0], args.z[1] + 1, args.step):
        pol, ok = sample(z)
        if pol is None:
            print(f"  {z:6d}   (no file)")
            continue
        b = pol > thr
        cover = float(b.mean())
        # count air runs per ring
        gaps = 0
        for i in range(b.shape[0]):
            row = b[i]
            # wrap so a gap across 0 deg is not double counted
            e = np.flatnonzero(np.diff(np.r_[row[-1], row].astype(np.int8)) < 0)
            s = np.flatnonzero(np.diff(np.r_[row[-1], row].astype(np.int8)) > 0)
            if len(e) == 0 or len(s) == 0:
                continue
            n = 0
            for k in e:
                nxt = s[s > k]
                end = nxt[0] if len(nxt) else len(row) + s[0]
                if end - k >= minrun:
                    n += 1
            gaps += n
        print(f"  {z:6d} {cover:8.4f} {gaps / b.shape[0]:11.2f}")


if __name__ == "__main__":
    main()
