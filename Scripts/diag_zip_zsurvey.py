"""Survey the FULL Mickey reconstruction (Mickey_merged_rigid_nobin-tif.zip) in z:
which slices are film, which are packaging, and where the two PERFORATION rows sit.

The delivered unroll uses only z832-2015 of the OLD stitch = the picture band; the
sprocket rows were cropped away. This zip has them back (2419 slices, 3738x3769,
a DIFFERENT and larger reconstruction than the old 3063x3062 stitch -- so its z
index is its own and must be mapped separately, see diag_zip_match.py).

Metrics per sampled slice, all from a simple film/air threshold (the histogram is
cleanly bimodal: air ~9.6k, film ~14.5k):
  area      -- film pixel count. A perforation row loses ~1.27mm of every 7.62mm
               of film length => a ~17% DIP vs the neighbouring full-film margin.
               Film edges show as the fall to ~0.
  r_lo/r_hi -- radial extent of film about the image centre (packaging debris sits
               well outside the roll, so a jump in r_hi flags it).
  n_comp    -- connected components of the film mask, coarse: the picture band is
               ~35 long arcs; a perf row shatters each arc into ~5 pieces, so this
               roughly QUINTUPLES inside a perf band. Independent of `area`, and
               the sharper of the two signals.

Usage:
  python Scripts/diag_zip_zsurvey.py --zip data/Mickey_merged_rigid_nobin-tif.zip \
      --out-dir <dir> [--step 8] [--z0 0 --z1 2418]
"""
import argparse
import io
import os
import re
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tf
from scipy.ndimage import label


def otsu(a, nbins=512):
    """Otsu threshold. Local implementation -- skimage is NOT in the thesis env."""
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
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--z0", type=int, default=None)
    ap.add_argument("--z1", type=int, default=None)
    ap.add_argument("--thr", type=float, default=0.0,
                    help="Film threshold in raw units; 0 = auto (histogram valley "
                         "between the air and film modes of a mid slice).")
    ap.add_argument("--sub", type=int, default=2,
                    help="Spatial subsample for the metrics (2 = quarter the px).")
    ap.add_argument("--montage", type=int, default=0,
                    help="Save a contrast-stretched montage of this many sampled "
                         "slices (0 = none) -- the visual check on the metrics.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    zf = zipfile.ZipFile(args.zip)
    pat = re.compile(r"_(\d+)\.tif$", re.I)
    zmap = {int(pat.search(n).group(1)): n
            for n in zf.namelist() if pat.search(n)}
    zs = sorted(zmap)
    z0 = args.z0 if args.z0 is not None else zs[0]
    z1 = args.z1 if args.z1 is not None else zs[-1]
    sample = [k for k in zs if z0 <= k <= z1][:: args.step]
    print(f"{len(zmap)} slices z{zs[0]}-{zs[-1]}; sampling {len(sample)} "
          f"(step {args.step})", flush=True)

    def read(k):
        return tf.imread(io.BytesIO(zf.read(zmap[k])))[:: args.sub, :: args.sub]

    # PER-SLICE Otsu by default. A single global cut does NOT work here: overall
    # brightness varies a lot with z (a fixed threshold tuned mid-stack found ~0
    # film in the perf rows, which are visibly full of film), and a hand-rolled
    # "valley between the two modes" rule latched onto the bright EMULSION peak
    # instead of the air/film valley.
    thrs = []

    rec = []
    tiles = []
    for j, k in enumerate(sample):
        a = read(k)
        thr = float(args.thr) if args.thr > 0 else float(otsu(a))
        thrs.append(thr)
        m = a > thr
        area = int(m.sum())
        ys, xs = np.nonzero(m)
        if area > 500:
            cy, cx = a.shape[0] / 2, a.shape[1] / 2
            r = np.hypot(ys - cy, xs - cx)
            r_lo, r_hi = float(np.percentile(r, 0.5)), float(np.percentile(r, 99.5))
        else:
            r_lo = r_hi = 0.0
        n_comp = int(label(m)[1])
        rec.append((k, area, r_lo, r_hi, n_comp))
        if args.montage:
            b = a[::4, ::4].astype(np.float32)
            lo_, hi_ = np.percentile(b, [1, 99.5])
            tiles.append((k, np.clip((b - lo_) / (hi_ - lo_ + 1e-6), 0, 1)))
        if j % 10 == 0 or j == len(sample) - 1:
            print(f"  [{j + 1}/{len(sample)}] z{k}: thr {thr:.0f} area {area} "
                  f"r {r_lo:.0f}-{r_hi:.0f} comps {n_comp}", flush=True)

    if args.montage and tiles:
        from PIL import Image
        sel = tiles[:: max(1, len(tiles) // args.montage)][:args.montage]
        S = 300; cols = 8; rows = int(np.ceil(len(sel) / cols))
        mon = Image.new("L", (S * cols, S * rows), 0)
        for i, (k, im) in enumerate(sel):
            t = Image.fromarray((im * 255).astype(np.uint8)).resize((S, S))
            mon.paste(t, ((i % cols) * S, (i // cols) * S))
        mon.save(os.path.join(args.out_dir, "zsurvey_montage.png"))
        with open(os.path.join(args.out_dir, "zsurvey_montage.txt"), "w") as f:
            f.write(" ".join(str(k) for k, _ in sel) + "\n")
        print(f"  montage of {len(sel)} slices (z indices in zsurvey_montage.txt)")

    R = np.array(rec, float)
    np.savez(os.path.join(args.out_dir, "zsurvey.npz"), rec=R, thr=np.array(thrs),
             sub=args.sub, step=args.step)

    z, area, r_lo, r_hi, ncomp = R.T
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    ax[0].plot(z, area / 1e3, "-", lw=1)
    ax[0].set_ylabel("film area (kpx)")
    ax[0].set_title("film area: 0 = outside the film; a ~17% dip = a PERFORATION row")
    ax[1].plot(z, ncomp, "-", lw=1, color="C1")
    ax[1].set_ylabel("mask components")
    ax[1].set_title("components: ~35 arcs in the picture band, ~5x that in a perf row")
    ax[2].plot(z, r_lo, "-", lw=1, label="r_lo")
    ax[2].plot(z, r_hi, "-", lw=1, label="r_hi")
    ax[2].set_ylabel("film radius (px)"); ax[2].set_xlabel("zip slice index")
    ax[2].set_title("radial extent: a jump in r_hi = packaging, not film")
    ax[2].legend(fontsize=8)
    for a_ in ax:
        a_.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "zsurvey.png"), dpi=130)
    print(f"Done. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
