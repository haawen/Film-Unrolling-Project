"""Visual check sheet for the perforation detection.

Draws each detected hole as it actually sits in the strip, with the fit drawn on
top, so the detection can be judged by eye rather than by a number.  Two views:

  perf_check_crops.png   a grid of individual holes spread over the whole reel,
                         each with the fitted centre (+) and box.  This is the
                         one for judging whether the CENTRE is on the hole.
  perf_check_runs.png    continuous stretches of the strip with every fit drawn,
                         so a missed or duplicated hole is obvious.

    python -m unwrapping.eval.perf_check_sheet <strip.npy> --perfs <perfs.npz> \
        --out-dir <dir> [--n-crops 24] [--n-runs 6]
"""

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strip")
    ap.add_argument("--perfs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-crops", type=int, default=24,
                    help="Holes per row shown in the crop grid.")
    ap.add_argument("--n-runs", type=int, default=6,
                    help="Continuous stretches shown, per row.")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(args.out_dir, exist_ok=True)
    s = np.load(args.strip, mmap_mode="r")
    P = np.load(args.perfs)
    Z, W = s.shape
    pitch = float(P["pitch"])
    hz, hc = [int(v) for v in P["tmpl"]]
    bands = {"low": P["band_lo"], "high": P["band_hi"]}
    rows = {"low": {k[3:]: P[k] for k in P.files if k.startswith("lo_")},
            "high": {k[3:]: P[k] for k in P.files if k.startswith("hi_")}}

    # ── 1. crop grid ──
    pad_z, pad_c = int(0.55 * hz), int(0.75 * hc)
    ncol = 8
    for name, d in rows.items():
        n = len(d["c"])
        idx = np.unique(np.linspace(0, n - 1, args.n_crops).astype(int))
        nrow = int(np.ceil(len(idx) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(2.1 * ncol, 2.3 * nrow))
        axs = np.atleast_1d(axs).ravel()
        for ax in axs:
            ax.axis("off")
        for j, i in enumerate(idx):
            cc, zc = d["c"][i], d["z"][i]
            a = int(cc - hc / 2 - pad_c); b = int(cc + hc / 2 + pad_c)
            z0 = int(zc - hz / 2 - pad_z); z1 = int(zc + hz / 2 + pad_z)
            if a < 0 or b > W or z0 < 0 or z1 > Z:
                continue
            crop = np.asarray(s[z0:z1, a:b], np.float32)
            ax = axs[j]; ax.axis("on")
            ax.imshow(crop, cmap="gray", extent=[a, b, z1, z0], aspect="auto")
            ax.add_patch(plt.Rectangle((cc - hc / 2, zc - hz / 2), hc, hz,
                                       fill=False, ec="lime", lw=1.6))
            ax.plot(cc, zc, "+", color="red", ms=13, mew=2)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"#{i}  score {d['score'][i]:.2f}", fontsize=7)
        fig.suptitle(f"{name} perforation row -- fitted centre (red +) and "
                     f"nominal {hz}x{hc} px box (green). "
                     f"{len(d['c'])} holes found in the reel.", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        p = os.path.join(args.out_dir, f"perf_check_crops_{name}.png")
        fig.savefig(p, dpi=110); plt.close(fig)
        print(f"wrote {p}", flush=True)

    # ── 2. continuous runs ──
    for name, d in rows.items():
        band = bands[name]
        z0 = max(0, int(band[0]) - int(0.8 * hz))
        z1 = min(Z, int(band[1]) + int(0.8 * hz))
        fig, axs = plt.subplots(args.n_runs, 1,
                                figsize=(17, 2.1 * args.n_runs))
        axs = np.atleast_1d(axs)
        for j, f in enumerate(np.linspace(0.03, 0.90, args.n_runs)):
            a = int(f * W); b = min(W, a + int(4.5 * pitch))
            ax = axs[j]
            ax.imshow(np.asarray(s[z0:z1, a:b], np.float32), cmap="gray",
                      extent=[a, b, z1, z0], aspect="auto")
            m = (d["c"] > a) & (d["c"] < b)
            for i in np.nonzero(m)[0]:
                ax.add_patch(plt.Rectangle(
                    (d["c"][i] - hc / 2, d["z"][i] - hz / 2), hc, hz,
                    fill=False, ec="lime", lw=1.4))
                ax.plot(d["c"][i], d["z"][i], "+", color="red", ms=12, mew=2)
            ax.set_yticks([])
            ax.set_title(f"strip columns {a}-{b}   ({m.sum()} holes, "
                         f"expect {(b - a) / pitch:.1f})", fontsize=8)
        fig.suptitle(f"{name} perforation row -- continuous stretches; a miss or "
                     f"a duplicate shows up as a gap or a double box", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        p = os.path.join(args.out_dir, f"perf_check_runs_{name}.png")
        fig.savefig(p, dpi=100); plt.close(fig)
        print(f"wrote {p}", flush=True)


if __name__ == "__main__":
    main()
