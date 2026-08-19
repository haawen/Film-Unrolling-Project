"""GT-free stillness measurement from the picture APERTURE EDGE.

The question a perf-locked render has to answer is "is the picture actually
stiller?", and the GT fidelity metrics could not answer it: baseline and both
perf-locked variants came out tied (grad_corr 0.077-0.084) with a 58-82 px
residual registration shift, meaning crop mismatch dominated the comparison.

The aperture edge answers it directly and independently.  It is the boundary of
the exposed picture area -- a physical, content-independent line running the
length of the film, and it lives in the WALKED picture band, whereas the
correction is driven by perforations in the EXTRAPOLATED bands.  So its residual
frame-to-frame motion is not circular: nothing in the rectification was fitted
to it.

Per frame we find the two across-film aperture edges (the strong intensity step
between exposed picture and unexposed margin) and report

    edge position sd        how much the picture wanders across the film
    edge separation sd      how much it breathes across the film
    frame-to-frame |d|      the jitter a viewer actually sees

    python -m unwrapping.eval.perf_stillness --videos a.mp4 b.mp4 --labels A B \
        --out-dir <dir>
"""

import argparse
import os

import numpy as np


def load_video(path):
    import cv2
    cap = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0)
    cap.release()
    if not fr:
        raise SystemExit(f"no frames read from {path}")
    return np.stack(fr)


def aperture_edges(frames, guess, half_window):
    """Sub-pixel across-film position of the two aperture edges, per frame.

    In these videos the across-film axis is HORIZONTAL (the render rotates so
    the picture stands upright), so the edges are steps in the per-column mean.
    Averaging down the whole column suppresses picture content, which is not
    aligned to the edge.

    `guess` is (left, right) in FRACTIONS of the width and the search is
    confined to +-half_window fractions around each.  A guess is required
    because the two renders show different things: the baseline carries the full
    film width, so an unconstrained "strongest step near the border" search
    locks onto the film/margin edge instead of the aperture and measures a
    1204 px aperture where the truth is 1520 strip rows -- not the same feature
    the perf-locked render's search finds, which would make the comparison
    meaningless.
    """
    n, h, w = frames.shape
    out = np.full((n, 2), np.nan)
    k = np.ones(9) / 9
    hw = int(half_window * w)
    for i in range(n):
        prof = frames[i].mean(axis=0)
        prof = np.convolve(np.pad(prof, 4, mode="edge"), k, "valid")[:w]
        g = np.abs(np.gradient(prof))
        for side in (0, 1):
            c0 = int(guess[side] * w)
            a0, a1 = max(1, c0 - hw), min(w - 1, c0 + hw)
            if a1 - a0 < 5:
                continue
            seg = g[a0:a1]
            j = int(np.argmax(seg))
            if not (1 <= j < len(seg) - 1):
                continue
            a, b, c = seg[j - 1], seg[j], seg[j + 1]
            d = a - 2 * b + c
            sub = 0.5 * (a - c) / d if abs(d) > 1e-9 else 0.0
            out[i, side] = a0 + j + np.clip(sub, -1, 1)
    return out


def report(edges, label, px_per_frame_px=None):
    left, right = edges[:, 0], edges[:, 1]
    ok = np.isfinite(left) & np.isfinite(right)
    left, right = left[ok], right[ok]
    centre = 0.5 * (left + right)
    sep = right - left

    def rob(x):
        return float(np.median(np.abs(x - np.median(x))) * 1.4826)

    d_centre = np.abs(np.diff(centre))
    stats = dict(
        n=int(ok.sum()),
        centre_sd=rob(centre),
        centre_ptp=float(np.percentile(centre, 97.5) - np.percentile(centre, 2.5)),
        sep_sd=rob(sep),
        sep_pct=100 * rob(sep) / max(np.median(sep), 1e-9),
        jitter_med=float(np.median(d_centre)),
        jitter_p90=float(np.percentile(d_centre, 90)),
    )
    print(f"\n--- {label}  (n={stats['n']} frames, aperture "
          f"{np.median(sep):.1f} px wide)")
    print(f"  across-film position   robust sd {stats['centre_sd']:6.2f} px"
          f"   95% range {stats['centre_ptp']:6.2f} px")
    print(f"  aperture width (scale) robust sd {stats['sep_sd']:6.2f} px"
          f"   = {stats['sep_pct']:.2f} %")
    print(f"  frame-to-frame jitter  median {stats['jitter_med']:6.2f} px"
          f"   p90 {stats['jitter_p90']:6.2f} px")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--guesses", nargs="+", required=True,
                    help="One 'left,right' pair of width FRACTIONS per video, "
                         "giving where the aperture edges are expected. Required "
                         "so that every video is measured on the SAME physical "
                         "edge -- see aperture_edges().")
    ap.add_argument("--half-window", type=float, default=0.05,
                    help="Search half-width around each guess, in width fractions.")
    ap.add_argument("--px-per-strip-row", nargs="+", type=float, default=None,
                    help="Optional per-video conversion so results are reported "
                         "in strip rows rather than each video's own pixels.")
    args = ap.parse_args()
    if not (len(args.videos) == len(args.labels) == len(args.guesses)):
        raise SystemExit("--videos, --labels and --guesses must be the same length")
    os.makedirs(args.out_dir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 1, figsize=(14, 8))
    allstats = {}
    for i, (path, lbl, gs) in enumerate(zip(args.videos, args.labels,
                                            args.guesses)):
        F = load_video(path)
        g = tuple(float(x) for x in gs.split(","))
        scale = (args.px_per_strip_row[i] if args.px_per_strip_row else 1.0)
        print(f"{lbl}: {F.shape}  edge guess {g}  scale {scale:.4f} rows/px")
        e = aperture_edges(F, g, args.half_window) * scale
        allstats[lbl] = report(e, lbl)
        c = 0.5 * (e[:, 0] + e[:, 1])
        axs[0].plot(c - np.nanmedian(c), lw=.9, label=lbl)
        axs[1].plot(e[:, 1] - e[:, 0], lw=.9, label=lbl)
        np.save(os.path.join(args.out_dir, f"edges_{lbl}.npy"), e)

    axs[0].set_title("aperture CENTRE across the film, per frame (px, median "
                     "removed) -- this is the sway a viewer sees")
    axs[0].set_xlabel("frame"); axs[0].legend(); axs[0].grid(alpha=.3)
    axs[1].set_title("aperture WIDTH per frame (px) -- across-film scale breathing")
    axs[1].set_xlabel("frame"); axs[1].legend(); axs[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "stillness.png"), dpi=110)
    print(f"\nwrote {args.out_dir}/stillness.png")

    import json
    with open(os.path.join(args.out_dir, "stillness.json"), "w") as f:
        json.dump(allstats, f, indent=2)


if __name__ == "__main__":
    main()
