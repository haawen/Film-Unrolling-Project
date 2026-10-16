"""Option 1: de-sway the unrolled strip at the source, GT-FREE.

The telescoping (roll not stacked straight) leaves the film's z-position (film
width) drifting slowly along the arc -> the horizontal sway in the video. The
only content-independent z-fiducial left (sprockets + edges were cropped away)
is the film's DENSITY ENVELOPE across its width -- a fixed physical brightness
ramp. So: register each local arc-window's z-profile to the global-average
z-profile (the clean ramp) to recover the film's z-drift(arc), smooth it, and
resample the strip in z to flatten it. Downstream framing then sees a still
film.

Stage A (`--drift-only`): compute + plot the drift and sanity-check it is a
coherent slow curve (not noise) before committing to the 2 GB rewrite.
Stage B: apply it -> de-swayed strip .npy.

Usage:
  python -m unwrapping.inr.desway_strip --strip .../wholeroll.npy --drift-only
  python -m unwrapping.inr.desway_strip --strip .../wholeroll.npy \
      --out .../wholeroll_desway.npy
"""

import argparse

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter


def phase1d_z(ref, sig, lim=0.15):
    """Sub-pixel z-shift s: moving sig by +s aligns it to ref (bounded)."""
    n = len(ref)
    R = np.fft.rfft(ref - ref.mean())
    S = np.fft.rfft(sig - sig.mean())
    X = R * np.conj(S)
    X /= np.abs(X) + 1e-12
    r = np.fft.irfft(X, n=n)
    m = max(1, int(lim * n))
    mask = np.zeros(n, bool)
    mask[:m + 1] = mask[-m:] = True
    rr = np.where(mask, r, -np.inf)
    k = int(np.argmax(rr))
    y0, y1, y2 = rr[(k - 1) % n], rr[k], rr[(k + 1) % n]
    if np.isfinite(y0) and np.isfinite(y2):
        d = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / d if abs(d) > 1e-9 else 0.0
    else:
        frac = 0.0
    s = k + frac
    return s - n if s > n / 2 else s


def compute_drift(strip, win=450, step=100, arc_smooth=1500, iters=2):
    """Film z-drift(arc) from envelope registration. Returns drift over all W
    columns (centred), plus the sampled (centers, dz) for plotting."""
    Z, W = strip.shape
    centers = np.arange(win, W - win, step)
    drift_full = np.zeros(W)
    dz = None
    for it in range(iters):
        # reference = global-average z-profile AFTER the current de-drift
        # (sharper ramp each iteration). Sample columns, shift by -drift.
        cs = np.arange(0, W, 400)
        profs = []
        for c in cs:
            p = np.asarray(strip[:, c], np.float32)
            if drift_full[c]:
                p = np.interp(np.arange(Z) + drift_full[c], np.arange(Z), p)
            profs.append(p)
        ref = np.mean(profs, axis=0)
        dz = np.zeros(len(centers))
        for i, c in enumerate(centers):
            prof = np.asarray(strip[:, c - win:c + win:4], np.float32).mean(1)
            # pre-shift by current estimate so we measure the residual
            if drift_full[c]:
                prof = np.interp(np.arange(Z) + drift_full[c],
                                 np.arange(Z), prof)
            dz[i] = drift_full[c] + phase1d_z(ref, prof)
        # robust + smooth over arc
        med = median_filter(dz, 7, mode="nearest")
        resid = dz - med
        mad = np.median(np.abs(resid - np.median(resid))) * 1.4826 + 1e-6
        dz = np.where(np.abs(resid) > 4 * mad, med, dz)
        drift_full = np.interp(np.arange(W), centers, dz)
        drift_full = gaussian_filter1d(drift_full, arc_smooth)
        drift_full -= np.median(drift_full)
    return drift_full, centers, dz


def film_centroid(strip, step=200):
    """Content-weighted film z-centroid along arc (a coherence cross-check)."""
    Z, W = strip.shape
    cs = np.arange(0, W, step)
    samp = np.asarray(strip[:, ::200], np.float32)
    lo, hi = np.percentile(samp, [1, 99])
    cen = np.full(len(cs), np.nan)
    for i, c in enumerate(cs):
        col = 1.0 - np.clip((np.asarray(strip[:, c], np.float32) - lo)
                            / (hi - lo + 1e-6), 0, 1)
        col = gaussian_filter1d(col, 8)
        col = np.clip(col - (col.min() + 0.3 * (col.max() - col.min())), 0, None)
        if col.sum() > 1:
            cen[i] = (col * np.arange(Z)).sum() / col.sum()
    return cs, cen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strip", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--drift-only", action="store_true")
    ap.add_argument("--arc-smooth", type=float, default=1500,
                    help="drift smoothing (columns); ~1-2 cells.")
    ap.add_argument("--chunk", type=int, default=4000)
    args = ap.parse_args()

    strip = np.load(args.strip, mmap_mode="r")
    Z, W = strip.shape
    print(f"strip {strip.shape}; computing envelope drift...", flush=True)
    drift, centers, dz = compute_drift(strip, arc_smooth=args.arc_smooth)
    print(f"  drift: range {drift.max()-drift.min():.0f}px  std {drift.std():.0f}px",
          flush=True)

    # coherence cross-check vs the (independent) content-weighted centroid
    cs, cen = film_centroid(strip)
    good = ~np.isnan(cen)
    cen_s = gaussian_filter1d(np.interp(np.arange(W), cs[good], cen[good]),
                              args.arc_smooth)
    corr = np.corrcoef(drift, cen_s - cen_s.mean())[0, 1]
    print(f"  cross-check corr(envelope-drift, centroid-drift) = {corr:.2f} "
          f"({'coherent -> real' if abs(corr) > 0.5 else 'weak -> suspect'})",
          flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(11, 4))
    plt.plot(centers, dz, ".", ms=2, alpha=0.3, label="envelope shift (raw)")
    plt.plot(np.arange(W), drift, lw=2, label="drift (smoothed)")
    plt.plot(np.arange(W), cen_s - cen_s.mean(), lw=1.5,
             label="centroid drift (cross-check)")
    plt.xlabel("arc (film length)"); plt.ylabel("z drift (px)")
    plt.legend(); plt.title("film z-drift along the strip (the sway)")
    plt.tight_layout(); plt.savefig("data/_desway_drift.png", dpi=110)
    print("  saved data/_desway_drift.png", flush=True)

    if args.drift_only or not args.out:
        return

    print(f"Applying de-sway -> {args.out} (chunked)...", flush=True)
    out = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.float32,
                                    shape=(Z, W))
    zc = np.arange(Z)
    for c0 in range(0, W, args.chunk):
        c1 = min(W, c0 + args.chunk)
        block = np.asarray(strip[:, c0:c1], np.float32)
        for j in range(c1 - c0):
            d = drift[c0 + j]
            out[:, c0 + j] = np.interp(zc + d, zc, block[:, j]) if d else block[:, j]
        if c0 % (args.chunk * 10) == 0:
            print(f"  {c0}/{W}", flush=True)
    out.flush()
    # verify: centroid of de-swayed strip should be flatter
    cs2, cen2 = film_centroid(out)
    g2 = ~np.isnan(cen2)
    print(f"Done. centroid smooth-range BEFORE {(cen_s.max()-cen_s.min()):.0f}px "
          f"-> AFTER {np.ptp(gaussian_filter1d(cen2[g2], 20)):.0f}px "
          f"(flatter = de-swayed)", flush=True)


if __name__ == "__main__":
    main()
