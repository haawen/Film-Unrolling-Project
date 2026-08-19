"""Measure the DELIVERED unroll path against the segmented emulsion.

Motivation (2026-07-27). The residual video artifacts include bright/dark
VERTICAL bands in the strip. A strip column is one azimuth of one winding sampled
over all z, so a vertical band = an error that is constant across z at that
azimuth = an IN-PLANE fit error. The obvious mechanism is the rendered path
sitting off the emulsion centre there, sampling film base or air instead.

Every previous fit check was either (a) a visual overlay at a handful of anchors,
or (b) `deviation from a smooth surface` — which is blind by construction to an
error that is itself smooth, e.g. a systematic bias off the emulsion centre. This
measures the thing directly: signed perpendicular distance from the delivered
curve to the local emulsion centre, everywhere.

Three suspects it is built to separate:
  1. `recenter_on_emulsion` low-passes its correction (sigma 20 points), so any
     bias varying faster than that is left uncorrected by design.
  2. z-heal rebuilds flagged bins by interpolating RADIUS across z and placing
     them on the column ray -- those rebuilt points are never re-locked onto the
     emulsion.
  3. windings not matched at an anchor are filled by COPYING the nearest z
     anchor's curve, which is off by that z-gap's drift.

Outputs `path_vs_emulsion.npz`:
  off   (n_anchor, n_wind, NB)  signed perpendicular offset to the emulsion
                                centre, px, +ve = outward. NaN where no emulsion
                                within `--search`.
  cov   (n_anchor, n_wind, NB)  emulsion found within the search window
  healed/copied flags per (anchor, winding) where derivable
plus a summary png. Run on Merlin7 (needs the seg); ~5 min/chunk.

Usage:
  python Scripts/diag_path_vs_emulsion.py \
      --geom unwrapping/inr/results/walk_dense_v11/matched_walks.npz \
      --seg-dir 01_Mickey_3d --out-dir unwrapping/inr/results/fitdiag_v12
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unwrapping.inr.unwrap_data_3d import discover_volumes
from unwrapping.inr.unroll_continuous import load_seg_slices

TWO_PI = 2.0 * math.pi


def offsets_to_emulsion(path, emul, search, sub=1):
    """Signed perpendicular offset (px) from each path point to the centre of the
    emulsion run it sits in. +ve = outward along the path normal. NaN if no
    class-2 pixel within +-search.

    Uses the run the path is INSIDE (or nearest run centre), not the centroid of
    everything in the window, so a neighbouring band 20px away cannot drag the
    measurement -- the same reasoning as the walk's nearest-run snap.
    """
    H, W = emul.shape
    p = path[::sub]
    N = len(p)
    if N < 5:
        return np.full(N, np.nan)
    d = np.gradient(p, axis=0)
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
    ts = np.arange(-search, search + 1.0)
    xs = p[:, 0:1] + ts[None, :] * nrm[:, 0:1]
    ys = p[:, 1:2] + ts[None, :] * nrm[:, 1:2]
    ix = np.clip(np.round(xs).astype(int), 0, W - 1)
    iy = np.clip(np.round(ys).astype(int), 0, H - 1)
    v = emul[iy, ix] > 0.5                                  # (N, M)
    M = len(ts)
    zero = M // 2
    out = np.full(N, np.nan)

    # ON-BAND rows: the path sits inside an emulsion run -> offset to that run's
    # centre. Grow the run outward from t=0, vectorized over all N points (a few
    # steps, not a per-point loop: at search=8 that is 8 iterations each way).
    inside = v[:, zero].copy()
    lo = np.full(N, zero); active = inside.copy()
    for k in range(1, zero + 1):
        c = zero - k
        active &= v[:, c]
        lo = np.where(active, c, lo)
    hi = np.full(N, zero); active = inside.copy()
    for k in range(1, M - zero):
        c = zero + k
        active &= v[:, c]
        hi = np.where(active, c, hi)
    # a run touching the window edge is TRUNCATED -> its centre is biased
    # toward the path; report NaN rather than a number that is wrong low.
    trunc = inside & ((lo == 0) | (hi == M - 1))
    ok = inside & ~trunc
    out[ok] = 0.5 * (ts[lo[ok]] + ts[hi[ok]])

    # OFF-BAND rows: no emulsion at the path itself. Report the signed distance
    # to the NEAREST emulsion pixel — for a path that has left its band that is
    # the meaningful number (how far off it is), and it is what a re-lock would
    # have to move.
    off = ~inside & v.any(axis=1)          # path is off its band
    if off.any():
        dist = np.where(v[off], np.abs(ts)[None, :], np.inf)
        j = np.argmin(dist, axis=1)
        out[off] = ts[j]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", required=True, help="matched_walks.npz (delivered)")
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--search", type=float, default=14.0,
                    help="Perpendicular half-window (px). MUST exceed the band "
                         "half-thickness (~8.5px for the 17px Mickey emulsion): "
                         "with a smaller window the run is truncated by the "
                         "window itself and the measured centre is dragged toward "
                         "the path, under-reporting the offset ~2x (and reporting "
                         "ZERO for offsets below ~2.5px, where the whole window "
                         "sits inside the band). Safe at 14 because the run is "
                         "GROWN from the path and stops at the film base, so it "
                         "cannot merge a neighbouring emulsion.")
    ap.add_argument("--bins", type=int, default=1200, help="azimuth bins")
    ap.add_argument("--sub", type=int, default=4,
                    help="Sample every Nth path point (speed; 4 -> ~1 per 4px).")
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--zooms", type=int, default=8,
                    help="Save this many zoomed path-vs-emulsion overlays at the "
                         "WORST systematic offsets (one per winding) so the "
                         "numbers can be checked by eye. 0 = none.")
    ap.add_argument("--ct-dir", default="01_Mickey_hdf",
                    help="Raw CT slices, for the zoom overlays.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    g = np.load(args.geom, allow_pickle=True)
    z_anchor = np.asarray(g["z_anchor"], float)
    cx_a = np.asarray(g["cx_a"], float); cy_a = np.asarray(g["cy_a"], float)
    seam = float(g["seam"]); paths = g["paths"]
    n = len(paths); nw = len(paths[0])
    print(f"{n} anchors, {nw} windings, seam {math.degrees(seam):.1f}deg", flush=True)

    from unwrapping.inr.unroll_continuous import discover_ct_slices
    ctmap = discover_ct_slices(args.ct_dir) if args.zooms > 0 else {}
    pairs = discover_volumes(args.seg_dir)
    if args.max_chunks:
        pairs = pairs[:args.max_chunks]
    NB = args.bins
    phi_bins = np.linspace(0, TWO_PI, NB + 1)
    OFF = np.full((n, nw, NB), np.nan, np.float32)
    COV = np.zeros((n, nw, NB), np.float32)

    # map each anchor z to (chunk, slice) via the chunk's z range in its filename
    used = 0
    for ci, (vol_p, probs_p, z0, z1) in enumerate(pairs):
        idx = np.where((z_anchor >= z0) & (z_anchor <= z1))[0]
        if idx.size == 0:
            continue
        segs = load_seg_slices(probs_p, 1.0)
        zmap = {z0 + si: s for s, si in segs}
        for a in idx:
            za = int(round(z_anchor[a]))
            if za not in zmap:
                za = min(zmap, key=lambda t: abs(t - z_anchor[a]))
            emul = (zmap[za] == 2).astype(np.float32)
            for wi in range(nw):
                p = paths[a][wi]
                o = offsets_to_emulsion(p, emul, args.search, args.sub)
                q = p[::args.sub]
                ph = np.mod(np.arctan2(q[:, 1] - cy_a[a], q[:, 0] - cx_a[a]) - seam,
                            TWO_PI)
                b = np.clip(np.digitize(ph, phi_bins) - 1, 0, NB - 1)
                good = ~np.isnan(o)
                if good.any():
                    np.add.at(COV[a, wi], b[good], 1.0)
                    s = np.zeros(NB); np.add.at(s, b[good], o[good])
                    with np.errstate(invalid="ignore", divide="ignore"):
                        OFF[a, wi] = np.where(COV[a, wi] > 0, s / COV[a, wi], np.nan)
            used += 1
        print(f"  chunk {ci+1}/{len(pairs)} z{z0}-{z1}: {idx.size} anchors "
              f"(total {used})", flush=True)
        del segs, zmap

    np.savez_compressed(os.path.join(args.out_dir, "path_vs_emulsion.npz"),
                        off=OFF, cov=COV, z_anchor=z_anchor, seam=seam)

    valid = np.isfinite(OFF)
    print(f"\nemulsion found for {100*valid.mean():.1f}% of (anchor,winding,bin)")
    ab = np.abs(OFF[valid])
    print(f"|offset to emulsion centre|: med {np.median(ab):.2f}px  "
          f"p90 {np.percentile(ab,90):.2f}px  p99 {np.percentile(ab,99):.2f}px  "
          f"max {ab.max():.1f}px")
    permed = np.nanmedian(np.where(valid, OFF, np.nan), axis=0)      # (nw, NB)
    print(f"z-median offset per (winding,bin): "
          f"p90 |.| {np.nanpercentile(np.abs(permed),90):.2f}px  "
          f"max {np.nanmax(np.abs(permed)):.2f}px")
    print(f"  (a SYSTEMATIC bias survives the z-median; per-slice noise does not "
          f"-- so this row is the part that makes a VERTICAL band)")

    # ── VISUAL: zoomed overlays where the systematic offset is WORST, so the
    #    numbers above can be checked by eye against the actual emulsion ──
    if args.zooms > 0:
        from unwrapping.inr.unroll_continuous import load_ct, norm_slice
        permed0 = np.nanmedian(np.where(np.isfinite(OFF), OFF, np.nan), axis=0)
        flat = np.abs(np.nan_to_num(permed0))
        # worst (winding, bin), spread across different windings
        order = np.dstack(np.unravel_index(np.argsort(flat, axis=None)[::-1],
                                           flat.shape))[0]
        picks, seen = [], set()
        for wi, b in order:
            if wi in seen or wi < 2 or wi > nw - 3:
                continue
            picks.append((int(wi), int(b))); seen.add(wi)
            if len(picks) >= args.zooms:
                break
        for wi, b in picks:
            # an anchor where this (winding, bin) is actually measured
            cand = np.where(np.isfinite(OFF[:, wi, b]))[0]
            if cand.size == 0:
                continue
            a = int(cand[len(cand) // 2])
            za = int(round(z_anchor[a]))
            # find the chunk/slice again for the seg + CT
            hit = [(vp, pp, s0, s1) for (vp, pp, s0, s1) in pairs
                   if s0 <= za <= s1]
            if not hit:
                continue
            vp, pp, s0, s1 = hit[0]
            segs = load_seg_slices(pp, 1.0)
            zm = {s0 + si: s for s, si in segs}
            zz = za if za in zm else min(zm, key=lambda t: abs(t - za))
            seg2 = zm[zz]
            p = paths[a][wi]
            ph = np.mod(np.arctan2(p[:, 1] - cy_a[a], p[:, 0] - cx_a[a]) - seam,
                        TWO_PI)
            j = int(np.argmin(np.abs(ph - (b + 0.5) * TWO_PI / NB)))
            px, py = p[j]
            R0 = 55
            x0, x1 = int(max(0, px - R0)), int(min(seg2.shape[1], px + R0))
            y0, y1 = int(max(0, py - R0)), int(min(seg2.shape[0], py + R0))
            ct = norm_slice(load_ct(ctmap[zz])) if ctmap.get(zz) is not None else None
            fig2, axz = plt.subplots(1, 2 if ct is not None else 1,
                                     figsize=(11, 5.5), squeeze=False)
            for col, (bg, ttl) in enumerate(
                    ([(ct, "CT")] if ct is not None else []) +
                    [(None, "SEG (emulsion=red, base=grey)")]):
                A = axz[0][col]
                if bg is not None:
                    A.imshow(bg[y0:y1, x0:x1], cmap="gray")
                else:
                    rgb = np.zeros(seg2.shape + (3,), np.float32)
                    rgb[seg2 == 1] = (0.5, 0.5, 0.5); rgb[seg2 == 2] = (1, .15, .15)
                    A.imshow(rgb[y0:y1, x0:x1])
                m = (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)
                A.plot(p[m, 0] - x0, p[m, 1] - y0, "-", color="yellow", lw=1.4)
                A.plot(px - x0, py - y0, "c+", ms=12, mew=2)
                A.set_title(f"{ttl}\nw{wi} phi{360*(b+0.5)/NB:.0f}deg z{zz} "
                            f"— measured offset {permed0[wi, b]:+.1f}px",
                            fontsize=9)
                A.set_xticks([]); A.set_yticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir,
                                     f"zoom_w{wi:02d}_b{b:04d}_z{zz}.png"), dpi=130)
            plt.close(fig2)
            del segs, zm
            print(f"  zoom w{wi} bin{b} z{zz} offset {permed0[wi,b]:+.1f}px",
                  flush=True)

    fig, ax = plt.subplots(3, 1, figsize=(14, 11))
    v = np.nanpercentile(np.abs(permed), 98)
    im = ax[0].imshow(permed, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v,
                      extent=[0, 360, nw - 0.5, -0.5])
    ax[0].set_title("z-MEDIAN signed offset of the delivered path from the "
                    "emulsion centre (px). Non-zero = systematic fit bias -> a "
                    "vertical band in the strip.")
    ax[0].set_xlabel("azimuth (deg from seam)"); ax[0].set_ylabel("winding")
    plt.colorbar(im, ax=ax[0])
    cvg = (COV > 0).mean(axis=0)
    im2 = ax[1].imshow(cvg, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                       extent=[0, 360, nw - 0.5, -0.5])
    ax[1].set_title("fraction of anchors where ANY emulsion was found within "
                    f"+-{args.search}px of the path (low = path is off the band)")
    ax[1].set_xlabel("azimuth (deg from seam)"); ax[1].set_ylabel("winding")
    plt.colorbar(im2, ax=ax[1])
    ax[2].hist(ab, bins=120, color="C0")
    ax[2].set_xlabel("|offset to emulsion centre| (px)"); ax[2].set_yscale("log")
    ax[2].set_title("distribution over all (anchor, winding, azimuth)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "path_vs_emulsion.png"), dpi=115)
    print(f"\nDone -> {args.out_dir}")


if __name__ == "__main__":
    main()
