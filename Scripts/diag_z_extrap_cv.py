"""Can the walked winding geometry be EXTRAPOLATED in z into the sprocket bands?

The sprocket (perforation) rows sit ~350 CT slices OUTSIDE the walked/segmented
picture band (z832-2015). To render them we either (a) extrapolate each winding's
(x,y) curve in z from the walked band -- no new segmentation, no labels -- or
(b) segment + walk the perf band too. This script decides which, using ONLY the
already-delivered geometry: hold out the last H anchors at each end of the walked
band, fit the z-model on the anchors just inside, predict the held-out block, and
measure the error against the ACTUAL walked curves there.

Acceptance threshold is physics, not taste: the class-2 emulsion band is 4.0px
thick (p10 3.2), so the sampling line must land within ~1-2px of truth to still
read emulsion. The error that matters is the component along the path NORMAL
(radial-ish) -- an along-path error only slides the sample along the film.

Reported for contrast: `hold` = what the renderer does today
(interp1d fill_value=(X[0],X[-1])), i.e. freeze the last walked slice.

Note what this CAN'T see: it cross-validates INSIDE the picture band, so it
cannot detect physically different behaviour past the picture edge (film edge
curl against the packaging). That residual risk is settled by overlaying the
extrapolated curves on real perf-band CT slices.

Usage:
  python Scripts/diag_z_extrap_cv.py \
      --geom unwrapping/inr/results/walk_full_final/matched_walks.npz \
      --out-dir unwrapping/inr/results/zextrap_cv
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def fit_predict(z_fit, V_fit, z_pred, deg):
    """Least-squares polynomial in z, per column, all columns at once.

    z_fit (F,), V_fit (F,K) -> (P,K) prediction at z_pred (P,).
    deg 0 = hold (constant, fitted as the mean of the fit window's LAST anchor).
    """
    if deg == 0:
        return np.repeat(V_fit[-1][None, :], len(z_pred), axis=0)
    t0 = z_fit[-1]                                  # centre on the boundary anchor
    A = np.vander(z_fit - t0, deg + 1)              # (F, deg+1)
    coef, *_ = np.linalg.lstsq(A, V_fit, rcond=None)
    return np.vander(z_pred - t0, deg + 1) @ coef


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", required=True, help="matched_walks.npz (delivered)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--holdout", type=int, default=350,
                    help="Anchors held out at each end = the extrapolation "
                         "distance we actually need (~350 slices per perf band).")
    ap.add_argument("--fit-windows", type=int, nargs="+", default=[100, 200, 400],
                    help="Anchors used for the fit, adjacent to the holdout.")
    ap.add_argument("--degrees", type=int, nargs="+", default=[0, 1, 2],
                    help="0=hold (today's renderer), 1=linear, 2=quadratic.")
    ap.add_argument("--edge-windings", type=int, default=2,
                    help="Exclude this many windings at EACH end from the pooled "
                         "summary: they carry the film's two ends (tongue / "
                         "termination), are partial and non-circular, and are cut "
                         "from the video anyway. Still reported per-winding.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    g = np.load(args.geom, allow_pickle=True)
    z = np.asarray(g["z_anchor"], float)
    paths = g["paths"]
    n = len(paths); nw = len(paths[0])
    order = np.argsort(z)
    z = z[order]
    print(f"{n} anchors z{z.min():.0f}-{z.max():.0f}, {nw} windings", flush=True)
    dz = np.diff(z)
    print(f"anchor z-spacing: med {np.median(dz):.2f} max {dz.max():.2f} "
          f"(1.0 = every slice walked)", flush=True)

    H = args.holdout
    if 2 * H + max(args.fit_windows) > n:
        raise SystemExit(f"holdout {H} x2 + fit {max(args.fit_windows)} > {n} anchors")

    # sides: (name, holdout indices ordered by INCREASING extrapolation distance,
    #         fit-window slice builder)
    idx = np.arange(n)
    sides = {
        "low":  (idx[H - 1::-1], lambda F: idx[H:H + F][::-1]),   # fit inward->out
        "high": (idx[n - H:],    lambda F: idx[n - H - F:n - H]),
    }

    # accumulators: [side][deg][F] -> list over distance d of pooled |err| samples
    combos = [(s, d, F) for s in sides for d in args.degrees
              for F in args.fit_windows]
    acc = {c: ([[] for _ in range(H)], [[] for _ in range(H)]) for c in combos}
    per_w = {c: {} for c in combos}
    mv = {s: [] for s in sides}            # how far the curve actually moves

    for wi in range(nw):                   # winding OUTER: stack the geometry once
        P = np.stack([np.asarray(paths[a][wi], np.float32) for a in order])
        X, Y = P[:, :, 0].astype(float), P[:, :, 1].astype(float)      # (n,K)
        # mask CLAMPED columns: resample_phi repeats a partial walk's endpoint, so
        # those columns are a frozen point, not geometry (windings 0/1 are 51%/93%
        # clamped -- the same trap that inflated the v13 arc length).
        ds = np.hypot(np.diff(X, axis=1), np.diff(Y, axis=1))
        good = np.median(np.concatenate([ds[:, :1], ds], axis=1), axis=0) > 1e-6
        if good.sum() < 32:
            print(f"  w{wi:02d}: skipped ({good.sum()} unclamped columns)", flush=True)
            continue
        Xg, Yg = X[:, good], Y[:, good]
        pooled = args.edge_windings <= wi < nw - args.edge_windings
        for side, (ho, fitsel) in sides.items():
            if pooled:
                mv[side].append(np.median(np.hypot(Xg[ho[-1]] - Xg[ho[0]],
                                                   Yg[ho[-1]] - Yg[ho[0]])))
            # local path normal from the ACTUAL curve at each held-out anchor
            tx = np.gradient(Xg[ho], axis=1); ty = np.gradient(Yg[ho], axis=1)
            L = np.hypot(tx, ty) + 1e-9
            nx, ny = -ty / L, tx / L
            for deg in args.degrees:
                for F in args.fit_windows:
                    fi = fitsel(F); zf = z[fi]
                    ex = fit_predict(zf, Xg[fi], z[ho], deg) - Xg[ho]
                    ey = fit_predict(zf, Yg[fi], z[ho], deg) - Yg[ho]
                    en = np.abs(ex * nx + ey * ny); ee = np.hypot(ex, ey)
                    c = (side, deg, F)
                    if pooled:
                        for d in range(H):
                            acc[c][0][d].append(en[d]); acc[c][1][d].append(ee[d])
                    per_w[c][wi] = dict(normal_med=float(np.median(en[-1])),
                                        normal_p90=float(np.percentile(en[-1], 90)),
                                        euclid_med=float(np.median(ee[-1])),
                                        clamped_frac=float(1.0 - good.mean()))
        print(f"  w{wi:02d}: {good.sum()} cols "
              f"({100 * (1 - good.mean()):.0f}% clamped)"
              f"{'' if pooled else '  [edge winding, not pooled]'}", flush=True)

    res = {s: {d: {} for d in args.degrees} for s in sides}
    for (side, deg, F), (an, ae) in acc.items():
        stats = {}
        for key, a in (("normal", an), ("euclid", ae)):
            stats[key + "_med"] = np.array([np.median(np.concatenate(x)) for x in a])
            stats[key + "_p90"] = np.array([np.percentile(np.concatenate(x), 90)
                                            for x in a])
        res[side][deg][F] = stats
        print(f"  {side:4s} deg{deg} F{F:4d}: |normal| med "
              f"{stats['normal_med'][-1]:8.2f}px p90 "
              f"{stats['normal_p90'][-1]:8.2f}px @d={H}  "
              f"(med @d=100: {stats['normal_med'][min(99, H - 1)]:7.2f}px)",
              flush=True)
    # ── error vs the TARGET'S OWN NOISE. The walks are independent per z (per-slice
    #    seed+walk; cross-z coupling is identity-only), so each actual curve carries
    #    sub-px-to-few-px per-slice jitter that NO z-model can predict. If the CV
    #    error is of this order, the limit is the target's noise, not the fit -- and
    #    an extrapolated path is then as close to the true film as a walked one. ──
    from scipy.ndimage import gaussian_filter1d
    jn = []
    for wi in range(args.edge_windings, nw - args.edge_windings):
        P = np.stack([np.asarray(paths[a][wi], np.float32) for a in order])
        X, Y = P[:, :, 0].astype(float), P[:, :, 1].astype(float)
        ds = np.hypot(np.diff(X, axis=1), np.diff(Y, axis=1))
        good = np.median(np.concatenate([ds[:, :1], ds], axis=1), axis=0) > 1e-6
        if good.sum() < 32:
            continue
        Xg, Yg = X[:, good], Y[:, good]
        rx = Xg - gaussian_filter1d(Xg, 10.0, axis=0, mode="nearest")
        ry = Yg - gaussian_filter1d(Yg, 10.0, axis=0, mode="nearest")
        tx = np.gradient(Xg, axis=1); ty = np.gradient(Yg, axis=1)
        L = np.hypot(tx, ty) + 1e-9
        jn.append(np.abs(rx * (-ty / L) + ry * (tx / L)))
    jn = np.concatenate([j.ravel() for j in jn])
    jitter = dict(med=float(np.median(jn)), p90=float(np.percentile(jn, 90)))
    print(f"  per-slice walk JITTER about the smooth z-trend (sigma=10): "
          f"|normal| med {jitter['med']:.2f}px p90 {jitter['p90']:.2f}px "
          f"-- the irreducible floor for ANY z-model", flush=True)

    sig = {s: float(np.median(v)) for s, v in mv.items()}
    for s, v in sig.items():
        print(f"  {s}: curve MOVES {v:.1f}px over the {H}-anchor span "
              f"(= the signal the fit must track)", flush=True)

    # ── plot: |normal| error vs extrapolation distance ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, side in zip(axes, ["low", "high"]):
        for deg in args.degrees:
            for F in args.fit_windows:
                s = res[side][deg][F]
                lbl = {0: "hold", 1: "linear", 2: "quad"}[deg] + f" F={F}"
                ls = {0: ":", 1: "-", 2: "--"}[deg]
                ax.plot(np.arange(1, H + 1), s["normal_med"], ls, lw=1.2, label=lbl)
        ax.axhline(2.0, color="k", lw=0.8)
        ax.axhline(4.0, color="k", lw=0.6, ls="--")
        ax.set_yscale("log"); ax.set_xlabel("extrapolation distance (slices)")
        ax.set_title(f"{side}-z end (curve moves {sig[side]:.0f}px over {H})")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("median |error along path normal| (px)")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle("z-extrapolation cross-validation: can we reach the sprocket "
                 "bands without segmenting them?  (solid line = 2px budget, "
                 "dashed = 4px emulsion thickness)", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "zextrap_cv.png"), dpi=130)
    plt.close()

    out = {"n_anchors": int(n), "n_windings": int(nw), "holdout": H,
           "z_min": float(z.min()), "z_max": float(z.max()),
           "curve_movement_px": sig, "walk_jitter_px": jitter,
           "per_winding": {f"{s}_deg{d}_F{F}": v
                           for (s, d, F), v in per_w.items()},
           "curves": {side: {str(deg): {str(F):
                             {k: v.tolist() for k, v in res[side][deg][F].items()}
                             for F in args.fit_windows} for deg in args.degrees}
                      for side in sides}}
    with open(os.path.join(args.out_dir, "zextrap_cv.json"), "w") as f:
        json.dump(out, f, indent=1)

    # ── verdict ──
    print("\nVERDICT (median |normal| error at the full "
          f"{H}-slice extrapolation distance):")
    best = None
    for side in sides:
        for deg in args.degrees:
            if deg == 0:
                continue
            for F in args.fit_windows:
                v = res[side][deg][F]["normal_med"][-1]
                if best is None or v < best[0]:
                    best = (v, side, deg, F)
    for deg in args.degrees:
        for F in args.fit_windows:
            lo = res["low"][deg][F]["normal_med"][-1]
            hi = res["high"][deg][F]["normal_med"][-1]
            print(f"  deg{deg} F{F:4d}: low {lo:8.2f}px | high {hi:8.2f}px")
    v, side, deg, F = best
    print(f"  best model: deg{deg} F={F} (worst side {max(res['low'][deg][F]['normal_med'][-1], res['high'][deg][F]['normal_med'][-1]):.2f}px)")
    print(f"  Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
