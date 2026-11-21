"""Thin vertical lines at the start (outer end) of the unrolled video: are they the
film-end windings' unhealed jumps / clamp stubs?

z_heal_windings never touches kept[0] / kept[-1] (written for the old scan, where
those were the film's ragged ends). On the new scan the real termination track is
already dropped by --min-coverage, so kept[-1] is an ordinary, near-complete
winding -- and any jump or clamped stub on it is left in the render.

Inputs: v13 --dump-geometry npz (2 inner + 2 outer kept windings, pre-heal and
final) and the rendered strip + z_index. For each dumped winding:
  * LINE map from the strip: residual of each pixel vs the running z-median of its
    column (9 rows). A "thin vertical line" in the video = a strip ROW that is an
    outlier along a long run of columns.
  * BAD map from the final geometry: (anchor, column) bins whose radius is > thresh
    off the winding's running z-median (61 anchors), or whose path is CLAMPED
    (zero-length step = resample_phi endpoint repeat).
  * how much of the line energy sits on BAD bins, and the same for the pre-heal
    geometry of the INTERIOR (healed) windings as a control.

    python Scripts/diag_end_windings.py --geo geo.npz --strip wholeroll.npy \
        --zidx z_index.npy --out-dir diag/end_windings
"""
import argparse
import json
import os

import numpy as np
from scipy.ndimage import median_filter

TWO_PI = 2 * np.pi


def radius(X, Y, cx, cy):
    return np.hypot(X - cx[:, None], Y - cy[:, None])


def bad_masks(X, Y, cx, cy, thresh, win=61):
    R = radius(X, Y, cx, cy)
    Rm = median_filter(R, size=(win, 1), mode="nearest")
    jump = np.abs(R - Rm) > thresh
    step = np.hypot(np.diff(X, axis=1), np.diff(Y, axis=1))
    clamp = np.zeros_like(jump)
    clamp[:, 1:] = step < 0.05
    clamp[:, 0] = clamp[:, 1]
    return R, jump, clamp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", required=True)
    ap.add_argument("--strip", required=True)
    ap.add_argument("--zidx", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--thresh", type=float, default=7.5)
    ap.add_argument("--eps-col", type=float, default=1.0, help="px_per_col")
    ap.add_argument("--ct-dir", default=None,
                    help="CT h5 dir (Mickey_merged_{z:04d}.h5) -> fit-check overlays")
    ap.add_argument("--n-overlay", type=int, default=5)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    G = np.load(a.geo)
    idx = [int(i) for i in G["idx"]]
    Kf = G["K_final"]; nk = len(Kf)
    cx, cy = G["cx_a"], G["cy_a"]; za = G["z_anchor"]; eps = float(G["eps_phi"])
    S = np.load(a.strip, mmap_mode="r"); zrow = np.load(a.zidx)
    W = S.shape[1]

    def bridge_after(i):                 # replicate the v13 seam-bridge width
        X, Y = G[f"fin_X{i}"], G[f"fin_Y{i}"]
        r_seam = float(np.median(np.hypot(X[:, -1] - cx, Y[:, -1] - cy)))
        return int(round(r_seam * 2 * eps / a.eps_col))

    start = {}
    start[0] = 0
    if 1 in idx:
        start[1] = int(Kf[0]) + bridge_after(0)
    start[nk - 1] = W - int(Kf[nk - 1])
    if nk - 2 in idx:
        start[nk - 2] = start[nk - 1] - bridge_after(nk - 2) - int(Kf[nk - 2])
    # rows that ARE anchors (dense cache) -> per-row geometry is exact there
    a_of_row = {int(z): int(np.argmin(np.abs(za - z))) for z in zrow}
    rows = [j for j, z in enumerate(zrow) if abs(za[a_of_row[int(z)]] - z) < 0.5]
    rows = np.array(rows); ra = np.array([a_of_row[int(zrow[j])] for j in rows])
    print(f"strip {S.shape}; {nk} kept windings; dumped {idx}; "
          f"{len(rows)} strip rows sit exactly on an anchor")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = {}
    for i in idx:
        K = int(Kf[i]); c0 = start[i]
        slab = np.asarray(S[:, c0:c0 + K], np.float32)
        med = median_filter(slab, size=(9, 1), mode="nearest")
        d = slab - med
        sig = 1.4826 * float(np.median(np.abs(d[rows])))
        Xf, Yf = G[f"fin_X{i}"], G[f"fin_Y{i}"]
        Rf, jf, cf = bad_masks(Xf, Yf, cx, cy, a.thresh)
        Xp, Yp = G[f"pre_X{i}"], G[f"pre_Y{i}"]
        Rp, jp, cp = bad_masks(Xp, Yp, cx, cy, a.thresh)
        # z_heal_windings' own gates, on the pre-heal geometry
        th = float(G["seam"]) + G[f"pre_phi{i}"][None, :]
        ta = np.arctan2(Yp - cy[:, None], Xp - cx[:, None])
        onray = np.abs((ta - th + np.pi) % TWO_PI - np.pi) <= np.radians(3.0)
        rng = Rp.max(axis=1) - Rp.min(axis=1)
        gates = {"p85_radius_range_px (skip if >150)": round(float(np.percentile(rng, 85)), 1),
                 "onray_frac (skip if <0.5)": round(float(onray.mean()), 4),
                 "complete_anchor_frac (heal-ends needs >=0.8)":
                     round(float((onray.mean(axis=1) >= 0.9).mean()), 4),
                 "radius_range_p50/p95/max": [round(float(v), 1) for v in
                                              np.percentile(rng, [50, 95, 100])]}
        print(f"  w{i} heal gates: {gates}")
        bad = (jf | cf)[ra]                      # (rows, K) aligned with the strip
        line = np.abs(d[rows]) > 4 * sig
        # a "line row" = an outlier along >= 15% of the winding's columns
        row_frac = line.mean(axis=1)
        line_rows = row_frac > 0.15
        on_bad = float(line[bad].mean()) if bad.any() else float("nan")
        on_good = float(line[~bad].mean())
        frac_line_on_bad = float((line & bad).sum() / max(1, line.sum()))
        # rows whose geometry is mostly bad vs rows that are line rows
        bad_row = bad.mean(axis=1) > 0.15
        both = int((bad_row & line_rows).sum())
        rr = float(np.median(Rf))
        rep = {"winding": i, "kept_of": nk, "r_med": round(rr, 1), "cols": K,
               "heal_gates": gates,
               "strip_cols": [c0, c0 + K],
               "pre_jump_frac": round(float(jp.mean()), 5),
               "pre_clamp_frac": round(float(cp.mean()), 5),
               "fin_jump_frac": round(float(jf.mean()), 5),
               "fin_clamp_frac": round(float(cf.mean()), 5),
               "anchors_with_any_bad_bin": int((jf | cf).any(axis=1).sum()),
               "line_rows": int(line_rows.sum()), "bad_rows": int(bad_row.sum()),
               "line_and_bad_rows": both,
               "P(line|bad bin)": round(on_bad, 4),
               "P(line|good bin)": round(on_good, 4),
               "frac_of_line_px_on_bad_bins": round(frac_line_on_bad, 4),
               "line_row_z": [int(zrow[rows[j]]) for j in np.where(line_rows)[0]][:80],
               "bad_row_z": [int(zrow[rows[j]]) for j in np.where(bad_row)[0]][:80]}
        report[f"w{i}"] = rep
        print(json.dumps({k: v for k, v in rep.items()
                          if k not in ("line_row_z", "bad_row_z")}))
        print(f"   line rows z: {rep['line_row_z'][:40]}")
        print(f"   bad  rows z: {rep['bad_row_z'][:40]}")
        # figure: signed residual | bad mask | radius deviation, picture-band rows
        step = max(1, K // 1500)
        fig, ax = plt.subplots(1, 3, figsize=(18, 7), sharey=True)
        zr = zrow[rows]
        ext = [0, K, zr[-1], zr[0]]
        ax[0].imshow(np.clip(d[rows][:, ::step] / (6 * sig), -1, 1), cmap="RdBu",
                     aspect="auto", extent=ext, interpolation="nearest")
        ax[0].set_title(f"w{i} strip residual vs z-median (+-6 sigma)")
        bm = np.zeros(bad.shape, np.uint8); bm[jf[ra]] = 1; bm[cf[ra]] = 2
        ax[1].imshow(bm[:, ::step], cmap="viridis", vmin=0, vmax=2, aspect="auto",
                     extent=ext, interpolation="nearest")
        ax[1].set_title("final geometry: jump (>thr) = 1, clamped = 2")
        Rm = median_filter(Rf, size=(61, 1), mode="nearest")
        ax[2].imshow(np.clip((Rf - Rm)[ra][:, ::step], -25, 25), cmap="RdBu",
                     aspect="auto", extent=ext, interpolation="nearest")
        ax[2].set_title("final r - running z-median r (+-25 px)")
        for x in ax:
            x.set_xlabel("column in winding")
        ax[0].set_ylabel("z")
        plt.tight_layout()
        plt.savefig(os.path.join(a.out_dir, f"w{i}_line_vs_geometry.png"), dpi=90)
        plt.close()
    with open(os.path.join(a.out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1)

    # ── FIT CHECK overlays: the worst (most-changed) anchors of each end winding,
    #    pre-heal path (red) vs final path (green) on the CT slice ──
    if a.ct_dir:
        import h5py
        for i in idx:
            Xp0, Yp0 = G[f"pre_X{i}"], G[f"pre_Y{i}"]
            Xf, Yf = G[f"fin_X{i}"], G[f"fin_Y{i}"]
            # pre-heal is on the uniform-phi grid, final on the arc-length grid:
            # resample pre onto the final columns' phi so they compare 1:1
            pp, pf = G[f"pre_phi{i}"], G[f"fin_phi{i}"]
            Xp = np.stack([np.interp(pf, pp, r) for r in Xp0])
            Yp = np.stack([np.interp(pf, pp, r) for r in Yp0])
            _, jp, cp = bad_masks(Xp0, Yp0, cx, cy, a.thresh)
            nb = (jp | cp).sum(axis=1)
            order = [int(v) for v in np.argsort(-nb) if nb[v] > 0 and za[v] >= 520]
            pick = []
            for v in order:                       # worst first, >= 40 slices apart
                if all(abs(za[v] - za[p]) >= 40 for p in pick):
                    pick.append(v)
                if len(pick) == a.n_overlay:
                    break
            pick.sort()
            if not pick:
                continue
            fig, axs = plt.subplots(2, len(pick), figsize=(5 * len(pick), 10),
                                    squeeze=False)
            for col, an in enumerate(pick):
                z = int(round(za[an]))
                with h5py.File(os.path.join(a.ct_dir, f"Mickey_merged_{z:04d}.h5"),
                               "r") as fh:
                    img = np.asarray(fh["image"], np.float32)
                lo, hi = np.percentile(img[::8, ::8], [1, 99.5])
                ds = 4
                ax = axs[0, col]
                ax.imshow(img[::ds, ::ds], cmap="gray", vmin=lo, vmax=hi)
                ax.plot(Xp[an] / ds, Yp[an] / ds, "r-", lw=0.6)
                ax.plot(Xf[an] / ds, Yf[an] / ds, "g-", lw=0.6)
                ax.set_title(f"w{i} z{z}: {nb[an]} bad bins pre-heal\n"
                             f"red = pre-heal, green = final", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                # zoom on the largest pre->final displacement
                dd = np.hypot(Xf[an] - Xp[an], Yf[an] - Yp[an])
                c = int(np.argmax(dd)); x0, y0 = Xf[an, c], Yf[an, c]
                h = 120
                ys0, ys1 = int(max(0, y0 - h)), int(min(img.shape[0], y0 + h))
                xs0, xs1 = int(max(0, x0 - h)), int(min(img.shape[1], x0 + h))
                ax = axs[1, col]
                ax.imshow(img[ys0:ys1, xs0:xs1], cmap="gray", vmin=lo, vmax=hi,
                          extent=[xs0, xs1, ys1, ys0])
                for X_, Y_, colr in ((Xp, Yp, "r"), (Xf, Yf, "g")):
                    m = (X_[an] > xs0) & (X_[an] < xs1) & (Y_[an] > ys0) & (Y_[an] < ys1)
                    ax.plot(X_[an][m], Y_[an][m], colr + ".", ms=1.2)
                ax.set_xlim(xs0, xs1); ax.set_ylim(ys1, ys0)
                ax.set_title(f"zoom at max shift {dd[c]:.0f}px (col {c})", fontsize=9)
            plt.tight_layout()
            plt.savefig(os.path.join(a.out_dir, f"w{i}_fitcheck.png"), dpi=80)
            plt.close()
            print(f"  fit-check overlay w{i}: anchors z{[int(round(za[p])) for p in pick]}")
    print("wrote", a.out_dir)


if __name__ == "__main__":
    main()
