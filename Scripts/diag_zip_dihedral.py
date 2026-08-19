"""Is the OLD -> NEW frame relation just SCALE + an AXIS CONVENTION (transpose /
flip), rather than an arbitrary rotation? Test all 8 exhaustively.

Motivation (user's hypothesis, and this project's history): the previous
Mickey bug of exactly this kind was a silent H<->W TRANSPOSE of the segmentation
(near-square images, 3062 vs 3063, so shape did not reveal it). If the two
reconstructions differ by an axis convention, then the relation is

    p_zip = s * D @ (p_old - c_old) + c_zip

with D one of the 8 symmetries of the square (identity, rot90/180/270, and each of
those composed with a mirror; TRANSPOSE = mirror about the main diagonal). A
continuous rotation search -- which is what I wasted time on -- cannot cleanly find
these because a roll of concentric bands is nearly rotation-symmetric, so every
angle scores about the same. Eight discrete candidates have no such degeneracy:
only the correct one puts the 36 delivered winding curves ON the bands.

Discriminator: the RIDGE-sampled objective (raw intensity is useless -- the 22px
film layer on a 26px spacing makes it a plateau) plus, decisively, the distribution
of radial offsets from each mapped curve point to the nearest band centre. The
correct D gives a TIGHT distribution near 0; a wrong one scatters over the whole
+-half-spacing window (measured p90 ~17px for the wrong ones).

Scale is taken as fixed (1.2508, established to 0.15% from film-ring radii, winding
spacing and curve alignment, all rotation-independent) and only the 2-parameter
centre is optimised per candidate; the winner then gets a small scale refinement.

Usage (see Scripts/slurm/slurm_zip_dihedral.sh):
  python Scripts/diag_zip_dihedral.py \
      --geom unwrapping/inr/results/walk_full_final/matched_walks.npz \
      --zip-dir 01_Mickey_sprockets --out-dir unwrapping/inr/results/zip_dihedral \
      --zip-z 500 1960 --scale 1.2508
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter, map_coordinates

# name -> D acting on (dx, dy)   [x' , y'] = D @ [dx, dy]
DIHEDRAL = {
    "identity":       np.array([[1, 0], [0, 1]]),
    "rot90":          np.array([[0, -1], [1, 0]]),
    "rot180":         np.array([[-1, 0], [0, -1]]),
    "rot270":         np.array([[0, 1], [-1, 0]]),
    "transpose":      np.array([[0, 1], [1, 0]]),     # mirror about main diagonal
    "anti_transpose": np.array([[0, -1], [-1, 0]]),
    "flip_x":         np.array([[-1, 0], [0, 1]]),
    "flip_y":         np.array([[1, 0], [0, -1]]),
}


def otsu(a, nbins=512):
    h, e = np.histogram(a, bins=nbins)
    c = 0.5 * (e[1:] + e[:-1])
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    s0 = np.cumsum(h * c); s1 = s0[-1] - s0
    ok = (w0 > 0) & (w1 > 0)
    var = np.zeros(nbins)
    var[ok] = w0[ok] * w1[ok] * (s0[ok] / w0[ok] - s1[ok] / w1[ok]) ** 2
    return float(c[int(np.argmax(var))])


def apply_D(P, c_old, D, s, c_zip):
    """P (N,2) old (x,y) -> zip (x,y). c_old/c_zip are (cy, cx)."""
    d = np.stack([P[:, 0] - c_old[1], P[:, 1] - c_old[0]], 1)      # (dx, dy)
    q = d @ D.T
    return np.stack([s * q[:, 0] + c_zip[1], s * q[:, 1] + c_zip[0]], 1)


def sample(img, Q):
    return map_coordinates(img, [Q[:, 1], Q[:, 0]], order=1, mode="constant", cval=0.0)


def offsets(mask, Q, c_zip, half=13.0, step=0.5):
    """Signed radial distance from each point to the centre of the FILM-BAND RUN it
    sits in; NaN where the point is not on film.

    Replaces an argmax-of-high-passed-profile that was pure noise: over a +-13px
    window it returned offsets matching a UNIFORM distribution (median 5.5-6.0,
    p90 12.5) for ALL EIGHT candidate transforms -- i.e. it discriminated nothing,
    and had never validated anything. Bands are ~22 zip px wide, so their run
    centre is robust to CT grain where a local intensity maximum is not.

    The expected answer is NOT zero: the walked curves sit on the EMULSION, a
    sublayer offset from the band centre. So the discriminator is the SPREAD about
    the median (tight = the transform puts every curve at the same place within its
    band) and NOT the median itself.
    """
    rx = Q[:, 0] - c_zip[1]; ry = Q[:, 1] - c_zip[0]
    L = np.hypot(rx, ry) + 1e-9
    ux, uy = rx / L, ry / L
    ts = np.arange(-half, half + 1e-9, step)
    prof = np.stack([sample(mask.astype(np.float32),
                            np.stack([Q[:, 0] + t * ux, Q[:, 1] + t * uy], 1)) > 0.5
                     for t in ts])                      # (nt, N) bool
    c0 = len(ts) // 2
    on = prof[c0]
    above = prof[c0:]
    below = prof[:c0 + 1][::-1]
    up = np.where(above.all(axis=0), len(ts) - c0, np.argmax(~above, axis=0))
    dn = np.where(below.all(axis=0), c0 + 1, np.argmax(~below, axis=0))
    off = ((up - 1) - (dn - 1)) / 2.0 * step            # run centre in t units
    return np.where(on, off, np.nan)


def robust(v):
    """median, IQR spread and valid fraction of a possibly-NaN offset array."""
    x = v[np.isfinite(v)]
    if x.size < 50:
        return float("nan"), float("nan"), 0.0
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return float(q2), float(q3 - q1), float(x.size / v.size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", required=True)
    ap.add_argument("--zip-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--zip-z", type=int, nargs="+", required=True)
    ap.add_argument("--z-map", type=float, nargs=2, default=[1.2722, -588.5])
    ap.add_argument("--scale", type=float, default=1.2508)
    ap.add_argument("--ridge-sigma", type=float, default=8.0)
    ap.add_argument("--sub", type=int, default=6)
    ap.add_argument("--edge-windings", type=int, default=2)
    ap.add_argument("--control-old-dir", default="",
                    help="old-frame slice dir; runs the identity control first")
    ap.add_argument("--control-z", type=int, nargs="+", default=[848, 1988])
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    g = np.load(args.geom, allow_pickle=True)
    z_anchor = np.asarray(g["z_anchor"], float)
    cx_a = np.asarray(g["cx_a"], float); cy_a = np.asarray(g["cy_a"], float)
    paths = g["paths"]
    a_z, b_z = args.z_map
    all_rows = []

    # ── CONTROL: the OLD frame with identity + scale 1. These very curves were
    #    walked FROM these slices, so the offset distribution MUST come out tight.
    #    If it does not, the measurement is broken and every verdict below is void.
    #    (Skipping this control is exactly how two earlier metrics -- a seam
    #    detector and an argmax offset -- were used for hours while being noise.)
    if args.control_old_dir:
        import glob as _glob
        import re as _re
        omap = {}
        for pp in _glob.glob(os.path.join(args.control_old_dir, "*.h5")):
            mm = _re.search(r"_(\d+)\.h5$", os.path.basename(pp))
            if mm:
                omap[int(mm.group(1))] = pp
        print("=== CONTROL: old frame, identity, scale 1 -> expect a TIGHT IQR ===")
        for zo in args.control_z:
            if zo not in omap:
                print(f"  old z{zo}: missing")
                continue
            with h5py.File(omap[zo], "r") as f:
                raw_o = np.asarray(f["image"], np.float32)
            hh, ee = np.histogram(raw_o, bins=512)
            cc = 0.5 * (ee[1:] + ee[:-1])
            ii = int(np.argmax(hh))
            if hh[ii] / hh.sum() > 0.15 and cc[ii] < np.median(raw_o):
                popu = raw_o[raw_o > cc[ii] + (ee[1] - ee[0])]   # strip FOV background
            else:
                popu = raw_o.ravel()
            mo = raw_o > otsu(popu)
            ao = int(np.argmin(np.abs(z_anchor - zo)))
            nwo = len(paths[ao])
            Po, gdo = [], []
            for wi, w in enumerate(paths[ao]):
                W = np.asarray(w, float)[:: args.sub]
                ds = np.hypot(np.diff(W[:, 0]), np.diff(W[:, 1]))
                k = np.concatenate([[True], ds > 1e-6])
                if wi < args.edge_windings or wi >= nwo - args.edge_windings:
                    k[:] = False
                Po.append(W)
                gdo.append(k)
            Po = np.concatenate(Po)[np.concatenate(gdo)]
            co = (cy_a[ao], cx_a[ao])
            Q = apply_D(Po, co, DIHEDRAL["identity"], 1.0, co)
            med, iqr, frac = robust(offsets(mo, Q, co, half=10.5))
            print(f"  old z{zo}: off med {med:+.2f}  IQR {iqr:.2f} px  on-film "
                  f"{100 * frac:.0f}%", flush=True)
        print()

    for zz in args.zip_z:
        z_old = (zz - b_z) / a_z
        ai = int(np.argmin(np.abs(z_anchor - z_old)))
        p = os.path.join(args.zip_dir, f"Mickey_merged_{zz:04d}.h5")
        with h5py.File(p, "r") as f:
            raw = np.asarray(f["image"], np.float32)
        img = raw - gaussian_filter(raw, args.ridge_sigma)
        m = raw > otsu(raw)
        ys, xs = np.nonzero(m)
        c0 = (float(ys.mean()), float(xs.mean()))

        nw = len(paths[ai])
        Ps, gd = [], []
        for wi, w in enumerate(paths[ai]):
            W = np.asarray(w, float)[:: args.sub]
            ds = np.hypot(np.diff(W[:, 0]), np.diff(W[:, 1]))
            k = np.concatenate([[True], ds > 1e-6])
            if wi < args.edge_windings or wi >= nw - args.edge_windings:
                k[:] = False
            Ps.append(W); gd.append(k)
        P = np.concatenate(Ps)[np.concatenate(gd)]
        c_old = (cy_a[ai], cx_a[ai])
        print(f"\n=== zip z{zz} <-> old z{z_anchor[ai]:.0f} (anchor {ai}), "
              f"{len(P)} usable pts, scale {args.scale} ===", flush=True)
        print(f"{'candidate':>15} {'objective':>10} {'off med':>10} "
              f"{'off IQR':>9} {'on-film':>8}  {'centre (cy,cx)':>20}")

        rows = []
        for name, D in DIHEDRAL.items():
            best, cz = -1e18, list(c0)
            for coarse, stepsz in ((25, 2.0), (4, 0.5)):
                base = list(cz)
                for dy in np.arange(-coarse, coarse + 1e-9, stepsz):
                    for dx in np.arange(-coarse, coarse + 1e-9, stepsz):
                        c = [base[0] + dy, base[1] + dx]
                        v = float(sample(img, apply_D(P, c_old, D, args.scale, c)).mean())
                        if v > best:
                            best, cz = v, c
            off = offsets(m, apply_D(P, c_old, D, args.scale, cz), cz)
            med, iqr, frac = robust(off)
            rows.append(dict(zip_z=zz, name=name, obj=best, off_med=med,
                             off_iqr=iqr, on_film=frac, c_zip=cz))
            print(f"{name:>15} {best:10.2f} {med:+10.2f} {iqr:9.2f} "
                  f"{100 * frac:7.0f}%  ({cz[0]:8.2f},{cz[1]:8.2f})", flush=True)

        rows.sort(key=lambda r: (r["off_iqr"] if np.isfinite(r["off_iqr"]) else 1e9))
        w = rows[0]
        print(f"  -> tightest: {w['name']} (IQR {w['off_iqr']:.2f}px, med "
              f"{w['off_med']:+.2f}, on-film {100 * w['on_film']:.0f}%); "
              f"runner-up {rows[1]['name']} (IQR {rows[1]['off_iqr']:.2f})")
        all_rows += rows

        # overlay for the winner + the identity, for eyeballing
        fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))
        lo, hi = np.percentile(raw, [1, 99.5])
        for a_, nm in zip(ax, [w["name"], "identity"]):
            r = next(x for x in rows if x["name"] == nm)
            Q = apply_D(P, c_old, DIHEDRAL[nm], args.scale, r["c_zip"])
            a_.imshow(np.clip((raw - lo) / (hi - lo), 0, 1), cmap="gray")
            a_.plot(Q[:, 0], Q[:, 1], ".", ms=0.3, color="red")
            a_.set_title(f"z{zz}  {nm}: |off| med {r['off_med']:.2f}px")
            a_.set_xticks([]); a_.set_yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"dihedral_z{zz}.png"), dpi=120)
        plt.close()

    # cross-slice verdict: the winner must be the SAME candidate at every z
    print("\n=== VERDICT (the true convention must win at EVERY z) ===")
    names = list(DIHEDRAL)
    for nm in names:
        v = [r["off_iqr"] for r in all_rows if r["name"] == nm]
        md = [r["off_med"] for r in all_rows if r["name"] == nm]
        print(f"  {nm:>15}: IQR/slice {np.round(v, 2).tolist()} mean "
              f"{np.mean(v):.2f} | med/slice {np.round(md, 2).tolist()}")
    with open(os.path.join(args.out_dir, "dihedral.json"), "w") as f:
        json.dump(all_rows, f, indent=1)
    print(f"Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
