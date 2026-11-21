"""Locate the left-band break (strip z~743-753, innermost windings) in the walk geometry.

The rendered strip shows a sharp row discontinuity at z~743-753 in cells ~22-28
(the innermost windings = the end of the film), with ~0 px along-arc lag across it:
the content CHANGES rather than shifts. Two readings: (a) the path below the break
traces a NEIGHBOURING winding (radial jump ~ one 26 px spacing -> content from one
turn away), or (b) the path stays on the winding but its along-film position
(phi <-> column map / tangential slide) jumps.

This reads the DELIVERABLE geometry (matched_walks.npz written by v13
--inspect-anchors: kept windings after tracking + z-heal + arc-columns) and, per
inner winding and per anchor in a z window, measures against a clean reference
band of anchors:
  dR = radius - reference radius        (a) shows as |dR| ~ spacing
  dT = tangential offset in px of film  (b) shows as a step in dT
Also reports, for the same anchors, the RAW pre-tracking walks from the merged
cache (median radius, azimuth span), so a missing / partial / doubled walk at the
break is visible. Then draws targeted overlays of the walk on the CT.

Run on Merlin7 (needs the CT slices):
  python -u Scripts/diag_winding_jump.py --matched <dir>/matched_walks.npz \
      --raw-cache <dir>/walk_anchors.npz --ct-dir 01_Mickey_sprockets --out-dir <dir>
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
from unwrapping.inr.unroll_continuous import load_ct, norm_slice, discover_ct_slices

TWO_PI = 2.0 * math.pi


def wrap(a):
    return (a + math.pi) % TWO_PI - math.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matched", required=True)
    ap.add_argument("--raw-cache", default=None)
    ap.add_argument("--ct-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--z-lo", type=float, default=680)
    ap.add_argument("--z-hi", type=float, default=840)
    ap.add_argument("--ref-lo", type=float, default=790)
    ap.add_argument("--ref-hi", type=float, default=830)
    ap.add_argument("--inner", type=int, default=12, help="innermost windings to analyse")
    ap.add_argument("--spacing", type=float, default=26.0, help="winding pitch, new scan")
    ap.add_argument("--pitch", type=float, default=1130.0, help="strip frame pitch")
    ap.add_argument("--overlay-z", default="700,730,740,745,748,752,755,760,770,790")
    ap.add_argument("--n-az", type=int, default=3, help="worst azimuths to zoom")
    ap.add_argument("--az-deg", default=None,
                    help="Absolute azimuths (deg) to zoom instead of the worst ones "
                         "-- pass the pre-fix run's values for a like-for-like view.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    d = np.load(args.matched, allow_pickle=True)
    z = np.asarray(d["z_anchor"], float)
    cx = np.asarray(d["cx_a"], float); cy = np.asarray(d["cy_a"], float)
    seam = float(d["seam"]); P = d["paths"]
    nw = len(P[0])
    K = [len(P[0][k]) for k in range(nw)]
    starts = np.cumsum([0] + K)
    print(f"{len(z)} anchors z{z.min():.0f}-{z.max():.0f}, {nw} kept windings, "
          f"seam {math.degrees(seam):.1f} deg")

    ref = np.nonzero((z >= args.ref_lo) & (z <= args.ref_hi))[0]
    sel = np.nonzero((z >= args.z_lo) & (z <= args.z_hi))[0]
    zs = z[sel]
    print(f"reference anchors: {len(ref)} (z{args.ref_lo:.0f}-{args.ref_hi:.0f}); "
          f"analysed: {len(sel)} (z{args.z_lo:.0f}-{args.z_hi:.0f})")

    def polar(a, k):
        p = np.asarray(P[a][k], float)
        return (np.hypot(p[:, 0] - cx[a], p[:, 1] - cy[a]),
                np.arctan2(p[:, 1] - cy[a], p[:, 0] - cx[a]))

    # reference radius + azimuth profile of EVERY winding (neighbour identification
    # must go through azimuth: windings have different column counts)
    Rref, Tref_all = [], []
    for k in range(nw):
        Rref.append(np.median(np.stack([polar(a, k)[0] for a in ref]), 0))
        Tref_all.append(np.angle(np.mean(np.exp(1j * np.stack(
            [polar(a, k)[1] for a in ref])), 0)))

    def r_at(kk, theta):
        """Neighbour kk's reference radius at absolute azimuths theta."""
        ph = (Tref_all[kk] - seam) % TWO_PI
        o = np.argsort(ph)
        return np.interp((theta - seam) % TWO_PI, ph[o], Rref[kk][o], period=TWO_PI)
    print("\nwinding table (strip columns before seam-bridge columns; cells = col/pitch):")
    print("   k     K   col0-col1        cells       r_ref med")
    for k in range(min(nw, args.inner + 2)):
        print(f"  {k:2d} {K[k]:5d}  {starts[k]:6d}-{starts[k+1]:6d}  "
              f"{starts[k]/args.pitch:5.1f}-{starts[k+1]/args.pitch:5.1f}  "
              f"{np.median(Rref[k]):8.1f}")

    nin = min(args.inner, nw)
    half = args.spacing / 2.0
    DR, DT, AZ = [], [], []
    print("\nper (winding, anchor) vs reference: med|dR|, frac|dR|>half-spacing, "
          "frac matching a NEIGHBOUR's ref radius, median tangential dT (px)")
    for k in range(nin):
        R = np.empty((len(sel), K[k])); T = np.empty_like(R)
        for i, a in enumerate(sel):
            R[i], T[i] = polar(a, k)
        Tref = Tref_all[k]
        dR = R - Rref[k]
        dT = wrap(T - Tref) * Rref[k]
        DR.append(dR); DT.append(dT); AZ.append(Tref)
        lines = []
        for i, a in enumerate(sel):
            big = np.abs(dR[i]) > half
            nb = 0
            for kk in (k - 1, k + 1):
                if 0 <= kk < nw:
                    nb += int(np.sum(big & (np.abs(R[i] - r_at(kk, T[i])) < half)))
            fb = big.mean(); mdt = float(np.median(dT[i]))
            flag = fb > 0.02 or abs(mdt) > 5
            if flag or int(zs[i]) % 10 == 0:
                lines.append(f"    z{zs[i]:6.0f}  med|dR| {np.median(np.abs(dR[i])):5.1f}  "
                             f"big {100*fb:5.1f}%  ->neighbour {100*nb/K[k]:5.1f}%  "
                             f"dT med {mdt:+7.1f}  p5/p95 {np.percentile(dT[i],5):+7.1f}/"
                             f"{np.percentile(dT[i],95):+7.1f}" + ("  <==" if flag else ""))
        print(f"  winding {k} (cells {starts[k]/args.pitch:.1f}-{starts[k+1]/args.pitch:.1f}):")
        print("\n".join(lines))

    # ── heat maps: dR and dT per inner winding, z vs azimuth-ordered column ──
    fig, axes = plt.subplots(nin, 2, figsize=(16, 2.0 * nin), squeeze=False)
    for k in range(nin):
        ext = [starts[k], starts[k + 1], zs[-1], zs[0]]
        im = axes[k, 0].imshow(DR[k], aspect="auto", cmap="RdBu_r", vmin=-30, vmax=30,
                               extent=ext)
        axes[k, 0].set_ylabel(f"w{k}\nz", fontsize=8)
        im2 = axes[k, 1].imshow(DT[k], aspect="auto", cmap="RdBu_r", vmin=-60, vmax=60,
                                extent=ext)
        for ax in axes[k]:
            ax.axhline(770, color="k", lw=0.6, ls="--")
            ax.tick_params(labelsize=7)
    axes[0, 0].set_title("dR radius - ref (px, +-30)  [dashed = b0|b1 cache join z770]")
    axes[0, 1].set_title("dT tangential offset along film (px, +-60)")
    fig.colorbar(im, ax=axes[:, 0], shrink=0.3); fig.colorbar(im2, ax=axes[:, 1], shrink=0.3)
    fig.savefig(os.path.join(args.out_dir, "dR_dT_inner.png"), dpi=110)
    plt.close(fig)

    # ── worst azimuths in the break zone z735-765 ──
    nb_az = 72
    score = np.zeros(nb_az)
    zone = (zs >= 735) & (zs <= 765)
    for k in range(nin):
        b = ((AZ[k] % TWO_PI) / TWO_PI * nb_az).astype(int) % nb_az
        s = (np.abs(DR[k][zone]) > half).sum(0) + (np.abs(DT[k][zone]) > 10).sum(0)
        np.add.at(score, b, s)
    worst = []
    for j in np.argsort(score)[::-1]:
        if all(min(abs(j - w), nb_az - abs(j - w)) > 4 for w in worst):
            worst.append(j)
        if len(worst) == args.n_az:
            break
    worst_deg = [(j + 0.5) * 360.0 / nb_az for j in worst]
    if args.az_deg:
        worst_deg = [float(v) for v in args.az_deg.split(",")]
        worst = [int(v / 360.0 * nb_az) % nb_az for v in worst_deg]
    print("\nbreak-zone (z735-765) score per 5deg azimuth bin, top:",
          ", ".join(f"{a:.0f}deg ({score[j]:.0f})" for a, j in zip(worst_deg, worst)))
    with open(os.path.join(args.out_dir, "hires_args.sh"), "w") as f:
        oz = [float(v) for v in args.overlay_z.split(",")]
        idx = sorted({int(np.argmin(np.abs(z - v))) for v in oz})
        rel = [((a - math.degrees(seam)) % 360.0) for a in worst_deg]
        f.write(f'ANCHORS="{",".join(map(str, idx))}"\n')
        f.write(f'WEDGES="{",".join(f"{r:.1f}" for r in rel)}"\n')
    print(f"wrote hires_args.sh (anchors {idx}, wedges rel. seam "
          f"{[round(r, 1) for r in rel]})")

    # ── raw pre-tracking walks at the overlay anchors ──
    if args.raw_cache and os.path.exists(args.raw_cache):
        r = np.load(args.raw_cache, allow_pickle=True)
        rz = np.asarray(r["z_anchor"], float); RP = r["paths"]
        rmax = float(np.median(Rref[nin - 1])) + args.spacing
        print(f"\nRAW walks (pre-tracking) with median r < {rmax:.0f}, at overlay z:")
        for v in [float(x) for x in args.overlay_z.split(",")]:
            a = int(np.argmin(np.abs(rz - v)))
            am = int(np.argmin(np.abs(z - rz[a])))     # matched anchor = refit centre
            rows = []
            for p in RP[a]:
                p = np.asarray(p, float)
                rr = np.hypot(p[:, 0] - cx[am], p[:, 1] - cy[am])
                if np.median(rr) > rmax:
                    continue
                th = np.unwrap(np.arctan2(p[:, 1] - cy[am], p[:, 0] - cx[am]))
                rows.append((np.median(rr), math.degrees(th.max() - th.min()), len(p)))
            rows.sort()
            print(f"  z{rz[a]:.0f}: " + "; ".join(f"r{m:.0f} span{s:.0f}deg" for m, s, _ in rows))

    # ── targeted overlays: inner windings at the worst azimuths, across z ──
    z_to_path = discover_ct_slices(args.ct_dir)
    oz = [float(v) for v in args.overlay_z.split(",")]
    cmap = plt.get_cmap("tab20")
    r_lo = float(np.median(Rref[0])) - 1.5 * args.spacing
    r_hi = float(np.median(Rref[nin - 1])) + 1.5 * args.spacing
    for wi, azd in enumerate(worst_deg):
        th = math.radians(azd)
        fig, axes = plt.subplots(1, len(oz), figsize=(3.2 * len(oz), 6.5), squeeze=False)
        for j, v in enumerate(oz):
            a = int(np.argmin(np.abs(z - v)))
            za = min(z_to_path, key=lambda q: abs(q - z[a]))
            img = norm_slice(load_ct(z_to_path[za]))
            rm = 0.5 * (r_lo + r_hi); hw = 0.5 * (r_hi - r_lo) + 10
            mx, my = cx[a] + rm * math.cos(th), cy[a] + rm * math.sin(th)
            x0, x1 = int(mx - hw), int(mx + hw); y0, y1 = int(my - hw), int(my + hw)
            ax = axes[0, j]
            ax.imshow(img[max(0, y0):y1, max(0, x0):x1], cmap="gray",
                      extent=[max(0, x0), x1, min(y1, img.shape[0]), max(0, y0)])
            for k in range(nin):
                p = np.asarray(P[a][k], float)
                m = (p[:, 0] >= x0) & (p[:, 0] < x1) & (p[:, 1] >= y0) & (p[:, 1] < y1)
                if not m.any():
                    continue
                q = p.copy(); q[~m] = np.nan
                ax.plot(q[:, 0], q[:, 1], "-", color=cmap(k % 20), lw=1.0)
                i0 = np.nonzero(m)[0][len(np.nonzero(m)[0]) // 2]
                ax.text(p[i0, 0], p[i0, 1], str(k), color=cmap(k % 20), fontsize=7,
                        fontweight="bold")
            ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
            ax.set_title(f"z{za}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"inner {nin} windings at az {azd:.0f} deg (abs) — labels = kept "
                     f"winding index; a label hopping bands across z = winding jump",
                     fontsize=10)
        plt.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"overlay_az{int(azd):03d}.png"), dpi=120)
        plt.close(fig)
        print(f"drew overlay_az{int(azd):03d}.png", flush=True)


if __name__ == "__main__":
    main()
