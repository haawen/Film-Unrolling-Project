"""Side-by-side hyperparameter comparison for the global-spline whole-roll fit.

One figure per anchor: a tall angular WEDGE spanning all windings, shown as
  [ segmentation | CT+curves(config 1) | CT+curves(config 2) | ... ]
so the fitted curve for each hyperparameter setting can be eyeballed against the
emulsion bands AND the raw segmentation (to see whether seg noise / dashed
emulsion is what drives skips/overlaps/wriggle). Each CT panel is titled with
that config's ring-check skips/overlaps for the anchor.

Reads the cached emulsion cloud (wholeroll_cache.npz) + film-band cache for the
fit; loads the chosen anchor's seg mid-slice + raw CT only for display.

Usage:
  python -m unwrapping.inr.compare_fits \
      --cache .../wholeroll_v2/wholeroll_cache.npz \
      --filmband-cache .../anchor_filmband_r2880.npz \
      --seg-dir 01_Mickey_3d --ct-dir 01_Mickey_hdf \
      --out-dir <out> --anchor-z 1605 --azimuth-deg -90 --wedge-deg 12
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.inr.unwrap_data_3d import discover_volumes
from unwrapping.inr.unroll_continuous import load_ct, norm_slice, discover_ct_slices, load_seg_mid
from unwrapping.inr.unroll_wholeroll_simple import filmband_seeds
from unwrapping.inr.unroll_spline_global import (
    fit_global_spiral, eval_fourier, ring_check, load_anchor_cloud, load_filmband,
    refine_per_winding, eval_winding, raw_max_seeds,
)

TWO_PI = 2.0 * math.pi

# All use the SAME robust global skeleton (spl8 + ecc8) for winding assignment;
# they differ only in how the FINAL per-winding curve is produced. "global" = the
# shared smooth base+ecc (over-penalized, can't follow per-winding shape);
# "refine N" = each winding fit independently with a degree-N robust poly on the
# skeleton's assignment (hugs each band's own shape, still no skips/overlaps).
# (name, kind, degree)   kind in {global, refine}
# (name, kind, degree, fit_type, smooth_px); kind in {global, refine}
CONFIGS = [
    ("poly deg6",        "refine", 6, "poly",   0.0),
    ("spline sm4",       "refine", 6, "spline", 4.0),
    ("spline sm2",       "refine", 6, "spline", 2.0),
    ("spline sm1 (tight)", "refine", 6, "spline", 1.0),
]
GATE = 11.0                       # assignment gate (px); band ~17px thick


def seg_rgb(seg):
    """bg=black, film(1)=gray, emulsion(2)=red."""
    rgb = np.zeros(seg.shape + (3,), np.float32)
    rgb[seg == 1] = (0.55, 0.55, 0.55)
    rgb[seg == 2] = (1.0, 0.15, 0.15)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--filmband-cache", required=True)
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--ct-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--anchor-z", type=float, action="append", default=[1605.0],
                    help="Anchor z to compare (repeatable). Ignored with --montage.")
    ap.add_argument("--azimuth-deg", type=float, default=-90.0)
    ap.add_argument("--wedge-deg", type=float, default=12.0)
    ap.add_argument("--r-lo", type=float, default=0.0,
                    help="Crop wedge to this min radius (px); 0 = all windings.")
    ap.add_argument("--r-hi", type=float, default=1e9,
                    help="Crop wedge to this max radius (px).")
    ap.add_argument("--grid", action="store_true",
                    help="Sweep smoothing x clip into ONE grid figure (per anchor).")
    ap.add_argument("--smooth-list", default="1,2,3", help="grid rows: spline smooth px")
    ap.add_argument("--clip-list", default="6,8,11", help="grid cols: skeleton clip px")
    ap.add_argument("--count-pct", type=float, default=99.5,
                    help="Percentile of raw per-ray band counts -> winding count.")
    ap.add_argument("--montage", action="store_true",
                    help="Fit EVERY anchor at one wedge (curves on seg), assemble a "
                         "montage + print a per-anchor count/skip/overlap audit.")
    ap.add_argument("--montage-smooth", type=float, default=2.0)
    ap.add_argument("--montage-clip", type=float, default=8.0)
    ap.add_argument("--fullslice", action="store_true",
                    help="Draw ALL per-winding curves over the FULL turn on the "
                         "whole CT slice (+ seg) for each --anchor-z. Final check.")
    ap.add_argument("--seam-exclude-deg", type=float, default=8.0)
    ap.add_argument("--seed-ray-halfdeg", type=float, default=6.0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    eps = math.radians(args.seam_exclude_deg)

    z_anchor, cx_a, cy_a, PHI, RR, seam = load_anchor_cloud(args.cache)
    FB, n_rays = load_filmband(args.filmband_cache)
    rayi = int(round(np.mod(seam + math.pi, TWO_PI) / TWO_PI * n_rays)) % n_rays
    pairs = discover_volumes(args.seg_dir)               # for seg mid-slice load
    z_to_ct = discover_ct_slices(args.ct_dir)

    # FIXED winding count from the MOST-COMPLETE anchor (one curve per emulsion
    # band everywhere). Per-anchor reseeding undercounts (27-35) -> too few curves
    # -> fits average onto the acetate between bands. Pick the anchor whose best
    # ray yields the most film bands; use its seeds for every anchor (ICP adapts
    # the radii per anchor, the COUNT stays fixed).
    counts = np.array([int(np.sum(~np.isnan(FB[a, r])))
                       for a in range(FB.shape[0]) for r in range(n_rays)])
    seeds_global, nw_true = raw_max_seeds(FB, n_rays, pct=args.count_pct)
    print(f"raw per-ray band counts: max={counts.max()} p99.5={np.percentile(counts,99.5):.0f} "
          f"p99={np.percentile(counts,99):.0f} p95={np.percentile(counts,95):.0f} "
          f"median={np.median(counts):.0f}  -> nw={nw_true}")

    # ── MONTAGE: fit every anchor at one wedge, audit + assemble ──
    if args.montage:
        from PIL import Image as _Im
        wmid = math.radians(args.azimuth_deg); wband = math.radians(args.wedge_deg)
        tiles = []
        print("  per-anchor audit (nw, curves fit, ring-check):")
        for i in range(len(z_anchor)):
            zi = int(z_anchor[i])
            phi_pix = np.asarray(PHI[i], float); r_pix = np.asarray(RR[i], float)
            cx, cy = cx_a[i], cy_a[i]
            vp, pp, z0, z1 = pairs[i]
            seg_mid, _ = load_seg_mid(pp)
            base, ecc, phi_asg, keep, kstar, info = fit_global_spiral(
                phi_pix, r_pix, seam, seeds_global, 8.0, 8, gate_px=args.montage_clip,
                base_type="spline", base_degree=4)
            nw = info["nw"]; phimax = float(phi_asg[keep].max())
            coefs = refine_per_winding(phi_pix, r_pix, kstar, keep, nw, eps,
                                       base, ecc, seam, fit_type="spline",
                                       smooth_px=args.montage_smooth, clip_px=args.montage_clip)
            nfit = sum(c is not None for c in coefs)
            rc = ring_check(base, ecc, seam, FB[i], n_rays, phimax, eps,
                            os.path.join(args.out_dir, f"_tmp_rc_{i}.png"))
            theta = seam + phi_pix
            sel = (np.abs(np.mod(theta - wmid + math.pi, TWO_PI) - math.pi) < wband) \
                & (r_pix >= args.r_lo) & (r_pix <= args.r_hi)
            exs = cx + r_pix[sel] * np.cos(theta[sel]); eys = cy + r_pix[sel] * np.sin(theta[sel])
            x0, x1 = max(0, int(exs.min()) - 12), int(exs.max()) + 12
            y0, y1 = max(0, int(eys.min()) - 12), int(eys.max()) + 12
            phic = np.mod(wmid - seam, TWO_PI)
            wphi = phic + np.linspace(-wband, wband, 200)
            fig, ax = plt.subplots(1, 1, figsize=(3.0, 3.0))
            ax.imshow(seg_rgb(seg_mid)[y0:y1, x0:x1])
            for kk in range(nw):
                if coefs[kk] is None:
                    continue
                rr = eval_winding(coefs[kk], wphi); tt = seam + wphi
                ax.plot(cx + rr * np.cos(tt) - x0, cy + rr * np.sin(tt) - y0,
                        "-", color="yellow", lw=0.8)
            ax.set_xlim(0, x1 - x0); ax.set_ylim(y1 - y0, 0)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"z{zi} {nfit}/{nw}w sk{rc['skips']} ov{rc['overlaps']}",
                         fontsize=8)
            tp = os.path.join(args.out_dir, f"_m_{i:02d}.png")
            plt.tight_layout(); plt.savefig(tp, dpi=90); plt.close(); tiles.append(tp)
            os.path.exists(os.path.join(args.out_dir, f"_tmp_rc_{i}.png")) and \
                os.remove(os.path.join(args.out_dir, f"_tmp_rc_{i}.png"))
            print(f"    z{zi:>4} a{i:02d}: {nfit}/{nw}w  skips {rc['skips']}  "
                  f"overlaps {rc['overlaps']}", flush=True)
        ims = [_Im.open(t).convert("RGB") for t in tiles]
        w, h = ims[0].size; cols = 5; rows = int(np.ceil(len(ims) / cols))
        mon = _Im.new("RGB", (w * cols, h * rows), "white")
        for j, im in enumerate(ims):
            mon.paste(im, ((j % cols) * w, (j // cols) * h))
        mon.save(os.path.join(args.out_dir, f"montage_az{int(args.azimuth_deg)}.png"))
        for t in tiles:
            os.remove(t)
        print(f"  wrote montage ({len(tiles)} anchors)")
        return

    # ── FULL-SLICE: all per-winding curves over the whole turn on the CT slice ──
    if args.fullslice:
        phig = np.linspace(eps, TWO_PI - eps, 4000)
        for zq in args.anchor_z:
            i = int(np.argmin(np.abs(z_anchor - zq))); zi = int(z_anchor[i])
            phi_pix = np.asarray(PHI[i], float); r_pix = np.asarray(RR[i], float)
            cx, cy = cx_a[i], cy_a[i]
            vp, pp, z0, z1 = pairs[i]
            seg_mid, _ = load_seg_mid(pp)
            za = min(z_to_ct, key=lambda zz: abs(zz - zi))
            ct = norm_slice(load_ct(z_to_ct[za]))
            base, ecc, phi_asg, keep, kstar, info = fit_global_spiral(
                phi_pix, r_pix, seam, seeds_global, 8.0, 8, gate_px=args.montage_clip,
                base_type="spline", base_degree=4)
            nw = info["nw"]
            coefs = refine_per_winding(phi_pix, r_pix, kstar, keep, nw, eps,
                                       base, ecc, seam, fit_type="spline",
                                       smooth_px=args.montage_smooth, clip_px=args.montage_clip)
            nfit = sum(c is not None for c in coefs)
            tt = seam + phig
            for bg, img, tag in [("ct", ct, "CT"),
                                 ("seg", seg_rgb(seg_mid), "SEG(emul=red)")]:
                fig, ax = plt.subplots(1, 1, figsize=(12, 12))
                if bg == "ct":
                    ax.imshow(img, cmap="gray")
                else:
                    ax.imshow(img)
                for kk in range(nw):
                    if coefs[kk] is None:
                        continue
                    rr = eval_winding(coefs[kk], phig)
                    ax.plot(cx + rr * np.cos(tt), cy + rr * np.sin(tt), "-",
                            color="yellow", lw=0.6)
                ax.plot(cx, cy, "c+", ms=10)
                ax.set_title(f"z{zi}: {nfit}/{nw} per-winding curves on {tag}", fontsize=11)
                ax.set_xticks([]); ax.set_yticks([])
                plt.tight_layout()
                plt.savefig(os.path.join(args.out_dir, f"fullslice_z{zi}_{bg}.png"), dpi=150)
                plt.close()
            print(f"  z{zi}: {nfit}/{nw} curves -> fullslice_z{zi}_(ct|seg).png", flush=True)
        print("done (fullslice)")
        return

    for zq in args.anchor_z:
        i = int(np.argmin(np.abs(z_anchor - zq)))
        zi = int(z_anchor[i])
        phi_pix = np.asarray(PHI[i], float); r_pix = np.asarray(RR[i], float)
        cx, cy = cx_a[i], cy_a[i]
        seeds = seeds_global               # FIXED count for every anchor

        # seg + CT for display (nearest available)
        vp, pp, z0, z1 = pairs[i]
        seg_mid, _ = load_seg_mid(pp)
        za = min(z_to_ct, key=lambda zz: abs(zz - zi))
        ct = norm_slice(load_ct(z_to_ct[za]))

        # wedge crop bbox (full radial extent at this azimuth)
        wmid = math.radians(args.azimuth_deg); wband = math.radians(args.wedge_deg)
        theta = seam + phi_pix
        sel = (np.abs(np.mod(theta - wmid + math.pi, TWO_PI) - math.pi) < wband) \
            & (r_pix >= args.r_lo) & (r_pix <= args.r_hi)
        exs = cx + r_pix[sel] * np.cos(theta[sel])
        eys = cy + r_pix[sel] * np.sin(theta[sel])
        x0, x1 = max(0, int(exs.min()) - 15), int(exs.max()) + 15
        y0, y1 = max(0, int(eys.min()) - 15), int(eys.max()) + 15

        phic = np.mod(wmid - seam, TWO_PI)
        # ONE robust global skeleton fit, reused by every panel for assignment.
        base, ecc, phi_asg, keep, kstar, info = fit_global_spiral(
            phi_pix, r_pix, seam, seeds, 8.0, 8, gate_px=GATE,
            base_type="spline", base_degree=4)
        nw = info["nw"]; phimax = float(phi_asg[keep].max())
        wphi = phic + np.linspace(-wband, wband, 200)        # wedge sample (wrapped)
        cw, ch = (x1 - x0), (y1 - y0)
        fw = max(7.0, cw / 110.0); fh = max(3.0, ch / 110.0)

        segcrop = seg_rgb(seg_mid)[y0:y1, x0:x1]

        def draw_curves(ax, coefs):
            for kk in range(nw):
                if coefs[kk] is None:
                    continue
                rr = eval_winding(coefs[kk], wphi)
                tt = seam + wphi
                ax.plot(cx + rr * np.cos(tt) - x0, cy + rr * np.sin(tt) - y0,
                        "-", color="yellow", lw=1.0)

        # ── GRID: smoothing (rows) x clip (cols), curves on SEG ──
        if args.grid:
            smooths = [float(s) for s in args.smooth_list.split(",")]
            clips = [float(c) for c in args.clip_list.split(",")]
            nr, nc = len(smooths), len(clips)
            fig, axes = plt.subplots(nr, nc, figsize=(2.6 * nc * cw / ch + 1, 2.6 * nr),
                                     squeeze=False)
            for ri, sm in enumerate(smooths):
                for ci, cl in enumerate(clips):
                    ax = axes[ri][ci]
                    coefs = refine_per_winding(phi_pix, r_pix, kstar, keep, nw, eps,
                                               base, ecc, seam, fit_type="spline",
                                               smooth_px=sm, clip_px=cl)
                    ax.imshow(segcrop)
                    draw_curves(ax, coefs)
                    ax.set_xlim(0, cw); ax.set_ylim(ch, 0)
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.set_title(f"sm{sm:g} clip{cl:g}", fontsize=8)
            fig.suptitle(f"z{zi} per-winding spline sweep on SEG (emul=red) — "
                         f"wedge @ {args.azimuth_deg:.0f}deg", fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, f"grid_z{zi}_az{int(args.azimuth_deg)}.png"),
                        dpi=160)
            plt.close()
            print(f"wrote grid for z{zi} az{args.azimuth_deg:.0f}", flush=True)
            continue

        def panel(name, draw, on_seg=False):
            """One standalone full-size panel: CT (or seg) crop + draw(ax)."""
            fig, ax = plt.subplots(1, 1, figsize=(fw, fh))
            if on_seg:
                ax.imshow(segcrop)
            else:
                ax.imshow(ct[y0:y1, x0:x1], cmap="gray")
                ax.scatter(exs - x0, eys - y0, s=3, c="cyan", alpha=0.30)
            draw(ax)
            ax.set_xlim(0, cw); ax.set_ylim(ch, 0)
            ax.set_xticks([]); ax.set_yticks([])
            bg = "SEG(emul=red)" if on_seg else "CT"
            ax.set_title(f"z{zi} {name} on {bg} — wedge @ {args.azimuth_deg:.0f}deg",
                         fontsize=10)
            plt.tight_layout()
            sfx = "_onseg" if on_seg else ""
            plt.savefig(os.path.join(args.out_dir, f"z{zi}_{name}{sfx}.png"), dpi=150)
            plt.close()

        # segmentation panel (own file)
        fig, ax = plt.subplots(1, 1, figsize=(fw, fh))
        ax.imshow(seg_rgb(seg_mid)[y0:y1, x0:x1])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"z{zi} segmentation (emul=red, film=gray)", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"z{zi}_0seg.png"), dpi=150); plt.close()

        for name, kind, deg, ftype, smpx in CONFIGS:
            if kind == "refine":
                coefs = refine_per_winding(phi_pix, r_pix, kstar, keep, nw, eps,
                                           base, ecc, seam, degree=deg,
                                           fit_type=ftype, smooth_px=smpx, clip_px=GATE)

            def draw(ax, kind=kind, coefs=(coefs if kind == "refine" else None)):
                for kk in range(nw):
                    if kind == "global":
                        rr = base(wphi + TWO_PI * kk) + eval_fourier(seam + wphi, ecc)
                    else:
                        if coefs[kk] is None:
                            continue
                        rr = eval_winding(coefs[kk], wphi)
                    tt = seam + wphi
                    ax.plot(cx + rr * np.cos(tt) - x0, cy + rr * np.sin(tt) - y0,
                            "-", color="yellow", lw=1.2)
            nm = name.replace(" ", "_").replace("+", "")
            panel(nm, draw)
            if kind == "refine":                       # also draw on segmentation
                panel(nm, draw, on_seg=True)
        print(f"wrote panels for z{zi}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
