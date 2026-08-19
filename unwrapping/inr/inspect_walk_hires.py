"""High-res visual fit check for the emulsion-walk whole-roll, from the CACHE.

Prefers the HEALED winding geometry (healed_windings.npz: per-winding r(phi) about
the center, after multi-ray seeding + z-jump heal) so the check shows the DELIVERABLE
curves. Falls back to raw walk_anchors.npz if healed is absent.

For each requested anchor it draws:
  * a full-resolution overlay of every winding on the CT (high dpi), and
  * a ROW of ZOOM crops at SEVERAL azimuths (default 4 around the roll) where the
    individual emulsion bands resolve, so missed windings / band-to-band SKIPS are
    visible anywhere on the turn, not just opposite the seam.

No seg load -> fast. Run on Merlin7 (needs the CT h5 + the cache).

Usage:
  python -m unwrapping.inr.inspect_walk_hires \
      --cache-dir unwrapping/inr/results/walk_wholeroll_v4 \
      --ct-dir 01_Mickey_hdf --anchors 0,14,24 --wedge-az-deg 45,135,225,315
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
from unwrapping.inr.unroll_continuous import load_ct, norm_slice, discover_ct_slices

TWO_PI = 2.0 * math.pi


def load_geometry(cache_dir):
    """Return (kind, dict). Prefer the MATCHED walks (raw x,y, deliverable geometry);
    fall back to the pre-match walk cache."""
    mp = os.path.join(cache_dir, "matched_walks.npz")
    if os.path.exists(mp):
        d = np.load(mp, allow_pickle=True)
        return "raw", d
    d = np.load(os.path.join(cache_dir, "walk_anchors.npz"), allow_pickle=True)
    return "raw", d


def anchor_curves(kind, d, a):
    """List of (N,2) xy curves for anchor a (from healed r(phi) or raw paths)."""
    seam = float(d["seam"]); cx, cy = float(d["cx_a"][a]), float(d["cy_a"][a])
    if kind == "healed":
        curves = []
        for phi, R in zip(d["phi"], d["R"]):
            th = seam + phi; r = R[a]
            curves.append(np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)]))
        return curves, cx, cy, seam
    return [np.asarray(p) for p in d["paths"][a]], cx, cy, seam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--ct-dir", required=True)
    ap.add_argument("--anchors", default="0,14,24",
                    help="Comma indices, or 'all' for every anchor in the geometry.")
    ap.add_argument("--out-dir", default=None,
                    help="Where to write images (default = cache-dir).")
    ap.add_argument("--full-only", action="store_true",
                    help="Only the full-ring overlay per anchor (skip zoom wedges).")
    ap.add_argument("--wedge-az-deg", default="45,135,225,315",
                    help="Azimuths (deg, relative to seam) to zoom into.")
    ap.add_argument("--wedge-deg", type=float, default=22.0,
                    help="Azimuthal half-width of each zoom wedge.")
    args = ap.parse_args()

    kind, d = load_geometry(args.cache_dir)
    z_anchor = d["z_anchor"]
    out_dir = args.out_dir or args.cache_dir
    os.makedirs(out_dir, exist_ok=True)
    z_to_path = discover_ct_slices(args.ct_dir)
    cmap = plt.get_cmap("hsv")
    idxs = (list(range(len(z_anchor))) if args.anchors.strip() == "all"
            else [int(x) for x in args.anchors.split(",")])
    wedge_az = [math.radians(float(x)) for x in args.wedge_az_deg.split(",")]
    wob = math.radians(args.wedge_deg)
    print(f"geometry = {kind}")

    for a in idxs:
        curves, cx, cy, seam = anchor_curves(kind, d, a)
        nw = len(curves)
        za = min(z_to_path, key=lambda zz: abs(zz - z_anchor[a]))
        img = norm_slice(load_ct(z_to_path[za]))

        # ── full overlay ──
        fig, ax = plt.subplots(1, 1, figsize=(13, 13))
        ax.imshow(img, cmap="gray")
        for k, p in enumerate(curves):
            ax.plot(p[:, 0], p[:, 1], "-", color=cmap(k / max(1, nw)), lw=0.8)
        ax.plot(cx, cy, "c+", ms=12)
        ax.set_title(f"anchor {a} z{za} — {nw} windings ({kind})", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"anchor_{a:03d}_z{za}_full.png"), dpi=150)
        plt.close()
        if args.full_only:
            print(f"anchor {a} z{za}: {nw} windings -> full", flush=True)
            continue

        # ── one LARGE zoom file per azimuth (pixel-resolvable) ──
        for wi, waz in enumerate(wedge_az):
            az = seam + waz
            allp = np.vstack(curves)
            rr = np.hypot(allp[:, 0] - cx, allp[:, 1] - cy)
            rs = np.linspace(rr.min(), rr.max(), 60)
            aa = np.linspace(az - wob, az + wob, 60)
            gx = cx + np.outer(rs, np.cos(aa)); gy = cy + np.outer(rs, np.sin(aa))
            x0, x1 = int(max(0, gx.min())), int(min(img.shape[1], gx.max()))
            y0, y1 = int(max(0, gy.min())), int(min(img.shape[0], gy.max()))
            fig, ax = plt.subplots(1, 1, figsize=(15, 15 * (y1 - y0) / max(1, x1 - x0)))
            ax.imshow(img[y0:y1, x0:x1], cmap="gray")
            for k, p in enumerate(curves):
                th = np.arctan2(p[:, 1] - cy, p[:, 0] - cx)
                m = np.abs((th - az + math.pi) % TWO_PI - math.pi) < wob
                # BREAK the polyline wherever the selected points are not
                # consecutive along the path. At a wedge ON the seam a ~360deg
                # walk enters the wedge TWICE (its start and its end), and
                # joining those two runs draws a chord straight across the roll
                # -- which reads as a catastrophic winding crossing that is not
                # there. NaN separators keep each run its own segment.
                sel = np.nonzero(m)[0]
                if sel.size == 0:
                    continue
                cut = np.nonzero(np.diff(sel) > 1)[0] + 1
                xx = p[sel, 0] - x0
                yy = p[sel, 1] - y0
                if cut.size:
                    xx = np.insert(xx.astype(float), cut, np.nan)
                    yy = np.insert(yy.astype(float), cut, np.nan)
                ax.plot(xx, yy, "-", color=cmap(k / max(1, nw)), lw=1.6)
            ax.set_title(f"anchor {a} z{za} ({kind}) az {math.degrees(waz):.0f}° rel "
                         f"seam — one line/winding; crossing bands = SKIP, bare band "
                         f"= MISS", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir,
                        f"anchor_{a:03d}_z{za}_zoom{wi}_az{int(math.degrees(waz))}.png"),
                        dpi=140)
            plt.close()
        print(f"anchor {a} z{za}: {nw} windings -> full + {len(wedge_az)} zoom files",
              flush=True)


if __name__ == "__main__":
    main()
