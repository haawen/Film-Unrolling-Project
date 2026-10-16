"""Diagnostic: overlay the raycast winding centerline on the actual seg mask.

Checks whether the per_angle wriggle on mid-roll chunks is genuine segmentation
noise (jagged emulsion ring) or a detector winding-jump on an otherwise clean
ring. Renders, for each requested chunk, the seg map (0=air/1=base/2=emulsion)
with the chosen winding's raycast centerline drawn on top — full view + a zoom
crop of the arc region.

Usage:
  python -m unwrapping.inr.diag_seg_centerline \
      --seg-dir 01_Mickey_3d --out-dir <out> --winding 17 \
      --angle-extent-deg 140 --n-rays 2880 --chunks 0836 1595
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from unwrapping.center_detection import find_spool_center
from unwrapping.inr.surface_data import detect_winding_boundaries_raycast
from unwrapping.inr.unwrap_data_3d import discover_volumes
from unwrapping.inr.sanity_winding_strip import circ_interp_radius, arclength_angles
from unwrapping.inr.unroll_continuous import load_seg_mid


SEG_CMAP = ListedColormap(["black", "tab:blue", "tab:orange"])  # air/base/emulsion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--winding", type=int, default=17)
    ap.add_argument("--angle-start-deg", type=float, default=0.0)
    ap.add_argument("--angle-extent-deg", type=float, default=140.0)
    ap.add_argument("--n-rays", type=int, default=2880)
    ap.add_argument("--chunks", nargs="+", default=["0836", "1595"],
                    help="Chunk z-start tags to diagnose.")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pairs = discover_volumes(args.seg_dir)
    by_tag = {f"{z0:04d}": (vp, pp, z0, z1) for vp, pp, z0, z1 in pairs}
    k = args.winding
    a0 = math.radians(args.angle_start_deg)
    ext = math.radians(args.angle_extent_deg)

    for tag in args.chunks:
        if tag not in by_tag:
            print(f"  chunk {tag} not found; have {list(by_tag)[:5]}..."); continue
        vp, pp, z0, z1 = by_tag[tag]
        seg, mid = load_seg_mid(pp)
        cy, cx = find_spool_center(seg)
        _, nl, per_angle = detect_winding_boundaries_raycast(
            seg, (cy, cx), n_rays=args.n_rays, return_per_angle=True)
        angles, s_total, n_cols = arclength_angles(per_angle[k], a0, ext, 0)
        r_c = circ_interp_radius(per_angle[k], angles)
        xs = cx + r_c * np.cos(angles); ys = cy + r_c * np.sin(angles)
        print(f"chunk {tag} z{z0 + mid}: nl={nl} winding {k} "
              f"r=[{r_c.min():.0f},{r_c.max():.0f}] (swing {r_c.max() - r_c.min():.0f}px)")

        # full view
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        ax.imshow(seg, cmap=SEG_CMAP, vmin=0, vmax=2, interpolation="nearest")
        ax.plot(xs, ys, "-", color="red", lw=1.5)
        ax.plot(cx, cy, "c+", ms=14)
        ax.set_title(f"chunk {tag} (z{z0 + mid}) seg + winding-{k} centerline")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, f"seg_{tag}_full.png"), dpi=130)
        plt.close()

        # zoom crop around the arc bounding box (padded)
        pad = 60
        x0 = max(0, int(xs.min()) - pad); x1 = min(seg.shape[1], int(xs.max()) + pad)
        y0 = max(0, int(ys.min()) - pad); y1 = min(seg.shape[0], int(ys.max()) + pad)
        fig, ax = plt.subplots(1, 1, figsize=(14, 14 * (y1 - y0) / max(1, x1 - x0)))
        ax.imshow(seg[y0:y1, x0:x1], cmap=SEG_CMAP, vmin=0, vmax=2,
                  interpolation="nearest")
        ax.plot(xs - x0, ys - y0, "-", color="red", lw=1.5)
        ax.set_title(f"chunk {tag} (z{z0 + mid}) ZOOM — winding {k} centerline on seg "
                     f"(blue=base, orange=emulsion); swing {r_c.max() - r_c.min():.0f}px")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout(); plt.savefig(os.path.join(args.out_dir, f"seg_{tag}_zoom.png"), dpi=150)
        plt.close()
    print(f"Done. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
