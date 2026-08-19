"""What model class removes the residual wobble the affine cannot reach?

Fits, per frame, block displacements to GT and compares nested model classes:

  affine      6 params  (what stabilize_affine --mode affine already does)
  +polyY      affine, but the ALONG-FILM displacement gets a cubic in y
  +rowfree    affine, but the ALONG-FILM displacement is free per block row
              (the upper bound on any purely 1-D along-film warp)

WHY ALONG-FILM AND WHY 1-D. The known residual error mode of this pipeline is
arc-length parameterisation WITHIN a winding: the columns of a frame are cut at
slightly wrong film positions, which STRETCHES the frame along the film,
non-uniformly. np.rot90 in make_film_video puts along-film on the video's
VERTICAL axis and z (across film) on the horizontal, so that error shows up as
a vertical stretch varying down the frame -- quadratic or worse in y, hence not
reachable by a 6-parameter affine, hence still there after stabilize_affine.

If R_rowfree << R_affine the wobble is 1-D along-film and a cheap structured
warp fixes it with NO loss of sharpness (every pixel keeps its neighbours along
a scanline). If R_rowfree ~ R_affine the residual is genuinely 2-D and only a
dense flow could touch it -- which costs image quality.

Analysed on the PRE-affine cut. stabilize_affine crops 4% off every edge and
rescales to full height, so its output content is ~8.7% larger than its input;
comparing that to GT reads the zoom as a constant scale error (~21 px at the
frame edge) and corrupts every residual. Do not compare a cropped output
against GT without undoing that.

  python Scripts/diag_affine_wobble.py --in .../FLAGSHIP_borderlock.mp4 \
      --gt .../gt_perflock_picture.mp4 --fps 26 --focus 4.0 --out-dir .../diag
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from unwrapping.eval.stabilize_horizontal import load_luma, gmag, _resize
from unwrapping.eval.compare_videos import load_video_luma
from unwrapping.eval.stabilize_affine import grid_disp


def fits(P, D, G):
    """Residual (median px) of the nested models. P=(px,py), D=(dx,dy).

    Video axes: x = z = ACROSS film, y = ALONG film. The along-film component
    is D[:,1] and the one we allow to get richer.
    """
    A = np.column_stack([np.ones(len(P)), P[:, 0], P[:, 1]])
    c, *_ = np.linalg.lstsq(A, D, rcond=None)
    r_aff = np.hypot(*(D - A @ c).T)

    # +polyY: dx stays affine, dy becomes a cubic in py
    py = P[:, 1]
    s = py.std() + 1e-9
    Ay = np.column_stack([np.ones(len(py)), P[:, 0], py / s,
                          (py / s) ** 2, (py / s) ** 3])
    cy, *_ = np.linalg.lstsq(Ay, D[:, 1], rcond=None)
    dy_res = D[:, 1] - Ay @ cy
    dx_res = D[:, 0] - A @ c[:, 0]
    r_poly = np.hypot(dx_res, dy_res)

    # +rowfree: dy free per block row (upper bound for a 1-D along-film warp)
    rows = np.unique(py)
    dy_free = D[:, 1].copy()
    for r in rows:
        m = py == r
        dy_free[m] = D[m, 1] - D[m, 1].mean()
    r_row = np.hypot(dx_res, dy_free)

    return (float(np.median(r_aff)), float(np.median(r_poly)),
            float(np.median(r_row)),
            float(c[1, 0]), float(c[2, 1]), float(0.5 * (c[1, 1] - c[2, 0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--grid", type=int, default=7)
    ap.add_argument("--fps", type=float, default=26.0)
    ap.add_argument("--pitch", type=float, default=1130.0)
    ap.add_argument("--focus", type=float, default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fr = load_luma(args.inp)
    n, (H, W) = len(fr), fr[0].shape
    gt = [_resize(g, H, W) for g in load_video_luma(args.gt)]
    idx = np.linspace(0, len(gt) - 1, n).round().astype(int)
    print(f"{n} frames {H}x{W}; grid {args.grid}x{args.grid}  "
          f"(video x = z = across film, y = along film)", flush=True)

    M = []
    for j in range(n):
        P, D, _ = grid_disp(gmag(fr[j]), gmag(gt[idx[j]]), args.grid)
        M.append(fits(P, D, args.grid))
    M = np.array(M)
    sec = np.arange(n) / args.fps
    cell = (n - 1 - np.arange(n))
    r_aff, r_poly, r_row = M[:, 0], M[:, 1], M[:, 2]

    print(f"\n=== reel-wide median residual (px) ===", flush=True)
    print(f"  affine   {np.median(r_aff):6.2f}", flush=True)
    print(f"  +polyY   {np.median(r_poly):6.2f}   "
          f"({100*(1-np.median(r_poly)/np.median(r_aff)):+.0f}%)", flush=True)
    print(f"  +rowfree {np.median(r_row):6.2f}   "
          f"({100*(1-np.median(r_row)/np.median(r_aff)):+.0f}%)", flush=True)

    print(f"\n=== worst 15 frames by affine residual ===", flush=True)
    print("  frame    sec   cell     col   affine  +polyY +rowfree", flush=True)
    for j in np.argsort(-r_aff)[:15]:
        print(f"  {j:5d} {sec[j]:6.2f} {cell[j]:6d} {int(cell[j]*args.pitch):8d} "
              f"  {r_aff[j]:6.2f}  {r_poly[j]:6.2f}  {r_row[j]:6.2f}", flush=True)

    if args.focus is not None:
        lo = max(0, int((args.focus - 1.0) * args.fps))
        hi = min(n, int((args.focus + 1.0) * args.fps))
        o = slice(None)
        print(f"\n=== window {args.focus-1:.1f}-{args.focus+1:.1f}s "
              f"(frames {lo}-{hi}) vs reel ===", flush=True)
        print(f"            affine  +polyY +rowfree", flush=True)
        print(f"  window   {np.median(r_aff[lo:hi]):6.2f}  "
              f"{np.median(r_poly[lo:hi]):6.2f}  {np.median(r_row[lo:hi]):6.2f}",
              flush=True)
        print(f"  reel     {np.median(r_aff[o]):6.2f}  {np.median(r_poly[o]):6.2f}"
              f"  {np.median(r_row[o]):6.2f}", flush=True)

    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axs[0].plot(sec, r_aff, lw=1.1, label="affine (6 param)")
    axs[0].plot(sec, r_poly, lw=1.1, label="+ cubic along-film")
    axs[0].plot(sec, r_row, lw=1.1, label="+ free per row (1-D bound)")
    axs[0].set_ylabel("median block residual (px)")
    axs[0].legend(fontsize=8); axs[0].grid(alpha=0.3)
    axs[1].plot(sec, M[:, 3], lw=1.0, label="scale_x (across)")
    axs[1].plot(sec, M[:, 4], lw=1.0, label="scale_y (along)")
    axs[1].plot(sec, M[:, 5], lw=1.0, label="rot")
    axs[1].set_ylabel("affine component"); axs[1].set_xlabel("video seconds")
    axs[1].legend(fontsize=8); axs[1].grid(alpha=0.3)
    if args.focus is not None:
        for ax in axs:
            ax.axvspan(args.focus - 1.0, args.focus + 1.0, color="r", alpha=0.10)
    p = os.path.join(args.out_dir, "affine_wobble.png")
    fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    print(f"\nwrote {p}", flush=True)


if __name__ == "__main__":
    main()
