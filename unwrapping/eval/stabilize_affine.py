"""Stabilized VIEWING COPY: remove the per-frame geometric distortion (scale /
rotation / shear / warp) the unrolling leaves, using GT as the still, content-
matched reference.

  --mode affine : per-frame affine (translation+scale+rotation+shear) fit to
                  GT and applied. Keeps OUR pixels, only corrects the geometric
                  breathing. An honest "stabilized viewing copy".
  --mode dense  : per-frame dense (block-flow) warp to GT -> also removes the
                  non-rigid leftover. Near-perfect stillness, but this
                  reprojects our frames onto GT's geometry (viewing aid only).

NOT a faithful-reconstruction output -- label any figure made from it as a
GT-stabilized viewing copy. The real fix is geometric consistency in the
unrolling itself.

Usage:
  python -m unwrapping.eval.stabilize_affine \
    --in unwrapping/eval/results/frame_match_v12/film_stabilized.mp4 \
    --gt data/h265_1080p.mp4 --mode affine \
    --out unwrapping/eval/results/frame_match_v12/film_affine.mp4
"""

import argparse

import numpy as np
import imageio.v2 as imageio
from PIL import Image
from scipy.ndimage import zoom, gaussian_filter, map_coordinates

from .stabilize_horizontal import load_luma, gmag, phase2d, _resize
from .compare_videos import (load_video_luma, trim_leader, parse_crop,
                             apply_crop)


def pair_frames(frames, gt, idx0, search, out_png=None, match_h=256,
                w_motion=0.7, lam=0.35):
    """Pair each of our frames with the GT frame it actually SHOWS.

    The default pairing is `linspace(0, len(gt)-1, n)`, correct only when the
    two videos already correspond frame-for-frame -- true of the old
    frame_match chain (234 ours vs 233 GT) and FALSE of a raw render (229 vs
    233, the outermost wrap's film length missing). linspace spreads that
    deficit uniformly instead of putting it where it physically belongs, so the
    mispairing is zero at both ends and maximal in the MIDDLE of the reel.

    SCORING ADJACENT FILM FRAMES IS THE WHOLE DIFFICULTY, and a plain
    correlation does NOT do it: neighbouring cells differ only in small moving
    regions, so a global score is dominated by the static layout and returns
    noise (measured: best-match offsets uniformly scattered over the entire
    +/-4 search range). This reuses frame_match's machinery, which was built
    for exactly this discrimination:
      * _lcn        contrast normalisation, so background/shading gradients do
                    not swamp the pose evidence separating adjacent cells;
      * motion_masks the region that differs from the temporal neighbours --
                    "a global score barely sees a closing mouth";
      * a DP over the sequence with a |delta offset| penalty, instead of a
                    per-frame argmax, since the true correspondence is smooth
                    and a per-frame pick injects the jitter we are removing.

    Returns the corrected index array.
    """
    from scipy.ndimage import zoom as _zoom
    from .frame_match import _lcn, motion_masks, weighted_corr
    from .compare_videos import grad_mag
    from . import metrics as M

    def sm(f):
        return _zoom(f, match_h / f.shape[0], order=1)

    ours = [_lcn(sm(f)) for f in frames]
    gtm = [sm(g) for g in gt]
    gts = [_lcn(g) for g in gtm]
    masks = motion_masks(gtm)
    n, ng = len(ours), len(gts)
    offs = np.arange(-search, search + 1)

    E = np.full((n, len(offs)), -2.0)
    for j in range(n):
        for k, d in enumerate(offs):
            i = idx0[j] + d
            if not (0 <= i < ng):
                continue
            s = M.gradient_correlation(gts[i], ours[j])
            if masks[i] is not None:
                s += w_motion * weighted_corr(grad_mag(gts[i]),
                                              grad_mag(ours[j]), masks[i])
            E[j, k] = s

    # DP: maximise sum of emissions minus lam*|change of offset|
    V = E[0].copy()
    back = np.zeros((n, len(offs)), int)
    for j in range(1, n):
        prev = V[None, :] - lam * np.abs(offs[:, None] - offs[None, :])
        back[j] = prev.argmax(1)
        V = E[j] + prev.max(1)
    path = np.zeros(n, int)
    path[-1] = int(V.argmax())
    for j in range(n - 1, 0, -1):
        path[j - 1] = back[j, path[j]]
    dp = offs[path]
    argmax = offs[E.argmax(1)]

    # is the score discriminative at all? margin between best and 2nd best
    Es = np.sort(E, 1)
    margin = float(np.median(Es[:, -1] - Es[:, -2]))
    agree = float((argmax == dp).mean())
    idx = np.clip(idx0 + dp, 0, ng - 1)
    print(f"  GT pairing: searched +/-{search}; per-frame argmax "
          f"{argmax.min():+d}..{argmax.max():+d}, DP path {dp.min():+d}..{dp.max():+d}, "
          f"argmax==DP {100*agree:.0f}%, median best-vs-2nd margin {margin:.4f}",
          flush=True)
    print(f"  shifted {int(np.abs(idx - idx0).max())} frames max vs linspace",
          flush=True)
    if out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(argmax, ".", ms=4, label="per-frame argmax (noisy if flat)")
        ax.plot(dp, lw=2, label="DP path (used)")
        ax.set_xlabel("our frame"); ax.set_ylabel("GT offset (frames)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)
        print(f"  wrote {out_png}", flush=True)
    return idx


def grid_disp(o_g, g_g, G, lim=0.25):
    """Displacement (to align our onto GT) on a GxG block grid. Returns block
    centres P (about frame centre) and displacements D=(dx,dy)."""
    H, W = o_g.shape
    ys = np.linspace(0, H, G + 1).astype(int)
    xs = np.linspace(0, W, G + 1).astype(int)
    P, D = [], []
    for a in range(G):
        for b in range(G):
            gb = g_g[ys[a]:ys[a+1], xs[b]:xs[b+1]]
            ob = o_g[ys[a]:ys[a+1], xs[b]:xs[b+1]]
            dy, dx = phase2d(gb, ob, lim)
            P.append([(xs[b]+xs[b+1])/2 - W/2, (ys[a]+ys[a+1])/2 - H/2])
            D.append([dx, dy])
    return np.array(P), np.array(D), (ys, xs)


def affine_field(P, D, H, W):
    """Robust affine fit d = t + M p ; return dense (Dy, Dx) over the frame."""
    A = np.column_stack([np.ones(len(P)), P[:, 0], P[:, 1]])
    coef, *_ = np.linalg.lstsq(A, D, rcond=None)
    for _ in range(2):                              # 2 robust reweightings
        r = np.hypot(*(D - A @ coef).T)
        w = 1.0 / (1.0 + (r / (np.median(r) + 1e-6)) ** 2)
        Aw = A * w[:, None]
        coef, *_ = np.linalg.lstsq(Aw, D * w[:, None], rcond=None)
    yy, xx = np.mgrid[0:H, 0:W]
    px = xx - W / 2; py = yy - H / 2
    Dx = coef[0, 0] + coef[1, 0] * px + coef[2, 0] * py
    Dy = coef[0, 1] + coef[1, 1] * px + coef[2, 1] * py
    return Dy, Dx


def dense_field(D, grid, H, W, smooth=1.2):
    """Dense (Dy, Dx) from the block grid: reshape -> zoom to full res ->
    smooth (a coarse optical-flow-to-GT)."""
    ys, xs = grid
    G = len(ys) - 1
    Dx = D[:, 0].reshape(G, G)
    Dy = D[:, 1].reshape(G, G)
    Dx = zoom(Dx, (H / G, W / G), order=1)
    Dy = zoom(Dy, (H / G, W / G), order=1)
    Dx = gaussian_filter(Dx, smooth * H / G)
    Dy = gaussian_filter(Dy, smooth * H / G)
    return Dy, Dx


def warp(frame, Dy, Dx, sign):
    H, W = frame.shape
    yy, xx = np.mgrid[0:H, 0:W]
    return map_coordinates(frame, [yy + sign * Dy, xx + sign * Dx],
                           order=1, mode="nearest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["affine", "dense"], default="affine")
    ap.add_argument("--gt-crop", default="0.235,0.12,0.81,0.88",
                    help="'none' when --gt is already cut to the picture, e.g. "
                         "gt_perflock.py's gt_perflock_picture.mp4.")
    ap.add_argument("--no-gt-trim", action="store_true",
                    help="Skip the leader trim. Required for an already-cut GT, "
                         "whose leader was dropped when it was made -- trimming "
                         "twice shifts the frame pairing by a few frames.")
    ap.add_argument("--grid", type=int, default=0,
                    help="block grid size (0 = auto: 5 affine / 10 dense).")
    ap.add_argument("--iters", type=int, default=3,
                    help="accumulate the correction over N passes (converge).")
    ap.add_argument("--fps", type=int, default=26)
    ap.add_argument("--extra-crop", type=float, default=0.04)
    ap.add_argument("--gt-search", type=int, default=0,
                    help="Pair frames by CONTENT within +/-N GT frames instead "
                         "of assuming linspace. Needed whenever our frame count "
                         "differs from GT's (a raw render: 229 vs 233) -- see "
                         "pair_frames(). 0 = old behaviour.")
    args = ap.parse_args()

    frames = load_luma(args.inp)
    n = len(frames); H, W = frames[0].shape
    gt = load_video_luma(args.gt)
    if not args.no_gt_trim:
        gt, _ = trim_leader(gt)
    if args.gt_crop.lower() not in ("none", ""):
        gt = apply_crop(gt, parse_crop(args.gt_crop))
    gt = [_resize(g, H, W) for g in gt]
    idx = np.linspace(0, len(gt) - 1, n).round().astype(int)
    if args.gt_search:
        idx = pair_frames(frames, gt, idx, args.gt_search,
                          out_png=args.out + ".pairing.png")
    G = args.grid or (5 if args.mode == "affine" else 10)
    print(f"{n} frames {H}x{W}; mode={args.mode} grid={G}x{G}", flush=True)

    def field_of(fr, j):
        P, D, grid = grid_disp(gmag(fr), gmag(gt[idx[j]]), G)
        if args.mode == "affine":
            return affine_field(P, D, H, W)
        return dense_field(D, grid, H, W)

    # resolve warp sign once (which reduces the residual to GT)
    f0 = [field_of(frames[j], j) for j in range(0, n, 8)]
    def resid(sign):
        s = 0.0
        for k, j in enumerate(range(0, n, 8)):
            w = warp(frames[j], *f0[k], sign)
            dy, dx = phase2d(gmag(gt[idx[j]]), gmag(w), 0.25)
            s += abs(dy) + abs(dx)
        return s
    sign = 1.0 if resid(1.0) <= resid(-1.0) else -1.0

    # ITERATE: accumulate the correction field so the deformation converges to
    # GT (each pass measures the residual on the current warp and adds it).
    tot = [(np.zeros((H, W)), np.zeros((H, W))) for _ in range(n)]
    for it in range(args.iters):
        for j in range(n):
            w = warp(frames[j], tot[j][0], tot[j][1], sign)
            Dy, Dx = field_of(w, j)
            tot[j] = (tot[j][0] + Dy, tot[j][1] + Dx)
    warped = [warp(frames[j], tot[j][0], tot[j][1], sign) for j in range(n)]

    # verify stillness gain: per-frame affine deformation std, before vs after
    def deform_std(fr):
        comps = []
        for j in range(0, n, 3):
            P, D, _ = grid_disp(gmag(fr[j]), gmag(gt[idx[j]]), 4)
            A = np.column_stack([np.ones(len(P)), P[:, 0], P[:, 1]])
            c, *_ = np.linalg.lstsq(A, D, rcond=None)
            comps.append([c[1, 0], c[2, 1], 0.5 * (c[1, 1] - c[2, 0])])  # sx,sy,rot
        comps = np.array(comps)
        return comps.std(0)
    b = deform_std(frames); a = deform_std(warped)
    print(f"  frame-to-frame deformation std (scale-x,scale-y,rot):", flush=True)
    print(f"    before {b.round(4)}  ->  after {a.round(4)}", flush=True)

    # crop to a safe common region (deformation warps pull in edges)
    ec = args.extra_crop
    y0, y1 = int(H * ec), int(H * (1 - ec))
    x0, x1 = int(W * ec), int(W * (1 - ec))
    cropped = [w[y0:y1, x0:x1] for w in warped]
    ch, cw = cropped[0].shape
    Wf = int(round(H * cw / ch)); Wf += Wf % 2
    wr = imageio.get_writer(args.out, fps=args.fps, codec="libx264",
                            quality=9, macro_block_size=1)
    for c in cropped:
        wr.append_data(np.asarray(Image.fromarray(
            (np.clip(c, 0, 1) * 255).astype(np.uint8)).resize((Wf, H))))
    wr.close()
    print(f"Done -> {args.out}  ({H}x{Wf}); GT-STABILIZED VIEWING COPY "
          f"({args.mode}) -- label as such.", flush=True)


if __name__ == "__main__":
    main()
