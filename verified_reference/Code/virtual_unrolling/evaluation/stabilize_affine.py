"""GT-anchored affine stabilisation: remove the per-frame geometric distortion
(scale / rotation / shear) the unrolling leaves, using the optical scan as the
still, content-matched reference.

Frame j is paired with GT frame linspace(0, n_gt-1, n)[j]; one affine field per
frame (block phase correlation on gradient maps, robust least squares) is fitted
and accumulated over 3 passes; the result is cropped by 4 % on every side. Our
pixels are kept, only the geometry is corrected, so this is a GT-stabilised copy,
not a GT-free reconstruction.

Usage:
  python -m virtual_unrolling.evaluation.stabilize_affine \
    --in <frame_match>/film_stabilized_gtsync.mp4 --gt $GT_VIDEO --out film_affine.mp4
"""

import argparse

import numpy as np
import imageio.v2 as imageio
from PIL import Image
from scipy.ndimage import map_coordinates

from .video_utils import (load_luma, gmag, phase2d, _resize, load_video_luma,
                          trim_leader, parse_crop, apply_crop)


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
    ap.add_argument("--gt-crop", default="0.235,0.12,0.81,0.88",
                    help="picture-area crop of the GT frames (x0,y0,x1,y1 fractions)")
    args = ap.parse_args()

    frames = load_luma(args.inp)
    n = len(frames); H, W = frames[0].shape
    gt = load_video_luma(args.gt)
    gt, _ = trim_leader(gt)
    gt = apply_crop(gt, parse_crop(args.gt_crop))
    gt = [_resize(g, H, W) for g in gt]
    idx = np.linspace(0, len(gt) - 1, n).round().astype(int)
    G = 5                                          # block grid
    print(f"{n} frames {H}x{W}; mode=affine grid={G}x{G}", flush=True)

    def field_of(fr, j):
        P, D, grid = grid_disp(gmag(fr), gmag(gt[idx[j]]), G)
        return affine_field(P, D, H, W)

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
    for it in range(3):
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
    print("  frame-to-frame deformation std (scale-x,scale-y,rot):", flush=True)
    print(f"    before {b.round(4)}  ->  after {a.round(4)}", flush=True)

    # crop to a safe common region (deformation warps pull in edges)
    ec = 0.04
    y0, y1 = int(H * ec), int(H * (1 - ec))
    x0, x1 = int(W * ec), int(W * (1 - ec))
    cropped = [w[y0:y1, x0:x1] for w in warped]
    ch, cw = cropped[0].shape
    Wf = int(round(H * cw / ch)); Wf += Wf % 2
    wr = imageio.get_writer(args.out, fps=26, codec="libx264",
                            quality=9, macro_block_size=1)
    for c in cropped:
        wr.append_data(np.asarray(Image.fromarray(
            (np.clip(c, 0, 1) * 255).astype(np.uint8)).resize((Wf, H))))
    wr.close()
    print(f"Done -> {args.out}  ({H}x{Wf}); GT-STABILIZED VIEWING COPY "
          "(affine) -- label as such.", flush=True)


if __name__ == "__main__":
    main()
