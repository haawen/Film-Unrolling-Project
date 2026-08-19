"""Option 3 done properly: dense OPTICAL FLOW warp to GT (per-pixel), removing
the local/non-rigid distortion the affine can't reach.

For each frame, compute a dense Farneback flow from OUR frame to its matched GT
frame -- on EDGE maps (gradient magnitude), since our CT-derived intensities and
GT's optical intensities break the brightness-constancy flow assumes but their
edges match -- then remap our INTENSITY frame by that flow so our picture takes
GT's geometry. Multi-scale (pyramid) + smoothing keeps the field clean.

This is a GT-STABILIZED VIEWING COPY (our pixels reprojected onto GT's shape) --
near-perfect stillness, but NOT a faithful-reconstruction output. Label figures
made from it as such. The honest fix remains geometric consistency in the
unrolling itself.

Usage:
  python -m unwrapping.eval.stabilize_flow \
    --in unwrapping/eval/results/frame_match_v12/film_stabilized.mp4 \
    --gt data/h265_1080p.mp4 \
    --out unwrapping/eval/results/frame_match_v12/film_flow.mp4
"""

import argparse

import numpy as np
import cv2
import imageio.v2 as imageio
from PIL import Image

from .stabilize_horizontal import load_luma, gmag, phase2d, _resize
from .compare_videos import (load_video_luma, trim_leader, parse_crop,
                             apply_crop)


def edge_u8(f):
    """Modality-robust edge map, mildly blurred, as uint8 for cv2 flow."""
    g = gmag(f)
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.2)
    g -= g.min(); mx = g.max()
    return (g / mx * 255).astype(np.uint8) if mx > 0 else g.astype(np.uint8)


def flow_to_gt(our, gt, max_disp):
    """Dense flow mapping our->gt (on edges), clipped to +/-max_disp px so a
    bad local match can't tear the frame. Returns (fy, fx)."""
    fixed, moving = edge_u8(gt), edge_u8(our)
    fl = cv2.calcOpticalFlowFarneback(
        fixed, moving, None, pyr_scale=0.5, levels=4, winsize=41,
        iterations=5, poly_n=7, poly_sigma=1.5, flags=0)   # fixed(p)=moving(p+fl)
    fx, fy = fl[..., 0], fl[..., 1]
    fx = cv2.GaussianBlur(fx, (0, 0), 4.0)
    fy = cv2.GaussianBlur(fy, (0, 0), 4.0)
    np.clip(fx, -max_disp, max_disp, fx)
    np.clip(fy, -max_disp, max_disp, fy)
    return fy, fx


def warp(frame, fy, fx):
    H, W = frame.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(frame.astype(np.float32), (xx + fx).astype(np.float32),
                     (yy + fy).astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def deform_std(frames, gtl, idx):
    """Frame-to-frame std of the affine scale/rotation vs GT (the wobble)."""
    comps = []
    for j in range(0, len(frames), 3):
        H, W = frames[j].shape
        ys = np.linspace(0, H, 5).astype(int); xs = np.linspace(0, W, 5).astype(int)
        P, D = [], []
        og, gg = gmag(frames[j]), gmag(gtl[idx[j]])
        for a in range(4):
            for b in range(4):
                dy, dx = phase2d(gg[ys[a]:ys[a+1], xs[b]:xs[b+1]],
                                 og[ys[a]:ys[a+1], xs[b]:xs[b+1]], 0.25)
                P.append([(xs[b]+xs[b+1])/2 - W/2, (ys[a]+ys[a+1])/2 - H/2])
                D.append([dx, dy])
        A = np.column_stack([np.ones(16), np.array(P)[:, 0], np.array(P)[:, 1]])
        c, *_ = np.linalg.lstsq(A, np.array(D), rcond=None)
        comps.append([c[1, 0], c[2, 1], 0.5 * (c[1, 1] - c[2, 0])])
    return np.array(comps).std(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-crop", default="0.235,0.12,0.81,0.88",
                    help="'none' when --gt is already cut to the picture, e.g. "
                         "gt_perflock.py's gt_perflock_picture.mp4.")
    ap.add_argument("--no-gt-trim", action="store_true",
                    help="Skip the leader trim. Required for an already-cut GT, "
                         "whose leader was dropped when it was made -- trimming "
                         "twice shifts the frame pairing by a few frames.")
    ap.add_argument("--max-disp", type=float, default=45.0)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--extra-crop", type=float, default=0.05)
    ap.add_argument("--fps", type=int, default=26)
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
    print(f"{n} frames {H}x{W}; dense Farneback flow to GT", flush=True)

    warped = list(frames)
    for it in range(args.iters):          # re-flow the warped frame -> converge
        for j in range(n):
            fy, fx = flow_to_gt(warped[j], gt[idx[j]], args.max_disp)
            warped[j] = warp(warped[j], fy, fx)
        b = deform_std(frames, gt, idx) if it == 0 else b
        a = deform_std(warped, gt, idx)
        print(f"  iter {it}: deformation std (sx,sy,rot) "
              f"{np.round(b,4)} -> {np.round(a,4)}", flush=True)

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
    print(f"Done -> {args.out} ({H}x{Wf}); GT-STABILIZED VIEWING COPY (dense "
          f"flow) -- label as such.", flush=True)


if __name__ == "__main__":
    main()
