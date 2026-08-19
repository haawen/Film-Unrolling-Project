"""Re-cut the GT telecine scan on ITS OWN perforations.

The GT scan is not a still reference.  Measured against its own temporal
median it wanders 7.2 px and breathes 13.5 px in size (1626 px = 16 mm), i.e.
about 6 px and 12 px in the units of our 970 px render.  Our border-locked
render holds the across-film axis to 2.5 px, so warping our frames onto raw GT
*imports* motion instead of removing it -- which is exactly what the first
affine attempt did (across-film 2.5 -> 11.4 px).

Unlike our render, GT's perforations are REAL: they are scanned film, not
geometry the walk extrapolated into an untraced z band.  Four holes are visible
per frame (two per edge, one pair at each end of the pitch), they threshold out
cleanly at 0.90, and they are found in 221 of 233 frames.  Four points give a
full affine -- translation, both scales, rotation and shear -- so GT can be
locked the way we originally wanted to lock our own render.

Outputs
  gt_perflock_sprockets.mp4  full width, perforations kept (the "GT with
                             sprockets" reference)
  gt_perflock_picture.mp4    cropped to the aperture, measured from the
                             temporal mean of the locked reel, for use as an
                             affine target against our picture-only render
  perflock.npz               per-frame perf points, the reference configuration
                             and the measured aperture box
  gt_perflock_check.png      detections drawn on sample frames

    python -m unwrapping.eval.gt_perflock --gt data/h265_1080p.mp4 \
        --out-dir unwrapping/eval/results/stabilization/gt
"""

import argparse
import os

import numpy as np


def find_perfs(frame, thresh=0.90, min_area=10000, w_rng=(120, 230),
               h_rng=(80, 180)):
    """The four perforation centroids of one GT frame, as a (4, 2) array of
    (x, y) ordered [top-left, top-right, bottom-left, bottom-right].

    Missing corners come back as NaN; the caller fills them from the reel.
    """
    import cv2
    H, W = frame.shape
    th = (frame > thresh).astype(np.uint8)
    nl, _, st, cen = cv2.connectedComponentsWithStats(th, 8)
    out = np.full((4, 2), np.nan)
    for i in range(1, nl):
        x, y, w, h, a = st[i]
        if a < min_area or not (w_rng[0] < w < w_rng[1]) or \
           not (h_rng[0] < h < h_rng[1]):
            continue
        cx, cy = cen[i]
        k = (2 if cy > H / 2 else 0) + (1 if cx > W / 2 else 0)
        if np.isnan(out[k, 0]) or a > 0:      # keep the last/biggest per corner
            out[k] = (cx, cy)
    return out


def fill_tracks(P):
    """Interpolate missing corners over the reel.

    A corner is missing where the scan is torn (frames 192-197 here) or where
    the reel runs out (0-1, 229-232).  Each coordinate is smooth in frame index,
    so linear interpolation with edge hold is enough; nothing is invented that
    the neighbours do not already say.
    """
    P = np.array(P, float)
    n = len(P)
    g = np.arange(n)
    nbad = 0
    for k in range(4):
        for c in range(2):
            y = P[:, k, c]
            ok = np.isfinite(y)
            nbad += int((~ok).sum())
            if ok.sum() < 5:
                raise SystemExit(f"corner {k} coord {c}: too few detections")
            P[:, k, c] = np.interp(g, g[ok], y[ok])
    return P, nbad


def affine_to_ref(src, ref):
    """Least-squares affine (2x3) taking `src` points onto `ref` points."""
    A = np.column_stack([src, np.ones(len(src))])
    M, *_ = np.linalg.lstsq(A, ref, rcond=None)
    return M.T.astype(np.float32)


def aperture_box(mean_img, pts_ref, pad_frac=0.06):
    """Measure the picture aperture in the locked reel's temporal mean.

    Locking makes the frame lines sharp in the mean, so the aperture edges are
    the strongest dark lines: the two vertical ones between the perforation
    columns, and the two horizontal ones at the ends of the pitch.  Measuring
    beats assuming the ISO aperture, because on this reel the aperture edge and
    the inner edge of the perforation are only ~0.1 mm apart -- a spec-derived
    box would sit on top of a hole.
    """
    from scipy.ndimage import gaussian_filter1d
    H, W = mean_img.shape
    xl, xr = pts_ref[:, 0].min(), pts_ref[:, 0].max()
    yt, yb = pts_ref[:, 1].min(), pts_ref[:, 1].max()

    # ACROSS: search inboard of each perforation column for the darkest line.
    col = gaussian_filter1d(mean_img.mean(axis=0), 2.0)
    span = xr - xl
    lo = slice(int(xl), int(xl + pad_frac * span))
    hi = slice(int(xr - pad_frac * span), int(xr))
    x0 = lo.start + int(np.argmin(col[lo]))
    x1 = hi.start + int(np.argmin(col[hi]))

    # ALONG: the frame line sits near the perforation row; search around it.
    row = gaussian_filter1d(mean_img.mean(axis=1), 2.0)
    sp = yb - yt
    a = slice(max(0, int(yt - pad_frac * sp)), int(yt + pad_frac * sp))
    b = slice(int(yb - pad_frac * sp), min(H, int(yb + pad_frac * sp)))
    y0 = a.start + int(np.argmin(row[a]))
    y1 = b.start + int(np.argmin(row[b]))
    return np.array([x0, y0, x1, y1], float)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", default="data/h265_1080p.mp4")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--thresh", type=float, default=0.90)
    ap.add_argument("--smooth-frames", type=float, default=0.0,
                    help="Optional smoothing of the perf tracks, in frames. 0 "
                         "locks to every frame's own measurement, which is what "
                         "a fiducial is for; raise it only if the detections "
                         "turn out noisy.")
    ap.add_argument("--fps", type=int, default=26)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    import cv2
    import imageio.v2 as imageio
    from PIL import Image
    from .compare_videos import load_video_luma, trim_leader

    os.makedirs(args.out_dir, exist_ok=True)
    gt = load_video_luma(args.gt)
    gt, nlead = trim_leader(gt)
    n, H, W = gt.shape
    print(f"GT {n} frames {H}x{W} (leader {nlead} dropped)", flush=True)

    P = np.stack([find_perfs(gt[j], args.thresh) for j in range(n)])
    found = np.isfinite(P[:, :, 0]).all(axis=1)
    print(f"  all four perforations in {found.sum()}/{n} frames", flush=True)
    P, nbad = fill_tracks(P)
    if args.smooth_frames > 0:
        from scipy.ndimage import gaussian_filter1d
        P = gaussian_filter1d(P, args.smooth_frames, axis=0, mode="nearest")
    print(f"  {nbad} corner coordinates interpolated", flush=True)

    ref = np.median(P, axis=0)
    print("  reference configuration (x, y):", flush=True)
    for k, nm in enumerate(["top-left", "top-right", "bot-left", "bot-right"]):
        print(f"    {nm:9s} {ref[k, 0]:8.1f} {ref[k, 1]:8.1f}", flush=True)
    px_mm_x = (ref[1, 0] - ref[0, 0]) / 12.18      # hole centres, measured
    px_mm_y = (ref[2, 1] - ref[0, 1]) / 7.62       # one frame pitch
    print(f"  scale {px_mm_x:.1f} px/mm across, {px_mm_y:.1f} px/mm along",
          flush=True)

    # residual of each frame from the reference, before and after
    def resid(Q):
        return float(np.sqrt(np.mean((Q - ref) ** 2)))
    warped = np.empty_like(gt)
    Pfix = np.empty_like(P)
    for j in range(n):
        M = affine_to_ref(P[j], ref)
        warped[j] = cv2.warpAffine(gt[j], M, (W, H), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
        Pfix[j] = (np.column_stack([P[j], np.ones(4)]) @ M.T)
    print(f"  perf RMS from reference: {resid(P):.2f} px -> "
          f"{resid(Pfix):.2f} px", flush=True)

    mean_img = warped.mean(axis=0)
    box = aperture_box(mean_img, ref)
    x0, y0, x1, y1 = box.astype(int)
    print(f"  aperture measured at x {x0}..{x1} ({x1 - x0} px = "
          f"{(x1 - x0) / px_mm_x:.2f} mm), y {y0}..{y1} ({y1 - y0} px = "
          f"{(y1 - y0) / px_mm_y:.2f} mm)", flush=True)

    lo, hi = np.percentile(warped[::5], [1, 99])

    def write(frames, name, height):
        h0, w0 = frames[0].shape
        Wf = int(round(height * w0 / h0)); Wf += Wf % 2
        path = os.path.join(args.out_dir, name)
        wr = imageio.get_writer(path, fps=args.fps, codec="libx264", quality=9,
                                macro_block_size=1)
        for f in frames:
            x = np.clip((f - lo) / (hi - lo + 1e-9), 0, 1)
            wr.append_data(np.asarray(Image.fromarray(
                (x * 255).astype(np.uint8)).resize((Wf, height))))
        wr.close()
        print(f"wrote {path}  ({height}x{Wf}, {len(frames)} frames)", flush=True)

    write(warped, "gt_perflock_sprockets.mp4", args.height)
    write(warped[:, y0:y1, x0:x1], "gt_perflock_picture.mp4", args.height)

    # check sheet: detections on the raw frames
    tiles = []
    for j in np.linspace(0, n - 1, 8).astype(int):
        im = cv2.cvtColor((np.clip(gt[j], 0, 1) * 255).astype(np.uint8),
                          cv2.COLOR_GRAY2BGR)
        for k in range(4):
            cv2.circle(im, tuple(np.round(P[j, k]).astype(int)), 26,
                       (0, 0, 255) if found[j] else (0, 165, 255), 5)
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 255, 0), 4)
        cv2.putText(im, f"f{j}" + ("" if found[j] else " INTERP"), (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)
        tiles.append(cv2.resize(im, (406, 270)))
    sheet = np.concatenate([np.concatenate(tiles[:4], axis=1),
                            np.concatenate(tiles[4:], axis=1)], axis=0)
    cv2.imwrite(os.path.join(args.out_dir, "gt_perflock_check.png"), sheet)

    np.savez(os.path.join(args.out_dir, "perflock.npz"), pts=P, ref=ref,
             box=box, found=found, px_mm=(px_mm_x, px_mm_y), n_leader=nlead)
    print("done", flush=True)


if __name__ == "__main__":
    main()
