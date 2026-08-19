"""Compare the unrolled film video against the GT optical scan.

Pipeline (see unwrapping/eval/__init__.py for rationale):

  Stage 0  load -> luma -> trim GT leader -> (optional) crop picture area
           -> resize to common height
  Stage 1  temporal alignment: DTW on modality-robust frame embeddings
           (downsampled gradient-magnitude maps, cosine cost) -> per-pred-frame
           the best-matching GT frame
  Stage 2  spatial registration per matched pair: FFT phase-correlation
           translation (+ optional cv2 ECC affine if cv2 present); the residual
           shift magnitude is logged as a geometric-fidelity proxy
  Stage 3  scoring: NMI + gradient-correlation + SSIM/MS-SSIM + PSNR on the
           registered, histogram-matched pair; no-reference BRISQUE/NIQE on the
           predicted frames; optional learned DISTS/LPIPS (needs pyiqa)

Outputs (--out-dir): scores.json (aggregate + per-pair), dtw_path.png,
aligned_montage.png (sample pred|GT|diff triplets).

Usage:
  python -m unwrapping.eval.compare_videos \
      --pred unwrapping/inr/results/arc_simple/wholeroll_v2/film_final_p902.mp4 \
      --gt   data/h265_1080p.mp4 \
      --out-dir unwrapping/eval/results/p902_vs_gt \
      [--learned] [--gt-crop 0.05,0.02,0.95,0.98] [--height 256]
"""

import argparse
import json
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from scipy.ndimage import sobel

from . import metrics as M
from .register import register as ecc_register


# --------------------------------------------------------------------------- #
# Stage 0 — load / preprocess
# --------------------------------------------------------------------------- #
def load_video_luma(path):
    """Read an mp4 -> (N, H, W) float32 luma in [0, 1]."""
    r = imageio.get_reader(path)
    frames = []
    for f in r:
        f = np.asarray(f, dtype=np.float32)
        if f.ndim == 3:  # RGB -> Rec.601 luma
            f = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
        frames.append(f)
    r.close()
    arr = np.stack(frames)
    arr -= arr.min()
    mx = arr.max()
    if mx > 0:
        arr /= mx
    return arr


def trim_leader(frames, diff_thresh=1.0 / 255):
    """Drop leading near-duplicate frames (static leader/title before first cut).

    Returns (trimmed_frames, n_dropped). Stops at the first frame whose mean
    abs-difference from its predecessor exceeds the threshold.
    """
    n = 0
    for i in range(1, len(frames)):
        if np.abs(frames[i] - frames[i - 1]).mean() > diff_thresh:
            n = i
            break
    return frames[n:], n


def parse_crop(s):
    if not s:
        return None
    return tuple(float(x) for x in s.split(","))  # x0,y0,x1,y1 fractions


def apply_crop(frames, box):
    if box is None:
        return frames
    _, h, w = frames.shape
    x0, y0, x1, y1 = box
    return frames[:, int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def apply_orientation(frames, rot90=0, fliplr=False):
    """Rotate (k*90 CCW) and/or mirror every frame to match the GT orientation."""
    if rot90:
        frames = np.rot90(frames, k=rot90, axes=(1, 2))
    if fliplr:
        frames = frames[:, :, ::-1]
    return np.ascontiguousarray(frames)


def resize_to_height(frames, height):
    out = []
    for f in frames:
        w = max(2, int(round(height * f.shape[1] / f.shape[0])))
        im = Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8))
        out.append(np.asarray(im.resize((w, height))).astype(np.float32) / 255)
    return out  # list (widths may differ)


def histogram_match(src, ref, bins=256):
    """Match src's intensity histogram to ref's (per-pair, for SSIM/PSNR)."""
    s = src.ravel()
    sv, idx, cnt = np.unique(s, return_inverse=True, return_counts=True)
    s_cdf = np.cumsum(cnt).astype(np.float64) / s.size
    rv, rc = np.unique(ref.ravel(), return_counts=True)
    r_cdf = np.cumsum(rc).astype(np.float64) / ref.size
    interp = np.interp(s_cdf, r_cdf, rv)
    return interp[idx].reshape(src.shape).astype(np.float32)


# --------------------------------------------------------------------------- #
# Stage 1 — temporal alignment (DTW)
# --------------------------------------------------------------------------- #
def frame_embedding(f, size=32):
    """Modality-robust per-frame descriptor: downsampled gradient magnitude,
    zero-mean unit-norm (so cosine similarity == correlation, intensity-blind)."""
    g = np.hypot(sobel(f, axis=0), sobel(f, axis=1))
    im = Image.fromarray((np.clip(g / (g.max() + 1e-12), 0, 1) * 255).astype(np.uint8))
    v = np.asarray(im.resize((size, size))).astype(np.float32).ravel()
    v -= v.mean()
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v


def dtw_align(pred, gt, band_frac=0.25):
    """DTW between two frame lists. Returns (path, cost_matrix).

    Cost = 1 - cosine of gradient embeddings. A Sakoe-Chiba band (band_frac of
    the longer sequence) keeps the path near-diagonal and bounds compute.
    `path` is a list of (i_pred, j_gt) index pairs.
    """
    ep = np.stack([frame_embedding(f) for f in pred])
    eg = np.stack([frame_embedding(f) for f in gt])
    cost = 1.0 - ep @ eg.T  # (Np, Ng)
    Np, Ng = cost.shape
    band = max(int(band_frac * max(Np, Ng)), 1)
    INF = 1e18
    D = np.full((Np + 1, Ng + 1), INF)
    D[0, 0] = 0.0
    for i in range(1, Np + 1):
        jc = (i - 1) * Ng / Np
        for j in range(max(1, int(jc - band)), min(Ng, int(jc + band)) + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1],
                                               D[i - 1, j - 1])
    # backtrack
    i, j, path = Np, Ng, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return path, cost


def linear_map(n_pred, n_gt):
    """Proportional index mapping pred_i -> gt frame. Best when the two
    sequences are already near-1:1 and same-direction (no time warping)."""
    return [int(round(i * (n_gt - 1) / max(n_pred - 1, 1))) for i in range(n_pred)]


def pred_to_gt_map(path, n_pred):
    """For each pred frame, the GT frame on the DTW path with lowest local cost
    (collapse many-to-one). Returns list len n_pred of gt indices (or -1)."""
    best = [-1] * n_pred
    seen = {}
    for ip, jg in path:
        seen.setdefault(ip, []).append(jg)
    for ip, js in seen.items():
        best[ip] = int(np.median(js))
    return best


# --------------------------------------------------------------------------- #
# Stage 2 — spatial registration (phase correlation translation)
# --------------------------------------------------------------------------- #
def grad_mag(img):
    """Normalized Sobel gradient magnitude (modality-invariant edge map)."""
    m = np.hypot(sobel(img, axis=0), sobel(img, axis=1))
    m -= m.min()
    mx = m.max()
    return (m / mx).astype(np.float32) if mx > 0 else m


def phase_correlation_shift(a, b):
    """Estimate (dy, dx) integer shift aligning b to a via FFT cross-power."""
    A = np.fft.rfft2(a - a.mean())
    B = np.fft.rfft2(b - b.mean())
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12
    r = np.fft.irfft2(R, s=a.shape)
    peak = np.unravel_index(np.argmax(r), r.shape)
    dy = peak[0] if peak[0] <= a.shape[0] // 2 else peak[0] - a.shape[0]
    dx = peak[1] if peak[1] <= a.shape[1] // 2 else peak[1] - a.shape[1]
    return int(dy), int(dx)


def apply_shift(img, dy, dx):
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)


def common_crop(a, b):
    """Crop two same-height arrays to common width (left-aligned)."""
    w = min(a.shape[1], b.shape[1])
    return a[:, :w], b[:, :w]


def register_pair(a, b, method):
    """Align b onto a. Returns (a_used, b_aligned, extra_dict).

    'gradphase' (default): FFT phase-correlation translation on gradient maps —
        modality-robust, fast, ignores the smooth shading that wrecked raw
        phase-corr on low-contrast frames.
    'ecc': gradient-LNCC similarity transform (translation+scale+rotation),
        slower (~10s/pair) but recovers scale/rotation.
    'phasecorr': raw-intensity translation (legacy).
    'none': no spatial alignment.
    """
    if method == "ecc":
        b2, info = ecc_register(a, b)  # b warped onto a's grid
        return a, b2, {"reg_lncc": info["lncc"], "scale": info["scale"],
                       "rot_deg": info["rot_deg"],
                       "warp_px": float(np.hypot(info["tx"] * a.shape[1] / 2,
                                                 info["ty"] * a.shape[0] / 2))}
    a2, b2 = common_crop(a, b)
    if method in ("gradphase", "phasecorr"):
        if method == "gradphase":
            dy, dx = phase_correlation_shift(grad_mag(a2), grad_mag(b2))
        else:
            dy, dx = phase_correlation_shift(a2, b2)
        b2 = apply_shift(b2, dy, dx)
        return a2, b2, {"shift_px": float(np.hypot(dy, dx))}
    return a2, b2, {}


def refine_temporal(pred_l, gt_l, p2g, window):
    """For each matched (ip, jg), search jg±window for the GT frame with the
    best gradient-correlation after fast gradient-phase registration. Absorbs
    the small (±frames) sync jitter the linear/DTW map leaves on moving scenes."""
    refined = list(p2g)
    n_gt = len(gt_l)
    for ip, jg in enumerate(p2g):
        if jg < 0:
            continue
        best_j, best_s = jg, -2.0
        for jj in range(max(0, jg - window), min(n_gt, jg + window + 1)):
            a, b, _ = register_pair(gt_l[jj], pred_l[ip], "gradphase")
            s = M.gradient_correlation(a, b)
            if s > best_s:
                best_s, best_j = s, jj
        refined[ip] = best_j
    return refined


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def aggregate(per_pair):
    keys = {k for d in per_pair for k in d}
    agg = {}
    for k in keys:
        vals = [d[k] for d in per_pair if k in d and np.isfinite(d[k])]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_median"] = float(np.median(vals))
    return agg


def _uniform_frames(frames, H, W):
    out = []
    for f in frames:
        im = Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8)).resize((W, H))
        out.append(np.asarray(im).astype(np.uint8))
    return out


def export_aligned_videos(triplets, out_dir, fps=12):
    """Write aligned_pred.mp4 + aligned_gt.mp4 (same WxH, same order) so a
    libvmaf tool (ffmpeg-quality-metrics) can score VMAF on the ALIGNED pair —
    raw videos would mismatch on rotation/crop/temporal. pred = registered +
    histogram-matched; gt = its matched frame."""
    H = triplets[0][0].shape[0]
    W = max(t[0].shape[1] for t in triplets)
    W += W % 2
    for name, idx in [("aligned_pred", 0), ("aligned_gt", 1)]:
        wr = imageio.get_writer(os.path.join(out_dir, f"{name}.mp4"), fps=fps,
                                codec="libx264", quality=9, macro_block_size=1)
        for t in triplets:
            wr.append_data(_uniform_frames([t[idx]], H, W)[0])
        wr.close()


def compute_fid(triplets, device, out_dir):
    """FID between the aligned pred-frame set and GT-frame set (dumps PNGs)."""
    dp = os.path.join(out_dir, "_fid_pred")
    dg = os.path.join(out_dir, "_fid_gt")
    os.makedirs(dp, exist_ok=True)
    os.makedirs(dg, exist_ok=True)
    for i, t in enumerate(triplets):
        Image.fromarray((np.clip(t[0], 0, 1) * 255).astype(np.uint8)).save(
            os.path.join(dp, f"{i:04d}.png"))
        Image.fromarray((np.clip(t[1], 0, 1) * 255).astype(np.uint8)).save(
            os.path.join(dg, f"{i:04d}.png"))
    return M.fid_folders(dp, dg, device)


def save_montage(triplets, out_path, n=6):
    """triplets: list of (pred, gt, diff). Save n evenly-spaced rows."""
    if not triplets:
        return
    idx = np.linspace(0, len(triplets) - 1, min(n, len(triplets)), dtype=int)
    rows = []
    for k in idx:
        p, g, d = triplets[k]
        h = p.shape[0]
        gap = np.ones((h, 4), np.float32)
        rows.append(np.concatenate([p, gap, g, gap, d], axis=1))
    H = max(r.shape[0] for r in rows)
    Wm = max(r.shape[1] for r in rows)
    canvas = np.ones((H * len(rows) + 4 * len(rows), Wm), np.float32)
    y = 0
    for r in rows:
        canvas[y:y + r.shape[0], :r.shape[1]] = r
        y += r.shape[0] + 4
    Image.fromarray((np.clip(canvas, 0, 1) * 255).astype(np.uint8)).save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="Unrolled film mp4.")
    ap.add_argument("--gt", required=True, help="GT optical-scan mp4.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--height", type=int, default=256,
                    help="Common frame height for scoring.")
    ap.add_argument("--gt-crop", type=parse_crop, default=None,
                    help="Crop GT picture area: x0,y0,x1,y1 fractions.")
    ap.add_argument("--pred-crop", type=parse_crop, default=None)
    ap.add_argument("--pred-rot90", type=int, default=0,
                    help="Rotate pred k*90 deg CCW to match GT orientation.")
    ap.add_argument("--pred-fliplr", action="store_true",
                    help="Mirror pred left-right to match GT.")
    ap.add_argument("--temporal", choices=["dtw", "linear"], default="dtw",
                    help="Frame correspondence: DTW warp or proportional 1:1.")
    ap.add_argument("--register", choices=["gradphase", "ecc", "phasecorr", "none"],
                    default="gradphase",
                    help="Spatial registration: gradient-domain phase-corr "
                         "translation (gradphase, fast+robust), gradient-LNCC "
                         "similarity (ecc, slow), raw FFT (phasecorr), or none.")
    ap.add_argument("--temporal-search", type=int, default=4,
                    help="Refine each match within +/-N GT frames by best "
                         "gradient-correlation (absorbs sync jitter). 0 = off.")
    ap.add_argument("--no-leader-trim", action="store_true")
    ap.add_argument("--fid", action="store_true",
                    help="Also compute set-level FID (needs pyiqa).")
    ap.add_argument("--export-aligned", action="store_true",
                    help="Write aligned_pred.mp4 + aligned_gt.mp4 for VMAF "
                         "(ffmpeg-quality-metrics) on the aligned pair.")
    ap.add_argument("--learned", action="store_true",
                    help="Also compute DISTS/LPIPS + BRISQUE/NIQE (needs pyiqa).")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Stage 0
    print("Loading videos...", flush=True)
    pred = load_video_luma(args.pred)
    gt = load_video_luma(args.gt)
    n_drop = 0
    if not args.no_leader_trim:
        gt, n_drop = trim_leader(gt)
    print(f"  pred {pred.shape}  gt {gt.shape} (dropped {n_drop} leader frames)")
    pred = apply_crop(pred, args.pred_crop)
    pred = apply_orientation(pred, args.pred_rot90, args.pred_fliplr)
    gt = apply_crop(gt, args.gt_crop)
    pred_l = resize_to_height(pred, args.height)
    gt_l = resize_to_height(gt, args.height)

    # Stage 1
    print(f"Temporal alignment ({args.temporal})...", flush=True)
    path = None
    if args.temporal == "dtw":
        path, cost = dtw_align(pred_l, gt_l)
        p2g = pred_to_gt_map(path, len(pred_l))
    else:
        p2g = linear_map(len(pred_l), len(gt_l))
    if args.temporal_search > 0:
        print(f"  refining matches within +/-{args.temporal_search} frames...",
              flush=True)
        p2g = refine_temporal(pred_l, gt_l, p2g, args.temporal_search)
    matched = [(ip, jg) for ip, jg in enumerate(p2g) if jg >= 0]
    print(f"  {len(matched)} matched frame pairs")

    # Stages 2 + 3
    print("Registering + scoring pairs...", flush=True)
    per_pair, triplets = [], []
    if args.learned and not M.HAS_PYIQA:
        print("  WARNING: --learned set but pyiqa unavailable; skipping "
              "DISTS/LPIPS/BRISQUE/NIQE.")
    do_learned = args.learned and M.HAS_PYIQA
    for ip, jg in matched:
        a, b, rec_extra = register_pair(gt_l[jg], pred_l[ip], args.register)
        bm = histogram_match(b, a)
        rec = M.fr_pair_metrics(a, bm, with_learned=do_learned, device=args.device)
        rec.update(rec_extra)
        if do_learned:
            # NR metrics on the actual (un-warped) pred frame, not the
            # edge-wrapped registered copy.
            for k, v in M.nr_metrics(pred_l[ip], args.device).items():
                rec[f"{k}_pred"] = v
        per_pair.append(rec)
        triplets.append((bm, a, np.abs(bm - a)))

    agg = aggregate(per_pair)
    if args.fid:
        if M.HAS_PYIQA:
            print("Computing FID (set-level)...", flush=True)
            agg["fid"] = compute_fid(triplets, args.device, args.out_dir)
        else:
            print("  WARNING: --fid needs pyiqa; skipping.")
    if args.export_aligned:
        print("Exporting aligned_pred.mp4 + aligned_gt.mp4...", flush=True)
        export_aligned_videos(triplets, args.out_dir)
    result = {
        "pred": os.path.abspath(args.pred),
        "gt": os.path.abspath(args.gt),
        "n_pred_frames": len(pred_l),
        "n_gt_frames": len(gt_l),
        "gt_leader_dropped": n_drop,
        "n_matched_pairs": len(matched),
        "height": args.height,
        "learned_enabled": do_learned,
        "aggregate": agg,
        "per_pair": per_pair,
    }
    with open(os.path.join(args.out_dir, "scores.json"), "w") as f:
        json.dump(result, f, indent=2)

    save_montage(triplets, os.path.join(args.out_dir, "aligned_montage.png"))
    if path is not None:
        _save_path_plot(path, cost.shape, os.path.join(args.out_dir, "dtw_path.png"))

    print("\n=== aggregate ===")
    for k in sorted(agg):
        print(f"  {k:18s} {agg[k]:.4f}")
    print(f"\nWrote {args.out_dir}/scores.json (+ montage, dtw_path)")


def _save_path_plot(path, shape, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = np.array(path)
    plt.figure(figsize=(5, 5))
    plt.plot(p[:, 1], p[:, 0], "-", lw=1)
    plt.plot([0, shape[1]], [0, shape[0]], "k--", lw=0.5, label="1:1")
    plt.xlabel("GT frame")
    plt.ylabel("pred frame")
    plt.title("DTW temporal path")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
