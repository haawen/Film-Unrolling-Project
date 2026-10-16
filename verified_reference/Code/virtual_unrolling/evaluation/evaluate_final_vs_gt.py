"""GT-fidelity scores of the final full-width video (thesis Table "fw-fidelity").

Scores ``film_affine.mp4`` (output of ``frame_match --export-video`` followed by
``stabilize_affine --mode affine``) against the optical ground-truth scan, frame
by frame, with the GT prepared EXACTLY the way ``stabilize_affine`` prepared it
when it fitted the per-frame affine warp:

    GT  : leader trim -> crop 0.235,0.12,0.81,0.88 -> resize to the stabiliser's
          input size (720 x 822) -> the same 4 % safe-interior crop
          ``stabilize_affine --extra-crop 0.04`` applies to its output -> resize
          to the output size.  Frame pairing = ``linspace`` (identity here, 233:233).
    pred: ``film_affine.mp4`` as is.

Then, like the original evaluation: drop the last 5 frames (the GT reel tail is a
splice + blank frames), score every pair at height 256 with NO further spatial
registration (alignment is established by construction; phase correlation locks
onto the repeating background of this footage), per-pair histogram matching of
pred to GT (as ``compare_videos``), and finally exclude GT frame 196 (a physical
tear on the optical reel itself) -> 227 pairs.

This reconstructs the evaluation that produced the reported numbers
(SSIM 0.776, MS-SSIM 0.857, grad-corr 0.788, NMI 1.145, PSNR 21.90 dB,
MSAD 0.053; LPIPS 0.143, DISTS 0.125). The original ad-hoc builder script was not
preserved; this script was re-derived from its documented recipe and checked
against the stored scores (see the deliverable README).

IMPORTANT: ``film_affine.mp4`` is fitted to the GT per frame, so these scores
measure best-aligned content agreement, not raw geometric accuracy.

Usage
-----
    python -m virtual_unrolling.evaluation.evaluate_final_vs_gt \
        --pred Results/.../film_affine.mp4 --gt Data/4_optical_ground_truth/optical_scan_h265_1080p.mp4 \
        --out-dir <dir> [--learned]
"""
import argparse
import json
import os
import tempfile

import imageio
import numpy as np
from PIL import Image

from . import metrics as M
from .video_utils import (aggregate, apply_crop, common_crop, histogram_match,
                          load_video_luma, parse_crop, resize_to_height, save_montage,
                          trim_leader)


def _resize(f, h, w):
    """Same uint8-quantising resize as stabilize_horizontal._resize."""
    return np.asarray(Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8)
                                      ).resize((w, h))).astype(np.float32) / 255


def _write(frames, path, fps=26):
    wr = imageio.get_writer(path, fps=fps, codec="libx264", quality=9,
                            macro_block_size=1)
    for f in frames:
        wr.append_data((np.clip(f, 0, 1) * 255).astype(np.uint8))
    wr.close()


def build_matched(pred_path, gt_path, gt_crop, in_hw, extra_crop, tail_drop):
    pred = load_video_luma(pred_path)
    n, H, W = pred.shape
    gt = load_video_luma(gt_path)
    gt, n_lead = trim_leader(gt)
    gt = apply_crop(gt, parse_crop(gt_crop))
    hi, wi = in_hw
    gt = [_resize(g, hi, wi) for g in gt]
    y0, y1 = int(hi * extra_crop), int(hi * (1 - extra_crop))
    x0, x1 = int(wi * extra_crop), int(wi * (1 - extra_crop))
    gt = [_resize(g[y0:y1, x0:x1], H, W) for g in gt]
    idx = np.linspace(0, len(gt) - 1, n).round().astype(int)
    gt_m = np.stack([gt[j] for j in idx])
    keep = n - tail_drop
    return pred[:keep], gt_m[:keep], n_lead, n, len(gt)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True, help="film_affine.mp4")
    ap.add_argument("--gt", required=True, help="raw optical GT scan mp4")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gt-crop", default="0.235,0.12,0.81,0.88")
    ap.add_argument("--stab-input-hw", default="720,822",
                    help="frame size of the video fed to stabilize_affine "
                         "(film_stabilized_gtsync.mp4)")
    ap.add_argument("--extra-crop", type=float, default=0.04)
    ap.add_argument("--tail-drop", type=int, default=5,
                    help="frames dropped at the end (GT splice + blank frames)")
    ap.add_argument("--exclude", default="196",
                    help="comma list of pair indices to exclude (GT tear)")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--no-via-mp4", action="store_true",
                    help="score in memory instead of writing/re-reading the matched "
                         "pair as mp4 (the original evaluation went through mp4)")
    ap.add_argument("--learned", action="store_true",
                    help="also LPIPS + DISTS (needs pyiqa)")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    hw = tuple(int(v) for v in args.stab_input_hw.split(","))
    pred, gt, n_lead, n_pred, n_gt = build_matched(
        args.pred, args.gt, args.gt_crop, hw, args.extra_crop, args.tail_drop)
    print(f"pred {n_pred} frames, GT {n_gt} after dropping {n_lead} leader frames; "
          f"{len(pred)} pairs after tail drop", flush=True)

    if not args.no_via_mp4:
        with tempfile.TemporaryDirectory() as td:
            pp, gp = os.path.join(td, "pred_matched.mp4"), os.path.join(td, "gt_matched.mp4")
            _write(pred, pp); _write(gt, gp)
            pred, gt = load_video_luma(pp), load_video_luma(gp)
    else:
        def _norm(a):
            a = a - a.min(); return a / a.max()
        pred, gt = _norm(pred), _norm(gt)

    pred_l = resize_to_height(pred, args.height)
    gt_l = resize_to_height(gt, args.height)
    excl = {int(v) for v in args.exclude.split(",") if v.strip()}
    do_learned = args.learned and M.HAS_PYIQA
    if args.learned and not M.HAS_PYIQA:
        print("WARNING: --learned needs pyiqa; skipping LPIPS/DISTS")
    per_pair, triplets = [], []
    for i in range(len(pred_l)):
        a, b = common_crop(gt_l[i], pred_l[i])          # no spatial registration
        bm = histogram_match(b, a)
        rec = M.fr_pair_metrics(a, bm)
        if do_learned:
            rec["lpips"] = M.lpips(a, bm, args.device)
            rec["dists"] = M.dists(a, bm, args.device)
        rec["pair"] = i
        per_pair.append(rec)
        if i not in excl:
            triplets.append((bm, a, np.abs(bm - a)))
    kept = [{k: v for k, v in r.items() if k != "pair"} for r in per_pair
            if r["pair"] not in excl]
    agg_all = aggregate([{k: v for k, v in r.items() if k != "pair"} for r in per_pair])
    agg = aggregate(kept)
    out = {
        "pred": os.path.abspath(args.pred), "gt": os.path.abspath(args.gt),
        "protocol": {
            "gt_crop": args.gt_crop, "stabilizer_input_hw": hw,
            "extra_crop": args.extra_crop, "tail_drop": args.tail_drop,
            "excluded_pairs": sorted(excl), "height": args.height,
            "registration": "none", "intensity": "per-pair histogram matching (pred->GT)",
            "via_mp4": not args.no_via_mp4,
        },
        "n_pairs_scored": len(per_pair), "n_pairs_kept": len(kept),
        "aggregate_kept": agg, "aggregate_all_scored": agg_all, "per_pair": per_pair,
    }
    with open(os.path.join(args.out_dir, "scores.json"), "w") as f:
        json.dump(out, f, indent=2)
    save_montage(triplets, os.path.join(args.out_dir, "aligned_montage.png"))
    print(f"\n=== {len(kept)} pairs (excluded {sorted(excl)}) ===")
    for k in sorted(agg):
        if k.endswith("_mean"):
            print(f"  {k:16s} {agg[k]:.4f}")


if __name__ == "__main__":
    main()
