"""Compare several unrolling variants with one shared affine protocol.

Every prediction is processed in exactly the same way:

1. crop and orient the reconstructed frames;
2. match each reconstructed frame to an optical ground-truth frame;
3. estimate a full 2-D affine transform from image gradients;
4. crop the same safe interior region and histogram-match intensities; and
5. compute full-reference metrics on the resulting pair.

The affine fit may translate, rotate, scale, and shear the prediction.  It is
used only for evaluation.  It therefore removes global pose differences but
does not correct local, non-rigid reconstruction errors.

PSNR is calculated from the aggregate pixel MSE, so the two reported values
are mathematically consistent: PSNR = 10 log10(1 / MSE) for images in [0, 1].
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from . import metrics as M
from .compare_videos import (
    apply_crop,
    apply_orientation,
    common_crop,
    grad_mag,
    histogram_match,
    linear_map,
    load_video_luma,
    parse_crop,
    refine_temporal,
    register_pair,
    resize_to_height,
    trim_leader,
)
from .stabilize_affine import affine_field, grid_disp, warp


def parse_variant(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("variant must be NAME=VIDEO")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("variant must be NAME=VIDEO")
    return name.strip(), path.strip()


def affine_align(gt: np.ndarray, pred: np.ndarray, grid: int, iters: int):
    """Warp *pred* onto *gt* with an iteratively estimated full affine field."""
    # First remove the potentially large translation.  The block-wise affine
    # estimator is intentionally local and is reliable only once corresponding
    # image regions overlap.
    gt, pred, _ = register_pair(gt, pred, "gradphase")
    gt, pred = common_crop(gt, pred)
    h, w = gt.shape

    def estimate(frame):
        points, disp, _ = grid_disp(grad_mag(frame), grad_mag(gt), grid)
        return affine_field(points, disp, h, w)

    # The displacement convention depends on the sampler.  Resolve it from
    # the first affine estimate by retaining the sign with better edge match.
    dy0, dx0 = estimate(pred)
    plus = warp(pred, dy0, dx0, 1.0)
    minus = warp(pred, dy0, dx0, -1.0)
    score_plus = M.gradient_correlation(gt, plus)
    score_minus = M.gradient_correlation(gt, minus)
    sign = 1.0 if score_plus >= score_minus else -1.0

    total_dy = np.zeros_like(gt, dtype=np.float64)
    total_dx = np.zeros_like(gt, dtype=np.float64)
    current = pred
    for _ in range(iters):
        dy, dx = estimate(current)
        total_dy += dy
        total_dx += dx
        current = warp(pred, total_dy, total_dx, sign)
    return gt, current


def crop_interior(a: np.ndarray, b: np.ndarray, fraction: float):
    h, w = a.shape
    y0, y1 = int(round(h * fraction)), int(round(h * (1.0 - fraction)))
    x0, x1 = int(round(w * fraction)), int(round(w * (1.0 - fraction)))
    return a[y0:y1, x0:x1], b[y0:y1, x0:x1]


def evaluate_variant(name, path, gt_l, args):
    print(f"\n[{name}] loading {path}", flush=True)
    pred = load_video_luma(path)
    pred = apply_crop(pred, args.pred_crop)
    pred = apply_orientation(pred, args.pred_rot90, args.pred_fliplr)
    pred_l = resize_to_height(pred, args.height)

    mapping = linear_map(len(pred_l), len(gt_l))
    if args.temporal_search:
        print(f"[{name}] refining temporal matches +/-{args.temporal_search}",
              flush=True)
        mapping = refine_temporal(pred_l, gt_l, mapping, args.temporal_search)

    per_frame = []
    aligned_pred = []
    aligned_gt = []
    for i, j in enumerate(mapping):
        a, b = affine_align(gt_l[j], pred_l[i], args.grid, args.iters)
        a, b = crop_interior(a, b, args.extra_crop)
        b = histogram_match(b, a)

        mse = float(np.mean((a - b) ** 2))
        per_frame.append({
            "pred_frame": i,
            "gt_frame": int(j),
            "gradient_correlation": M.gradient_correlation(a, b),
            "ms_ssim": M.ms_ssim(a, b),
            "mse": mse,
        })
        aligned_pred.append(b.astype(np.float32))
        aligned_gt.append(a.astype(np.float32))
        if i % 25 == 0:
            print(f"[{name}] affine + classical metrics {i}/{len(mapping)}",
                  flush=True)

    # MSE is aggregated over every evaluated pixel.  All frames have the same
    # size, so this is also the arithmetic mean of the per-frame MSE values.
    mse = float(np.mean([row["mse"] for row in per_frame]))
    aggregate = {
        "gradient_correlation": float(np.mean(
            [row["gradient_correlation"] for row in per_frame])),
        "ms_ssim": float(np.mean([row["ms_ssim"] for row in per_frame])),
        "mse": mse,
        "psnr_db": float(10.0 * math.log10(1.0 / mse)),
    }

    if args.lpips:
        if not M.HAS_PYIQA:
            raise RuntimeError("--lpips requested, but pyiqa is unavailable")
        print(f"[{name}] LPIPS on {len(aligned_pred)} affine pairs", flush=True)
        metric = M._pyiqa_metric("lpips", args.device)
        values = []
        import torch
        with torch.no_grad():
            for start in range(0, len(aligned_pred), args.batch_size):
                pa = np.stack(aligned_pred[start:start + args.batch_size])
                ga = np.stack(aligned_gt[start:start + args.batch_size])
                pt = torch.from_numpy(pa)[:, None].repeat(1, 3, 1, 1)
                gt = torch.from_numpy(ga)[:, None].repeat(1, 3, 1, 1)
                out = metric(pt.to(args.device), gt.to(args.device))
                vals = np.asarray(out.detach().cpu()).reshape(-1)
                # Some pyiqa versions reduce a batch to one value.  Fall back
                # to individual calls so every frame receives equal weight.
                if len(vals) == len(pa):
                    values.extend(float(v) for v in vals)
                else:
                    for p_one, g_one in zip(pt, gt):
                        v = metric(p_one[None].to(args.device),
                                   g_one[None].to(args.device))
                        values.append(float(v.item()))
                print(f"[{name}] LPIPS {min(start + args.batch_size, len(pa) + start)}/"
                      f"{len(aligned_pred)}", flush=True)
        aggregate["lpips"] = float(np.mean(values))

    return {
        "name": name,
        "video": os.path.abspath(path),
        "n_prediction_frames": len(pred_l),
        "n_pairs": len(per_frame),
        "aggregate": aggregate,
        "per_frame": per_frame,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", action="append", type=parse_variant,
                        required=True, metavar="NAME=VIDEO")
    parser.add_argument("--gt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--gt-crop", type=parse_crop,
                        default=parse_crop("0.235,0.12,0.81,0.88"))
    parser.add_argument("--pred-crop", type=parse_crop,
                        default=parse_crop("0.0,0.12,1.0,0.98"))
    parser.add_argument("--pred-rot90", type=int, default=1)
    parser.add_argument("--pred-fliplr", action="store_true", default=True)
    parser.add_argument("--temporal-search", type=int, default=4)
    parser.add_argument("--grid", type=int, default=5)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--extra-crop", type=float, default=0.04)
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    print("Loading and preparing optical ground truth...", flush=True)
    gt = load_video_luma(args.gt)
    gt, dropped = trim_leader(gt)
    gt = apply_crop(gt, args.gt_crop)
    gt_l = resize_to_height(gt, args.height)
    print(f"GT: {len(gt_l)} frames after dropping {dropped} leader frames",
          flush=True)

    result = {
        "protocol": {
            "description": "Common per-frame GT-assisted full affine evaluation",
            "affine_degrees_of_freedom": [
                "translation", "rotation", "scale", "shear"
            ],
            "local_non_rigid_warp": False,
            "gt_crop": args.gt_crop,
            "pred_crop": args.pred_crop,
            "pred_rot90_ccw": args.pred_rot90,
            "pred_horizontal_flip": args.pred_fliplr,
            "height": args.height,
            "temporal_mapping": "linear plus gradient-correlation search",
            "temporal_search_radius_frames": args.temporal_search,
            "affine_grid": args.grid,
            "affine_iterations": args.iters,
            "post_warp_interior_crop_fraction": args.extra_crop,
            "intensity_normalization": "per-pair histogram matching",
            "psnr_definition": "10*log10(1/MSE), using aggregate direct MSE",
            "gt_use": "evaluation only",
        },
        "gt": os.path.abspath(args.gt),
        "gt_frames_after_trim": len(gt_l),
        "variants": [],
    }
    for name, path in args.variant:
        result["variants"].append(evaluate_variant(name, path, gt_l, args))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\nAffine comparison", flush=True)
    for row in result["variants"]:
        a = row["aggregate"]
        print(name if False else row["name"],
              " ".join(f"{k}={v:.6f}" for k, v in a.items()), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
