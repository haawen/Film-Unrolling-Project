#!/bin/bash
# STAGE 5 (final model) - videos and evaluation against the optical GT scan.
#
#   sbatch Code/slurm/eval_01_final_vs_gt.sh                         # shipped final strip
#   STRIP=$WORK/final_render/wholeroll.npy sbatch Code/slurm/eval_01_final_vs_gt.sh
#
# A. GT used for frame alignment:
#    frame_match.py   - every GT frame is paired with the same film cell, re-cut
#                       from the strip at the position that best matches it
#                       (coarse-to-fine search + cell-quantised DP + per-frame GT lock)
#                       -> film_stabilized_gtsync.mp4  (our pixels, GT-chosen cut)
#    stabilize_affine - one affine warp per frame fitted to the GT
#                       -> film_affine.mp4             (GT-fitted viewing copy)
#    evaluate_final_vs_gt - SSIM/MS-SSIM/grad-corr/NMI/PSNR/MSAD (+LPIPS/DISTS) on 227 pairs
#    Flags specific to this scan: --z-crop 0.203,0.80 (picture band = rows 480-1890),
#    --pitch 1130, --no-fliplr (this reconstruction is not mirrored; with the
#    default fliplr frame_match matches a mirrored film and still reports a score).
# B. GT NOT used: border-locked video (picture border tracked in the strip itself,
#    locked across the film; along the film the strip's own phase lock).
# C. Stillness arbiter (temporal-median sharpness, central 70%, common size).
# Output: $WORK/eval_final/
#
#SBATCH --job-name=eval_final
#SBATCH --partition=daily
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --output=work/logs/%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/Code/slurm/_activate.sh"
STRIP=${STRIP:-$THESIS_ROOT/../unwrapping/inr/results/newscan/render/wholeroll_tw4.npy}
OUT="$WORK/eval_final"; mkdir -p "$OUT"
E="python -u -m virtual_unrolling.evaluation"

echo "=== A1. frame matching against the GT (GT chooses each frame's cut) ==="
$E.frame_match --strip "$STRIP" --gt "$GT_VIDEO" --z-crop 0.203,0.80 --pitch 1130 \
    --no-fliplr --export-video --montage --video --fps 25 --out-dir "$OUT/frame_match"
echo "=== A2. per-frame affine warp fitted to the GT ==="
$E.stabilize_affine --in "$OUT/frame_match/film_stabilized_gtsync.mp4" --gt "$GT_VIDEO" \
    --out "$OUT/frame_match/film_affine.mp4"
echo "=== A3. GT-fidelity scores (227 pairs) ==="
$E.evaluate_final_vs_gt --pred "$OUT/frame_match/film_affine.mp4" --gt "$GT_VIDEO" \
    --out-dir "$OUT/gt_fidelity" --learned

echo "=== B. GT-free border-locked video ==="
$E.detect_border "$STRIP" --out-dir "$OUT/border" --pitch 1130
$E.border_render "$STRIP" --border "$OUT/border/border.npz" --out-dir "$OUT/border_lock" \
    --smooth-frames 0.8 --reverse --invert --height 720 \
    --along-shift 0.156 \
    --margin 0.02 --no-refine-pitch --video film_borderlock_noGT.mp4

echo "=== C. stillness arbiter ==="
PLAIN=${PLAIN_VIDEO:-$THESIS_ROOT/Results/0_requested_videos/3a_final_model_NO_gt__plain_render.mp4}
$E.border_stillness --videos "$PLAIN" "$OUT/border_lock/film_borderlock_noGT.mp4" \
    "$GT_PERFLOCK" \
    --labels plain_render borderlock_noGT GT_perflock \
    --margins 0.15 0.15 0.15 --size 720 720 --out-dir "$OUT/stillness"
echo "=== done: $OUT ==="
