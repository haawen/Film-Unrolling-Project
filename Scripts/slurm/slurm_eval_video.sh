#!/bin/bash
#SBATCH --job-name=eval_video
#SBATCH --clusters=gmerlin7
#SBATCH --partition=a100-hourly
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/data/user/li_k1/M_thesis/unwrapping/eval/results/eval_video_%j.out

# Film-video metric eval (unrolled vs GT optical scan), GPU node.
# NEVER run this compute on a login node (strictly forbidden) -- always sbatch.
# All pyiqa/CLIP weights are pre-cached under ~/.cache/torch/hub, so the compute
# node runs fully offline.

set -e
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd /data/user/li_k1/M_thesis

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

PRED=${PRED:-unwrapping/inr/results/arc_simple/wholeroll_v2/film_final_p902.mp4}
GT=${GT:-data/h265_1080p.mp4}
OUT=${OUT:-unwrapping/eval/results/p902_vs_gt_learned}
GT_CROP=${GT_CROP:-0.13,0.07,0.80,0.93}
# PRED_FLAGS: orientation/crop of the pred video. Default = legacy strip
# renders (rotated+mirrored, margin crop). For frame_match's stabilized
# videos (already GT-oriented, picture-only, zoom-matched) pass PRED_FLAGS="".
PRED_FLAGS=${PRED_FLAGS---pred-crop 0.0,0.12,1.0,0.98 --pred-rot90 1 --pred-fliplr}

python -m unwrapping.eval.compare_videos \
    --pred "$PRED" --gt "$GT" --out-dir "$OUT" \
    --height 256 \
    --gt-crop "$GT_CROP" $PRED_FLAGS \
    --temporal linear --register gradphase --temporal-search 4 \
    --learned --fid --export-aligned --device cuda

echo "DONE -> $OUT/scores.json"

# VMAF on the ALIGNED pair (needs a libvmaf-enabled ffmpeg, e.g. conda env
# 'fftools'). Skipped cleanly if that ffmpeg isn't found.
FFMPEG=$(conda run -n fftools which ffmpeg 2>/dev/null || true)
if [ -n "$FFMPEG" ] && "$FFMPEG" -hide_banner -filters 2>/dev/null | grep -q " libvmaf "; then
    echo "=== VMAF (aligned) ==="
    "$FFMPEG" -hide_banner -i "$OUT/aligned_pred.mp4" -i "$OUT/aligned_gt.mp4" \
        -lavfi "libvmaf=log_path=$OUT/vmaf.json:log_fmt=json" -f null - 2>&1 | tail -3
    echo "VMAF -> $OUT/vmaf.json"
else
    echo "VMAF skipped: no libvmaf ffmpeg (create env 'fftools' with conda-forge ffmpeg)."
fi
