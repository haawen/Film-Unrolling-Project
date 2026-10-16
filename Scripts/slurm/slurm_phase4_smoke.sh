#!/bin/bash
# =============================================================================
# Phase 4 smoke — A3 (joint autodecoder) on HQ imperfect.
#
#   P4_A3         w=1.0,  start=500,  ramp=1000,  strip_tv=0
#   P4_A3_tv      w=1.0,  start=500,  ramp=1000,  strip_tv=0.1   (smooth strip)
#   P4_A3_warm    w=1.0,  start=1000, ramp=500                   (later start)
#   P4_A3_w10     w=10.0, start=500,  ramp=1000                  (probe high w)
#
# Pure-SS recipe + co-learned strip INR. Output gets `strip_render.png` from
# eval for visual inspection.
#
# Usage:
#   sbatch Scripts/slurm/slurm_phase4_smoke.sh P4_A3 hq_video_4k_outer_imperfect
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --job-name=phase4_sm
#SBATCH --output=logs/phase4_smoke_%x_%j.out
#SBATCH --error=logs/phase4_smoke_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=160G
#SBATCH --partition=a100-hourly

set -euo pipefail

VARIANT="${1:?usage: sbatch slurm_phase4_smoke.sh <variant> <preset>}"
PRESET="${2:?usage: sbatch slurm_phase4_smoke.sh <variant> <preset>}"

PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
SYN_DIR="${PROJECT_DIR}/unwrapping/synthetic/results/${PRESET}"
OUT_BASE="${PROJECT_DIR}/unwrapping/inr/results/phase4_smoke_${PRESET}/${VARIANT}"
TRAIN_DIR="${OUT_BASE}/train"
EVAL_DIR="${OUT_BASE}/eval_synthetic"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
mkdir -p logs "${TRAIN_DIR}" "${EVAL_DIR}"

STEPS=3000
RAMP=500
SUP_STEPS=0
EXTRA_FLAGS=()

case "${VARIANT}" in
  P4_A3)
    EXTRA_FLAGS+=(--w-autodecoder 1.0 --autodecoder-start-step 500 --autodecoder-ramp-steps 1000) ;;
  P4_A3_tv)
    EXTRA_FLAGS+=(--w-autodecoder 1.0 --autodecoder-start-step 500 --autodecoder-ramp-steps 1000)
    EXTRA_FLAGS+=(--w-strip-tv 0.1) ;;
  P4_A3_warm)
    EXTRA_FLAGS+=(--w-autodecoder 1.0 --autodecoder-start-step 1000 --autodecoder-ramp-steps 500) ;;
  P4_A3_w10)
    EXTRA_FLAGS+=(--w-autodecoder 10.0 --autodecoder-start-step 500 --autodecoder-ramp-steps 1000) ;;
  P4_A3_B3)
    # A3 + radial cap: narrows the wrong-mapping basin so g_φ has less
    # off-analytical space to "fit" CT under.
    EXTRA_FLAGS+=(--w-autodecoder 1.0 --autodecoder-start-step 500 --autodecoder-ramp-steps 1000)
    EXTRA_FLAGS+=(--w-radial-cap 1.0) ;;
  P4_A3_tiny)
    # Capacity-limited strip: ~5k params instead of ~50k. Can't memorize
    # arbitrary CT-explaining strips.
    EXTRA_FLAGS+=(--w-autodecoder 1.0 --autodecoder-start-step 500 --autodecoder-ramp-steps 1000)
    EXTRA_FLAGS+=(--strip-n-fourier 32 --strip-hidden-dim 32 --strip-n-layers 2) ;;
  P4_A3_Z1)
    # A3 + soft MAPS anchor: per-(u,xy) pseudo-labels at low weight break
    # the joint degeneracy by anchoring f_θ. Z1 alone was neutral; combined
    # with A3 the anchor might carry the symmetry-breaking the strip needs.
    EXTRA_FLAGS+=(--w-autodecoder 1.0 --autodecoder-start-step 500 --autodecoder-ramp-steps 1000)
    EXTRA_FLAGS+=(--w-maps 0.1 --maps-batch-size 8192) ;;
  *)
    echo "Unknown variant ${VARIANT}"
    echo "Available: P4_A3, P4_A3_tv, P4_A3_warm, P4_A3_w10, P4_A3_B3, P4_A3_tiny, P4_A3_Z1"
    exit 1 ;;
esac

echo "=== phase4 smoke ${VARIANT}/${PRESET} (steps=${STEPS}) start $(date) ==="
echo "    extra: ${EXTRA_FLAGS[*]}"
cd "${PROJECT_DIR}"

srun python -u -m unwrapping.inr.surface_train \
    --data-dir  "${SYN_DIR}" \
    --out-dir   "${TRAIN_DIR}" \
    --attachment emulsion \
    --centerline-erode 1 \
    --supervised-steps "${SUP_STEPS}" \
    --gt-npz "${SYN_DIR}/ground_truth.npz" \
    --steps "${STEPS}" \
    --ramp-steps "${RAMP}" \
    --batch-size 131072 \
    --lr 1e-4 \
    --w-attach 1.0 \
    --w-conformal 0.1 \
    --w-res-reg 0.0 \
    --sigma 20.0 \
    --hidden-dim 256 \
    --n-layers-mlp 4 \
    --model-type inr \
    --use-distance-attach \
    --log-every 100 \
    --ckpt-every 1000 \
    "${EXTRA_FLAGS[@]}"

srun python -u -m unwrapping.inr.surface_eval \
    --data-dir  "${SYN_DIR}" \
    --out-dir   "${EVAL_DIR}" \
    --attachment emulsion \
    --centerline-erode 1 \
    --ckpt   "${TRAIN_DIR}/model_final.pt" \
    --gt-npz "${SYN_DIR}/ground_truth.npz" \
    --pixels-per-winding 256

echo "=== phase4 smoke ${VARIANT}/${PRESET} complete at $(date) ==="
