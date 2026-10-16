#!/bin/bash
# =============================================================================
# Phase 3 smoke — three novel approaches for the SS/sup gap on HQ imperfect.
#
#   P3_Z1         MAPS (medial-axis pseudo-supervision) at w=0.1
#   P3_Z1_w1     MAPS at w=1.0 (probe stronger weight)
#   P3_Z2         mask-weighted intensity z-coh at w=1.0
#   P3_Z3        supervised + 5px label noise (Z3 diagnostic, R1S variant)
#
# All SS variants share the R1n recipe (sup_steps=0, distance-attach, σ=20,
# lr=1e-4, conformal=0.1) + --pixels-per-winding 256 for fast eval.
# Z3 is a SUPERVISED variant (sup_steps=10000, GT u-map + label noise).
#
# Usage:
#   sbatch Scripts/slurm/slurm_phase3_smoke.sh P3_Z1 hq_video_4k_outer_imperfect
#   sbatch Scripts/slurm/slurm_phase3_smoke.sh P3_Z2 hq_video_4k_outer_imperfect
#   sbatch Scripts/slurm/slurm_phase3_smoke.sh P3_Z3 hq_video_4k_outer_imperfect
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --job-name=phase3_sm
#SBATCH --output=logs/phase3_smoke_%x_%j.out
#SBATCH --error=logs/phase3_smoke_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=160G
#SBATCH --partition=a100-hourly

set -euo pipefail

VARIANT="${1:?usage: sbatch slurm_phase3_smoke.sh <variant> <preset>}"
PRESET="${2:?usage: sbatch slurm_phase3_smoke.sh <variant> <preset>}"

PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
SYN_DIR="${PROJECT_DIR}/unwrapping/synthetic/results/${PRESET}"
OUT_BASE="${PROJECT_DIR}/unwrapping/inr/results/phase3_smoke_${PRESET}/${VARIANT}"
TRAIN_DIR="${OUT_BASE}/train"
EVAL_DIR="${OUT_BASE}/eval_synthetic"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
mkdir -p logs "${TRAIN_DIR}" "${EVAL_DIR}"

STEPS=2000
RAMP=500
SUP_STEPS=0
EXTRA_FLAGS=()

case "${VARIANT}" in
  P3_Z1)
    # MAPS at w=0.1 (soft tie-breaker).
    EXTRA_FLAGS+=(--w-maps 0.1 --maps-batch-size 8192) ;;
  P3_Z1_w1)
    EXTRA_FLAGS+=(--w-maps 1.0 --maps-batch-size 8192) ;;
  P3_Z2)
    # Mask-weighted intensity z-coh.
    EXTRA_FLAGS+=(--w-intensity-zcoh 1.0 --mask-weight-int-zcoh) ;;
  P3_Z3)
    # Supervised with 5px label noise — diagnostic, not SS.
    SUP_STEPS=2000
    EXTRA_FLAGS+=(--label-noise-px 5.0) ;;
  *)
    echo "Unknown variant ${VARIANT}"
    echo "Available: P3_Z1, P3_Z1_w1, P3_Z2, P3_Z3"
    exit 1 ;;
esac

echo "=== phase3 smoke ${VARIANT}/${PRESET} (steps=${STEPS}, sup=${SUP_STEPS}) start $(date) ==="
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

echo "=== phase3 smoke ${VARIANT}/${PRESET} complete at $(date) ==="
