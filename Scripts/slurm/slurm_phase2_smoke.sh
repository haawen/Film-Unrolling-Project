#!/bin/bash
# =============================================================================
# Phase 2 smoke — A1 (frame-period periodicity) variants on HQ presets.
#
#   P2_A1         baseline + A1 at w=1.0
#   P2_A1_w10     w=10 (probe larger weight)
#   P2_A1_w0_3    w=0.3 (probe smaller weight)
#   P2_A1_late    A1 ramped in after 500 steps (avoid early-step degeneracy)
#
# All variants share R1n pure-SS recipe (sup_steps=0, distance-attach, σ=20,
# lr=1e-4, conformal=0.1) and pixels-per-winding=256 for fast eval.
#
# Usage:
#   sbatch Scripts/slurm/slurm_phase2_smoke.sh P2_A1 hq_video_4k_outer_imperfect
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --job-name=phase2_sm
#SBATCH --output=logs/phase2_smoke_%x_%j.out
#SBATCH --error=logs/phase2_smoke_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=160G
#SBATCH --partition=a100-hourly

set -euo pipefail

VARIANT="${1:?usage: sbatch slurm_phase2_smoke.sh <variant> <preset>}"
PRESET="${2:?usage: sbatch slurm_phase2_smoke.sh <variant> <preset>}"

PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
SYN_DIR="${PROJECT_DIR}/unwrapping/synthetic/results/${PRESET}"
OUT_BASE="${PROJECT_DIR}/unwrapping/inr/results/phase2_smoke_${PRESET}/${VARIANT}"
TRAIN_DIR="${OUT_BASE}/train"
EVAL_DIR="${OUT_BASE}/eval_synthetic"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
mkdir -p logs "${TRAIN_DIR}" "${EVAL_DIR}"

# Frame-period u-units, derived from generator:
#   frame_w_px = max(200, n_z * 4 // 3) = 341 for n_z=256
#   total_arc ≈ 143558, n_layers=28
#   frame_period_u = 341/143558*28 ≈ 0.0665
FRAME_PERIOD_U=0.0665

EXTRA_FLAGS=()
case "${VARIANT}" in
  P2_A1)
    EXTRA_FLAGS+=(--w-frame-periodicity 1.0 --frame-period-u "${FRAME_PERIOD_U}") ;;
  P2_A1_w10)
    EXTRA_FLAGS+=(--w-frame-periodicity 10.0 --frame-period-u "${FRAME_PERIOD_U}") ;;
  P2_A1_w0_3)
    EXTRA_FLAGS+=(--w-frame-periodicity 0.3 --frame-period-u "${FRAME_PERIOD_U}") ;;
  *)
    echo "Unknown variant ${VARIANT}"
    echo "Available: P2_A1, P2_A1_w10, P2_A1_w0_3"
    exit 1 ;;
esac

STEPS=2000
RAMP=500

echo "=== phase2 smoke ${VARIANT}/${PRESET} (steps=${STEPS}) start $(date) ==="
echo "    extra: ${EXTRA_FLAGS[*]}"
cd "${PROJECT_DIR}"

srun python -u -m unwrapping.inr.surface_train \
    --data-dir  "${SYN_DIR}" \
    --out-dir   "${TRAIN_DIR}" \
    --attachment emulsion \
    --centerline-erode 1 \
    --supervised-steps 0 \
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
    --patch-feature-n-patches 64 \
    --patch-feature-w 64 \
    --patch-feature-n-dct 16 \
    --patch-feature-target-norm 0.05 \
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

echo "=== phase2 smoke ${VARIANT}/${PRESET} complete at $(date) ==="
