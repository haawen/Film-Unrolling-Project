#!/bin/bash
# =============================================================================
# Phase 1 smoke — same recipe as slurm_phase1.sh but 2000 steps on a100-hourly.
# Use for go/no-go on a variant before committing to a full 10k run.
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --job-name=phase1_sm
#SBATCH --output=logs/phase1_smoke_%x_%j.out
#SBATCH --error=logs/phase1_smoke_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=160G
#SBATCH --partition=a100-hourly

set -euo pipefail

VARIANT="${1:?usage: sbatch slurm_phase1_smoke.sh <variant> <preset>}"
PRESET="${2:?usage: sbatch slurm_phase1_smoke.sh <variant> <preset>}"

PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
SYN_DIR="${PROJECT_DIR}/unwrapping/synthetic/results/${PRESET}"
OUT_BASE="${PROJECT_DIR}/unwrapping/inr/results/phase1_smoke_${PRESET}/${VARIANT}"
TRAIN_DIR="${OUT_BASE}/train"
EVAL_DIR="${OUT_BASE}/eval_synthetic"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
mkdir -p logs "${TRAIN_DIR}" "${EVAL_DIR}"

EXTRA_FLAGS=()
# Prior 1.0 (vs default 1e-4) keeps θ_offset bounded; attachment is rotation-
# invariant about the spool so without a strong prior the optimizer drifts
# θ_offset freely (167513 ran |θ|_mean=0.076 rad ≈ 110px shift at outer winding).
case "${VARIANT}" in
  P1_A4)
    EXTRA_FLAGS+=(--per-winding-theta-phase --w-theta-phase-prior 1.0) ;;
  P1_A4_B3)
    EXTRA_FLAGS+=(--per-winding-theta-phase --w-theta-phase-prior 1.0)
    EXTRA_FLAGS+=(--w-radial-cap 1.0) ;;
  P1_A4_A5)
    EXTRA_FLAGS+=(--per-winding-theta-phase --w-theta-phase-prior 1.0)
    EXTRA_FLAGS+=(--two-stage-base-steps 400) ;;
  P1_full)
    EXTRA_FLAGS+=(--per-winding-theta-phase --w-theta-phase-prior 1.0)
    EXTRA_FLAGS+=(--two-stage-base-steps 400)
    EXTRA_FLAGS+=(--w-radial-cap 1.0) ;;
  P1_B3)
    EXTRA_FLAGS+=(--w-radial-cap 1.0) ;;
  *)
    echo "Unknown variant ${VARIANT}"; exit 1 ;;
esac

STEPS=2000
RAMP=500

echo "=== phase1 smoke ${VARIANT}/${PRESET} (steps=${STEPS}) start $(date) ==="
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
    --ecc-lr-mult 100.0 \
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

echo "=== phase1 smoke ${VARIANT}/${PRESET} complete at $(date) ==="
