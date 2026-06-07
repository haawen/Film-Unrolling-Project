#!/bin/bash
# =============================================================================
# Phase 1 — analytical-base levers (A4 / A5 / B3) on HQ presets.
#
# All variants build on the R1n pure-SS recipe (sup_steps=0, distance-attach,
# σ=20). They differ only in which base-param levers are enabled:
#
#   P1_A4       per-winding θ_offset[k]                (A4 alone)
#   P1_A4_B3    A4 + radial-cap soft hinge             (A4 + B3)
#   P1_A4_A5    A4 + two-stage (base 1000 → INR)       (A4 + A5)
#   P1_full     A4 + A5 + B3                           (all three)
#   P1_B3       radial-cap only                        (B3 alone, control)
#
# Each compares against the R1n baseline at baselines_hq_outer/R1n_imperfect/.
#
# Usage:
#   sbatch Scripts/slurm/slurm_phase1.sh P1_A4        hq_video_4k_outer_imperfect
#   sbatch Scripts/slurm/slurm_phase1.sh P1_full      hq_video_4k_outer_imperfect
#   sbatch Scripts/slurm/slurm_phase1.sh P1_A4        hq_video_4k_outer            # sanity
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --job-name=phase1
#SBATCH --output=logs/phase1_%x_%j.out
#SBATCH --error=logs/phase1_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=160G
#SBATCH --partition=a100-daily

set -euo pipefail

VARIANT="${1:?usage: sbatch slurm_phase1.sh <variant> <preset>}"
PRESET="${2:?usage: sbatch slurm_phase1.sh <variant> <preset>}"

PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
SYN_DIR="${PROJECT_DIR}/unwrapping/synthetic/results/${PRESET}"
OUT_BASE="${PROJECT_DIR}/unwrapping/inr/results/phase1_${PRESET}/${VARIANT}"
TRAIN_DIR="${OUT_BASE}/train"
EVAL_DIR="${OUT_BASE}/eval_synthetic"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
mkdir -p logs "${TRAIN_DIR}" "${EVAL_DIR}"

# ── Per-variant flag bundles ────────────────────────────────────────────────
# Shared across all P1 variants: R1n pure-SS recipe, no GT.
EXTRA_FLAGS=()
case "${VARIANT}" in
  P1_A4)
    EXTRA_FLAGS+=(--per-winding-theta-phase --w-theta-phase-prior 1e-4)
    ;;
  P1_A4_B3)
    EXTRA_FLAGS+=(--per-winding-theta-phase --w-theta-phase-prior 1e-4)
    EXTRA_FLAGS+=(--w-radial-cap 1.0)
    ;;
  P1_A4_A5)
    EXTRA_FLAGS+=(--per-winding-theta-phase --w-theta-phase-prior 1e-4)
    EXTRA_FLAGS+=(--two-stage-base-steps 1000)
    ;;
  P1_full)
    EXTRA_FLAGS+=(--per-winding-theta-phase --w-theta-phase-prior 1e-4)
    EXTRA_FLAGS+=(--two-stage-base-steps 1000)
    EXTRA_FLAGS+=(--w-radial-cap 1.0)
    ;;
  P1_B3)
    EXTRA_FLAGS+=(--w-radial-cap 1.0)
    ;;
  *)
    echo "Unknown variant ${VARIANT}"
    echo "Available: P1_A4, P1_A4_B3, P1_A4_A5, P1_full, P1_B3"
    exit 1
    ;;
esac

ATTACH=emulsion
SUP_STEPS=0
LR="1e-4"
W_CONF=0.1
STEPS=10000
RAMP=2000
BATCH=131072

echo "============================================================"
echo "  Phase 1: ${VARIANT} / ${PRESET}"
echo "  extra  : ${EXTRA_FLAGS[*]}"
echo "  data   : ${SYN_DIR}"
echo "  out    : ${OUT_BASE}"
echo "  steps  : ${STEPS}  sup_steps=${SUP_STEPS}  lr=${LR}"
echo "  job    : ${SLURM_JOB_ID}"
echo "  start  : $(date)"
echo "============================================================"
cd "${PROJECT_DIR}"

# ── Train ───────────────────────────────────────────────────────────────────
srun python -u -m unwrapping.inr.surface_train \
    --data-dir  "${SYN_DIR}" \
    --out-dir   "${TRAIN_DIR}" \
    --attachment "${ATTACH}" \
    --centerline-erode 1 \
    --supervised-steps "${SUP_STEPS}" \
    --gt-npz "${SYN_DIR}/ground_truth.npz" \
    --steps "${STEPS}" \
    --ramp-steps "${RAMP}" \
    --batch-size "${BATCH}" \
    --lr "${LR}" \
    --w-attach 1.0 \
    --w-conformal "${W_CONF}" \
    --w-res-reg 0.0 \
    --sigma 20.0 \
    --hidden-dim 256 \
    --n-layers-mlp 4 \
    --model-type inr \
    --use-distance-attach \
    --ecc-lr-mult 100.0 \
    --log-every 100 \
    --ckpt-every 2000 \
    "${EXTRA_FLAGS[@]}"

# ── Eval ────────────────────────────────────────────────────────────────────
srun python -u -m unwrapping.inr.surface_eval \
    --data-dir  "${SYN_DIR}" \
    --out-dir   "${EVAL_DIR}" \
    --attachment "${ATTACH}" \
    --centerline-erode 1 \
    --ckpt   "${TRAIN_DIR}/model_final.pt" \
    --gt-npz "${SYN_DIR}/ground_truth.npz"

echo "=== phase1 ${VARIANT}/${PRESET} complete at $(date) ==="
