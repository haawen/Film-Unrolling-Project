#!/bin/bash
# =============================================================================
# Full training + eval on the HQ video presets (256 z-slices).
#
# No MAX_SLICES — trains and evaluates on all 256 slices.
# Uses a100-daily (24h) partition.
#
# Usage:
#   sbatch Scripts/slurm/slurm_R1_hq_full.sh R1S     hq_video_4k
#   sbatch Scripts/slurm/slurm_R1_hq_full.sh R1S     hq_video_4k_outer_imperfect
#   sbatch Scripts/slurm/slurm_R1_hq_full.sh R1n     hq_video_4k_outer_imperfect
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --job-name=hq_full
#SBATCH --output=logs/hq_full_%x_%j.out
#SBATCH --error=logs/hq_full_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=160G
#SBATCH --partition=a100-daily

set -euo pipefail

VARIANT="${1:?usage: sbatch slurm_R1_hq_full.sh <variant> <preset>}"
PRESET="${2:?usage: sbatch slurm_R1_hq_full.sh <variant> <preset>}"

PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
SYN_DIR="${PROJECT_DIR}/unwrapping/synthetic/results/${PRESET}"
OUT_BASE="${PROJECT_DIR}/unwrapping/inr/results/R1_hq_full_${PRESET}/${VARIANT}"
TRAIN_DIR="${OUT_BASE}/train"
EVAL_DIR="${OUT_BASE}/eval_synthetic"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
mkdir -p logs "${TRAIN_DIR}" "${EVAL_DIR}"

# ── Per-variant flags ───────────────────────────────────────────────────────
case "${VARIANT}" in
  R1S)  ATTACH=emulsion; SUP_STEPS=10000; LR="1e-4"; W_CONF=0.1; USE_DIST=1 ;;
  R1n)  ATTACH=emulsion; SUP_STEPS=0;     LR="1e-4"; W_CONF=0.1; USE_DIST=1 ;;
  *)    echo "Unknown variant ${VARIANT}"; exit 1 ;;
esac

STEPS=10000
RAMP=2000
BATCH=131072

echo "============================================================"
echo "  HQ full training: ${VARIANT} / ${PRESET}"
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
    --log-every 100 \
    --ckpt-every 2000

# ── Eval ────────────────────────────────────────────────────────────────────
srun python -u -m unwrapping.inr.surface_eval \
    --data-dir  "${SYN_DIR}" \
    --out-dir   "${EVAL_DIR}" \
    --attachment "${ATTACH}" \
    --centerline-erode 1 \
    --ckpt   "${TRAIN_DIR}/model_final.pt" \
    --gt-npz "${SYN_DIR}/ground_truth.npz"

echo "=== hq_full ${VARIANT}/${PRESET} complete at $(date) ==="
