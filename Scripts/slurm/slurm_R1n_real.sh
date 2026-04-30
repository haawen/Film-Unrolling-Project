#!/bin/bash
# =============================================================================
# Train R1n recipe on real Mickey CT data (no synthetic GT).
#
#   R1n = pure SS, attachment=emulsion, lr=1e-4, distance-transform attachment,
#         w_conf=0.1, ramp=500, 10000 steps, Fourier sigma=20, hidden=256, 4 layers.
#
# Usage:
#   sbatch Scripts/slurm/slurm_R1n_real.sh [smoke|full]
#
# smoke : 1500 steps, a100-hourly
# full  : 10000 steps, a100-daily
#
# Output: unwrapping/inr/results/R1n_real_<mode>/{train,eval_real}/
#
# No synthetic eval (no GT on real Mickey). Runs unsupervised strip_score.py
# on the output strip and writes unsup_scores.json alongside.
# =============================================================================
#SBATCH --job-name=R1n_real
#SBATCH --partition=a100-daily
#SBATCH --cluster=gmerlin7
#SBATCH --account=merlin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/R1n_real_%j.out
#SBATCH --error=logs/R1n_real_%j.err

set -euo pipefail

MODE="${1:-full}"
case "${MODE}" in
  smoke) STEPS=1500;  RAMP=300 ;;
  full)  STEPS=10000; RAMP=500 ;;
  *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac

PROJECT_DIR="/data/user/li_k1/M_thesis"
CONDA_ENV="nnunet"
REAL_DIR="${PROJECT_DIR}/01_Mickey_3d"
OUT_BASE="${PROJECT_DIR}/unwrapping/inr/results/R1n_real_${MODE}"
TRAIN_DIR="${OUT_BASE}/train"
EVAL_DIR="${OUT_BASE}/eval_real"

mkdir -p logs "${TRAIN_DIR}" "${EVAL_DIR}"

if [ -f "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

echo "============================================================"
echo "  R1n on real Mickey — mode ${MODE}"
echo "============================================================"
echo "  data_dir     : ${REAL_DIR}"
echo "  out          : ${OUT_BASE}"
echo "  steps        : ${STEPS}"
echo "  ramp_steps   : ${RAMP}"
echo "  job id       : ${SLURM_JOB_ID}"
echo "  partition    : ${SLURM_JOB_PARTITION:-unset}"
echo "  started      : $(date)"
echo "============================================================"

cd "${PROJECT_DIR}"

# ── Train on real Mickey (first 20 slices) ──────────────────────────────────
echo ""
echo "--- [1/3] Training on real Mickey ---"
srun python -u -m unwrapping.inr.surface_train \
    --data-dir  "${REAL_DIR}" \
    --out-dir   "${TRAIN_DIR}" \
    --max-slices 20 \
    --attachment emulsion \
    --use-distance-attach \
    --centerline-erode 1 \
    --steps "${STEPS}" \
    --ramp-steps "${RAMP}" \
    --batch-size 131072 \
    --lr 1e-4 \
    --w-attach 1.0 \
    --w-conformal 0.1 \
    --w-res-reg 0.0 \
    --n-fourier 256 \
    --sigma 20.0 \
    --hidden-dim 256 \
    --n-layers-mlp 4 \
    --log-every 50

# ── Eval on real Mickey ─────────────────────────────────────────────────────
echo ""
echo "--- [2/3] Eval on real Mickey ---"
srun python -u -m unwrapping.inr.surface_eval \
    --data-dir "${REAL_DIR}" \
    --out-dir  "${EVAL_DIR}" \
    --max-slices 20 \
    --attachment emulsion \
    --centerline-erode 1 \
    --ckpt "${TRAIN_DIR}/model_final.pt"

# ── Unsupervised scoring (no GT available on real) ──────────────────────────
echo ""
echo "--- [3/3] Unsupervised strip scoring ---"
srun python -u -m unwrapping.inr.strip_score \
    "${EVAL_DIR}/strip.npz" \
    --out-dir "${EVAL_DIR}" \
    --tag "R1n_real_${MODE}"

echo ""
echo "  finished     : $(date)"
