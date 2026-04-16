#!/bin/bash
# =============================================================================
# Residual inverse-mapping INR — full run (all slices)
#
#   f(u, z) = analytical_spiral(u) + INR_residual(u, z)
#
# Usage:
#   sbatch Scripts/slurm/slurm_surface_full.sh
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-daily
#SBATCH --job-name=surface-full
#SBATCH --output=logs/surface_full_%j.out
#SBATCH --error=logs/surface_full_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=128G

PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
mkdir -p logs

DATA_DIR="${PROJECT_DIR}/01_Mickey_3d"
OUT_DIR="${PROJECT_DIR}/unwrapping/inr/results/surface_full"
ANAL_DIR="${OUT_DIR}/analytical_only"
TRAIN_DIR="${OUT_DIR}/train"
EVAL_DIR="${OUT_DIR}/eval"

if [ -f "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

echo "============================================================"
echo "  Residual inverse-mapping INR — FULL RUN"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Started:    $(date)"
echo "============================================================"

cd "${PROJECT_DIR}"
mkdir -p "${ANAL_DIR}" "${TRAIN_DIR}" "${EVAL_DIR}"

echo ""
echo "--- [1/3] Analytical-only baseline ---"
srun python -u -m unwrapping.inr.surface_eval \
    --data-dir "${DATA_DIR}" \
    --out-dir "${ANAL_DIR}" \
    --analytical-only

echo ""
echo "--- [2/3] Training (15000 steps) ---"
srun python -u -m unwrapping.inr.surface_train \
    --data-dir "${DATA_DIR}" \
    --out-dir "${TRAIN_DIR}" \
    --steps 15000 \
    --ramp-steps 3000 \
    --batch-size 131072 \
    --lr 1e-3 \
    --w-attach 1.0 \
    --w-conformal 0.1 \
    --log-every 100 \
    --ckpt-every 5000

echo ""
echo "--- [3/3] Eval on trained model ---"
srun python -u -m unwrapping.inr.surface_eval \
    --data-dir "${DATA_DIR}" \
    --out-dir "${EVAL_DIR}" \
    --ckpt "${TRAIN_DIR}/model_final.pt"

echo ""
echo "=== Surface INR full run complete at $(date) ==="
