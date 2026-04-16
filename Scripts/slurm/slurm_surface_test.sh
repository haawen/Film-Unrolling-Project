#!/bin/bash
# =============================================================================
# Residual inverse-mapping INR — 5-slice smoke test
#
#   f(u, z) = analytical_spiral(u) + INR_residual(u, z)
#
# Runs:
#   1. analytical-only strip (sanity check)
#   2. 5000-step training (attachment + conformal, ramped)
#   3. trained-model strip + diagnostics
#
# Usage:
#   sbatch Scripts/slurm/slurm_surface_test.sh
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-hourly
#SBATCH --job-name=surface-test
#SBATCH --output=logs/surface_test_%j.out
#SBATCH --error=logs/surface_test_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G

PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
mkdir -p logs

DATA_DIR="${PROJECT_DIR}/01_Mickey_3d"
OUT_DIR="${PROJECT_DIR}/unwrapping/inr/results/surface_test"
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
echo "  Residual inverse-mapping INR — TEST (5 slices)"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Started:    $(date)"
echo "============================================================"

cd "${PROJECT_DIR}"
mkdir -p "${ANAL_DIR}" "${TRAIN_DIR}" "${EVAL_DIR}"

echo ""
echo "--- [1/3] Analytical-only sanity check ---"
srun python -u -m unwrapping.inr.surface_eval \
    --data-dir "${DATA_DIR}" \
    --out-dir "${ANAL_DIR}" \
    --max-slices 5 \
    --analytical-only

echo ""
echo "--- [2/3] Training (8000 steps) ---"
srun python -u -m unwrapping.inr.surface_train \
    --data-dir "${DATA_DIR}" \
    --out-dir "${TRAIN_DIR}" \
    --max-slices 5 \
    --steps 8000 \
    --ramp-steps 2000 \
    --batch-size 131072 \
    --lr 1e-3 \
    --w-attach 1.0 \
    --w-conformal 0.1 \
    --sigma 20.0 \
    --log-every 50

echo ""
echo "--- [3/3] Eval on trained model ---"
srun python -u -m unwrapping.inr.surface_eval \
    --data-dir "${DATA_DIR}" \
    --out-dir "${EVAL_DIR}" \
    --max-slices 5 \
    --ckpt "${TRAIN_DIR}/model_final.pt"

echo ""
echo "=== Surface INR smoke test complete at $(date) ==="
