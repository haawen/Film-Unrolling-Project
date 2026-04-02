#!/bin/bash
# =============================================================================
# SLURM Job Script — 3D ML Unwrapping Smoke Test (~30 min)
# =============================================================================
# 5 slices, 2000 steps, no smoothness loss — quick check on strip quality
# Usage:
#   sbatch Scripts/slurm/slurm_unwrap_3d_ml_smoke.sh
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-hourly
#SBATCH --job-name=unwrap-ml-smoke
#SBATCH --output=logs/unwrap_ml_smoke_%j.out
#SBATCH --error=logs/unwrap_ml_smoke_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
mkdir -p logs

DATA_DIR="${PROJECT_DIR}/01_Mickey_3d"
OUT_DIR="${PROJECT_DIR}/unwrapping/inr/results/unwrap_3d_ml_smoke"

# ─── Scratch setup ──────────────────────────────────────────────────────────
SCRATCH_DIR="/scratch/${USER}/unwrap_ml_smoke_${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_DIR}/data"

cleanup() {
    echo ""
    echo "─── Cleanup ───────────────────────────────────────────────"
    rm -rf "${SCRATCH_DIR}"
    echo "  Scratch cleaned. Results in: ${OUT_DIR}"
    echo "─────────────────────────────────────────────────────────"
}
trap cleanup EXIT

# Only copy first 2 volume chunks (enough for 5 slices)
echo "Copying data to /scratch..."
ls "${DATA_DIR}"/volume_*.h5 | head -2 | xargs -I{} cp {} "${SCRATCH_DIR}/data/"
echo "  $(ls ${SCRATCH_DIR}/data/*.h5 | wc -l) files copied"

# ─── Activate conda ─────────────────────────────────────────────────────────
if [ -f "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

echo "============================================================"
echo "  ML Unwrapping Smoke Test — Merlin7 A100"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Node:       ${SLURM_NODELIST}"
echo "  Started:    $(date)"
echo "  Config:     5 slices, 2000 steps, w-smooth=0 (no smoothing)"
echo "============================================================"

python -u -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
"

cd "${PROJECT_DIR}"
mkdir -p "${OUT_DIR}"

echo ""
echo "═══ Phase 1: Training (5 slices, 2000 steps, no smoothness) ═══"
srun python -u -m unwrapping.inr.unwrap_train_3d \
    --data-dir "${SCRATCH_DIR}/data" \
    --out-dir "${OUT_DIR}" \
    --n-fourier 128 \
    --sigma 10.0 \
    --hidden-dim 256 \
    --n-layers 4 \
    --steps 2000 \
    --batch-size 65536 \
    --lr 1e-3 \
    --w-smooth 0.0 \
    --log-every 100 \
    --save-every 500 \
    --max-slices 5

echo ""
echo "═══ Phase 2: Generating strip ═══"
srun python -u -m unwrapping.inr.unwrap_eval_3d \
    --data-dir "${SCRATCH_DIR}/data" \
    --model-dir "${OUT_DIR}" \
    --out-dir "${OUT_DIR}/eval" \
    --max-slices 5

echo ""
echo "============================================================"
echo "  Smoke test complete at $(date)"
echo "============================================================"
