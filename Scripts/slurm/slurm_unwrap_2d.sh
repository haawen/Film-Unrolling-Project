#!/bin/bash
# =============================================================================
# SLURM Job Script — Learned Unwrapping (Step 1b) on 2D CT Slice
# =============================================================================
# Usage:
#   sbatch Scripts/slurm/slurm_unwrap_2d.sh
#
# Submit from: /data/user/$USER/M_thesis
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-hourly
#SBATCH --job-name=unwrap-2d
#SBATCH --output=logs/unwrap_2d_%j.out
#SBATCH --error=logs/unwrap_2d_%j.err
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
mkdir -p logs

IMAGE="${PROJECT_DIR}/01_Mickey_hdf_subset/01_Mickey_Stitch_Stitch_Export_0835.h5"
PROBS="${PROJECT_DIR}/01_Mickey_hdf_subset/01_Mickey_Stitch_Stitch_Export_0835-image_Probabilities.h5"
OUT_DIR="${PROJECT_DIR}/unwrapping/inr/results/unwrap_2d_v2"

# ─── Scratch setup ──────────────────────────────────────────────────────────
SCRATCH_DIR="/scratch/${USER}/unwrap_2d_${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_DIR}"

cleanup() {
    echo ""
    echo "─── Cleanup ───────────────────────────────────────────────"
    if [ -d "${SCRATCH_DIR}/results" ]; then
        echo "  Copying results back to /data/user..."
        mkdir -p "${OUT_DIR}"
        rsync -a "${SCRATCH_DIR}/results/" "${OUT_DIR}/"
        echo "  Results saved to: ${OUT_DIR}"
    fi
    rm -rf "${SCRATCH_DIR}"
    echo "─────────────────────────────────────────────────────────"
}
trap cleanup EXIT

# Copy data to scratch
echo "Copying data to /scratch..."
cp "${IMAGE}" "${SCRATCH_DIR}/image.h5"
cp "${PROBS}" "${SCRATCH_DIR}/probs.h5"

# ─── Activate conda ─────────────────────────────────────────────────────────
if [ -f "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

# ─── Info ────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Learned Unwrapping (Step 1b) — 2D — Merlin7 A100"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Node:       ${SLURM_NODELIST}"
echo "  Started:    $(date)"
echo "============================================================"

python -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
"

# ─── Train ───────────────────────────────────────────────────────────────────
cd "${PROJECT_DIR}"

echo ""
echo "═══ Phase 1: Training deformation network ═══"
srun python -m unwrapping.inr.unwrap_train \
    --image "${SCRATCH_DIR}/image.h5" \
    --probs "${SCRATCH_DIR}/probs.h5" \
    --out-dir "${SCRATCH_DIR}/results" \
    --n-fourier 512 \
    --sigma 10.0 \
    --hidden-dim 512 \
    --n-layers 6 \
    --steps 10000 \
    --batch-size 131072 \
    --lr 1e-3 \
    --w-smooth 0.001 \
    --log-every 200 \
    --save-every 2000

# ─── Evaluate ────────────────────────────────────────────────────────────────
echo ""
echo "═══ Phase 2: Generating strip ═══"
srun python -m unwrapping.inr.unwrap_eval \
    --image "${SCRATCH_DIR}/image.h5" \
    --probs "${SCRATCH_DIR}/probs.h5" \
    --model-dir "${SCRATCH_DIR}/results" \
    --out-dir "${SCRATCH_DIR}/results/eval" \
    --strip-height 20

echo ""
echo "============================================================"
echo "  Unwrapping complete at $(date)"
echo "============================================================"
