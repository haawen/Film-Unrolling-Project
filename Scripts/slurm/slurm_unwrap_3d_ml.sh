#!/bin/bash
# =============================================================================
# SLURM Job Script — 3D Learned Unwrapping (Step 1b ML)
# =============================================================================
# Usage:
#   sbatch Scripts/slurm/slurm_unwrap_3d_ml.sh
#
# Submit from: /data/user/$USER/M_thesis
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-daily
#SBATCH --job-name=unwrap-3d-ml
#SBATCH --output=logs/unwrap_3d_ml_%j.out
#SBATCH --error=logs/unwrap_3d_ml_%j.err
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
mkdir -p logs

DATA_DIR="${PROJECT_DIR}/01_Mickey_3d"
OUT_DIR="${PROJECT_DIR}/unwrapping/inr/results/unwrap_3d_ml"

# ─── Scratch setup ──────────────────────────────────────────────────────────
SCRATCH_DIR="/scratch/${USER}/unwrap_3d_ml_${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_DIR}/data"

cleanup() {
    echo ""
    echo "─── Cleanup ───────────────────────────────────────────────"
    rm -rf "${SCRATCH_DIR}"
    echo "  Scratch cleaned. Results in: ${OUT_DIR}"
    echo "─────────────────────────────────────────────────────────"
}
trap cleanup EXIT

# Copy all volume data to scratch for fast I/O
echo "Copying data to /scratch..."
cp "${DATA_DIR}"/volume_*.h5 "${SCRATCH_DIR}/data/"
echo "  $(ls ${SCRATCH_DIR}/data/*.h5 | wc -l) files copied"

# ─── Activate conda ─────────────────────────────────────────────────────────
if [ -f "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

# ─── Info ────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  3D Learned Unwrapping (Step 1b ML) — Merlin7 A100"
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
mkdir -p "${OUT_DIR}"

echo ""
echo "═══ Phase 1: Training 3D deformation network ═══"
srun python -u -m unwrapping.inr.unwrap_train_3d \
    --data-dir "${SCRATCH_DIR}/data" \
    --out-dir "${OUT_DIR}" \
    --n-fourier 512 \
    --sigma 10.0 \
    --hidden-dim 512 \
    --n-layers 6 \
    --steps 20000 \
    --batch-size 131072 \
    --lr 1e-3 \
    --w-smooth 0.001 \
    --log-every 200 \
    --save-every 2000 \
    --max-slices 50

# ─── Evaluate ────────────────────────────────────────────────────────────────
echo ""
echo "═══ Phase 2: Generating 3D strip ═══"
srun python -u -m unwrapping.inr.unwrap_eval_3d \
    --data-dir "${SCRATCH_DIR}/data" \
    --model-dir "${OUT_DIR}" \
    --out-dir "${OUT_DIR}/eval" \
    --max-slices 50

echo ""
echo "============================================================"
echo "  3D Learned Unwrapping complete at $(date)"
echo "============================================================"
