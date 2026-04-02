#!/bin/bash
# =============================================================================
# SLURM Job Script — 3D Unwrapping (all volume chunks)
# =============================================================================
# Usage:
#   sbatch Scripts/slurm/slurm_unwrap_3d.sh
#
# Submit from: /data/user/$USER/M_thesis
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-general
#SBATCH --job-name=unwrap-3d
#SBATCH --output=logs/unwrap_3d_%j.out
#SBATCH --error=logs/unwrap_3d_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:0

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
mkdir -p logs

DATA_DIR="${PROJECT_DIR}/01_Mickey_3d"
OUT_DIR="${PROJECT_DIR}/unwrapping/inr/results/unwrap_3d_full"

# ─── Scratch setup ──────────────────────────────────────────────────────────
SCRATCH_DIR="/scratch/${USER}/unwrap_3d_${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_DIR}/data" "${SCRATCH_DIR}/results"

cleanup() {
    echo ""
    echo "─── Cleanup ───────────────────────────────────────────────"
    rm -rf "${SCRATCH_DIR}"
    echo "  Scratch cleaned. Results in: ${OUT_DIR}"
    echo "─────────────────────────────────────────────────────────"
}
trap cleanup EXIT

# Copy all volume data to scratch
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
echo "  3D Unwrapping — All Volume Chunks"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Node:       ${SLURM_NODELIST}"
echo "  Started:    $(date)"
echo "============================================================"

# ─── Run ─────────────────────────────────────────────────────────────────────
cd "${PROJECT_DIR}"

# Use persistent OUT_DIR for checkpoints (survive job restarts)
# Data reads from scratch for speed
mkdir -p "${OUT_DIR}"
srun python -m unwrapping.inr.unwrap_3d_stack \
    --data-dir "${SCRATCH_DIR}/data" \
    --out-dir "${OUT_DIR}"

echo ""
echo "============================================================"
echo "  3D Unwrapping complete at $(date)"
echo "============================================================"
