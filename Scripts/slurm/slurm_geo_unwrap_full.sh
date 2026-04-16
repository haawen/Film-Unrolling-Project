#!/bin/bash
# =============================================================================
# Geodesic unwrapping v4 — full run (200 slices)
#
# Usage:
#   sbatch Scripts/slurm/slurm_geo_unwrap_full.sh
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-daily
#SBATCH --job-name=geo-full
#SBATCH --output=logs/geo_unwrap_full_%j.out
#SBATCH --error=logs/geo_unwrap_full_%j.err
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G

PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
mkdir -p logs

DATA_DIR="${PROJECT_DIR}/01_Mickey_3d"
OUT_DIR="${PROJECT_DIR}/unwrapping/inr/results/geodesic_unwrap_full"

# ─── Scratch setup ──────────────────────────────────────────────────────────
SCRATCH_DIR="/scratch/${USER}/geo_unwrap_${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_DIR}/data"

cleanup() {
    echo ""
    echo "─── Cleanup ───────────────────────────────────────────────"
    rm -rf "${SCRATCH_DIR}"
    echo "  Scratch cleaned. Results in: ${OUT_DIR}"
    echo "─────────────────────────────────────────────────────────"
}
trap cleanup EXIT

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

echo "============================================================"
echo "  Geodesic Unwrapping v4 — FULL (200 slices)"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Node:       ${SLURM_NODELIST}"
echo "  Started:    $(date)"
echo "============================================================"

cd "${PROJECT_DIR}"
mkdir -p "${OUT_DIR}"

srun python -u -m unwrapping.inr.geodesic_unwrap \
    --data-dir "${SCRATCH_DIR}/data" \
    --out-dir "${OUT_DIR}" \
    --max-slices 200

echo ""
echo "============================================================"
echo "  Geodesic unwrapping complete at $(date)"
echo "============================================================"
