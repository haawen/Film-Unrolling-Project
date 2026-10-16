#!/bin/bash
# =============================================================================
# Geodesic unwrapping v4 — test (5 slices)
#
# Computes u as geodesic distance through film mask from a single source point.
# Air gaps block propagation → distance follows the spiral topology.
# No training — purely geometric computation.
#
# Usage:
#   sbatch Scripts/slurm/slurm_geo_unwrap_test.sh
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-hourly
#SBATCH --job-name=geo-test
#SBATCH --output=logs/geo_unwrap_test_%j.out
#SBATCH --error=logs/geo_unwrap_test_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
mkdir -p logs

DATA_DIR="${PROJECT_DIR}/01_Mickey_3d"
OUT_DIR="${PROJECT_DIR}/unwrapping/inr/results/geodesic_unwrap_test"

if [ -f "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

echo "============================================================"
echo "  Geodesic Unwrapping v4 — TEST (5 slices)"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Started:    $(date)"
echo "============================================================"

python -u -c "
import skimage; print(f'  scikit-image: {skimage.__version__}')
from skimage.graph import MCP_Geometric; print('  MCP_Geometric: available')
"

cd "${PROJECT_DIR}"
mkdir -p "${OUT_DIR}"

srun python -u -m unwrapping.inr.geodesic_unwrap \
    --data-dir "${DATA_DIR}" \
    --out-dir "${OUT_DIR}" \
    --max-slices 5

echo ""
echo "=== Geodesic test complete at $(date) ==="
