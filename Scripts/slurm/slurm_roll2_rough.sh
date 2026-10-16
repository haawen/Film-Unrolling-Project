#!/bin/bash
# Rough Roll2 unwrapping by z-interpolation of the 4 segmentation walks.
# CPU only (numpy/matplotlib) -> default merlin7 cluster, no GPU.
#   arg1 = mode : geometry (default) | full
#   arg2 = nphi : arc samples per turn (default 1440; use 8192 for frame-by-frame)
#SBATCH --job-name=roll2_rough
#SBATCH --output=logs/roll2_rough_%j.out
#SBATCH --error=logs/roll2_rough_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --partition=daily
#SBATCH --time=06:00:00

set -euo pipefail
MODE="${1:-geometry}"
NPHI="${2:-1440}"
PROJECT_DIR="$HOME/M_thesis"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "${PROJECT_DIR}"
mkdir -p logs

echo "=== roll2 rough unwrap ($MODE, nphi=$NPHI)  $(date) ==="
python -u -m unwrapping.inr.unroll_roll2_rough \
    --probs-dir data/roll2_probs --anchor-zs 200,700,1200,1700 \
    --ct-raw-dir 02_Sample_raw --ct-chunk-dir 02_Sample_3d \
    --z-range 1,1936 --mode "${MODE}" --nphi "${NPHI}" \
    --overlay-zs 60,200,430,700,950,1200,1450,1700,1880 \
    --out-dir unwrapping/inr/results/roll2_rough
echo "=== done $(date) ==="
