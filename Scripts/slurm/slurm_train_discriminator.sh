#!/bin/bash
# =============================================================================
# Train the strip discriminator CNN on synthetic clean_4k GT strip.
#
# Usage:
#   sbatch Scripts/slurm/slurm_train_discriminator.sh [preset]
#   preset defaults to clean_4k.
# =============================================================================
#SBATCH --job-name=disc_train
#SBATCH --partition=a100-hourly
#SBATCH --cluster=gmerlin7
#SBATCH --account=merlin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=logs/disc_train_%j.out
#SBATCH --error=logs/disc_train_%j.err

set -euo pipefail

PRESET="${1:-imperfect_4k}"
MODE="${2:-rendered_at_gt}"   # rendered_at_gt | gt_strip
PROJECT_DIR="/data/user/li_k1/M_thesis"
CONDA_ENV="nnunet"
GT_NPZ="${PROJECT_DIR}/unwrapping/synthetic/results/${PRESET}/ground_truth.npz"
VOL_H5="${PROJECT_DIR}/unwrapping/synthetic/results/${PRESET}/volume_0000-0019.h5"
PROBS_H5="${PROJECT_DIR}/unwrapping/synthetic/results/${PRESET}/volume_0000-0019_Probabilities.h5"
OUT_DIR="${PROJECT_DIR}/unwrapping/inr/results/discriminator_${PRESET}_${MODE}"

mkdir -p logs "${OUT_DIR}"
if [ -f "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"
echo "============================================================"
echo "  Training strip discriminator on ${PRESET}"
echo "============================================================"
echo "  gt_npz : ${GT_NPZ}"
echo "  out    : ${OUT_DIR}"
echo "  started: $(date)"
echo "============================================================"

srun python -u -m unwrapping.inr.strip_discriminator \
    --gt-npz "${GT_NPZ}" \
    --strip-mode "${MODE}" \
    --volume-h5 "${VOL_H5}" \
    --probs-h5 "${PROBS_H5}" \
    --out-dir "${OUT_DIR}" \
    --patch-w 64 \
    --batch-size 128 \
    --steps 2000 \
    --lr 1e-3 \
    --log-every 100

echo "  finished: $(date)"
