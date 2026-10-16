#!/bin/bash
# =============================================================================
# Eval-only: re-render an existing model_final.pt at higher pixels-per-winding,
# add the SSIM metric to eval_metrics.json. No training.
#
# Usage:
#   sbatch Scripts/slurm/slurm_eval_only.sh <variant> <preset> <ppw> [tag]
#   sbatch Scripts/slurm/slurm_eval_only.sh R1n hq_video_4k_outer_imperfect 2880 hires
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --job-name=eval_only
#SBATCH --output=logs/eval_only_%x_%j.out
#SBATCH --error=logs/eval_only_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=160G
#SBATCH --partition=a100-hourly

set -euo pipefail

VARIANT="${1:?usage: sbatch slurm_eval_only.sh <variant> <preset> <ppw> [tag]}"
PRESET="${2:?usage: sbatch slurm_eval_only.sh <variant> <preset> <ppw> [tag]}"
PPW="${3:?usage: sbatch slurm_eval_only.sh <variant> <preset> <ppw> [tag]}"
TAG="${4:-ppw${PPW}}"

PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
SYN_DIR="${PROJECT_DIR}/unwrapping/synthetic/results/${PRESET}"
CKPT="${PROJECT_DIR}/unwrapping/inr/results/R1_hq_full_${PRESET}/${VARIANT}/train/model_final.pt"
OUT_DIR="${PROJECT_DIR}/unwrapping/inr/results/eval_only_${PRESET}/${VARIANT}_${TAG}"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
mkdir -p logs "${OUT_DIR}"

[[ -f "${CKPT}" ]] || { echo "Checkpoint not found: ${CKPT}"; exit 1; }

echo "=== eval-only ${VARIANT}/${PRESET} ppw=${PPW} start $(date) ==="
echo "    ckpt : ${CKPT}"
echo "    out  : ${OUT_DIR}"
cd "${PROJECT_DIR}"

srun python -u -m unwrapping.inr.surface_eval \
    --data-dir "${SYN_DIR}" \
    --out-dir  "${OUT_DIR}" \
    --attachment emulsion \
    --centerline-erode 1 \
    --ckpt   "${CKPT}" \
    --gt-npz "${SYN_DIR}/ground_truth.npz" \
    --pixels-per-winding "${PPW}"

echo "=== eval-only ${VARIANT}/${PRESET} ppw=${PPW} complete at $(date) ==="
