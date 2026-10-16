#!/bin/bash
# Submit focused evaluations for R1n variants at GT pixels-per-winding.
#SBATCH --job-name=eval_R1n_ppw5127
#SBATCH --partition=a100-hourly
#SBATCH --cluster=gmerlin7
#SBATCH --account=merlin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=logs/eval_R1n_ppw5127_%j.out
#SBATCH --error=logs/eval_R1n_ppw5127_%j.err

set -euo pipefail

PROJECT_DIR="/data/user/li_k1/M_thesis"
CONDA_ENV="nnunet"
PPW=5127

if [ -f "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"

for PRESET in "video_4k" "video_4k_imperfect"; do
  VAR="R1n"
  EVAL_DIR="unwrapping/inr/results/R1_full_${PRESET}/${VAR}/eval_synthetic_ppw5127"
  COMPARE_DIR="docs/images/compare_${PRESET}_${VAR}_ppw5127"
  mkdir -p "${EVAL_DIR}" "${COMPARE_DIR}"
  echo "=== ${PRESET} / ${VAR} (PPW=${PPW}) ==="

  srun python -u -m unwrapping.inr.surface_eval \
    --data-dir "unwrapping/synthetic/results/${PRESET}" \
    --out-dir "${EVAL_DIR}" \
    --attachment emulsion \
    --centerline-erode 1 \
    --ckpt "unwrapping/inr/results/R1_full_${PRESET}/${VAR}/train/model_final.pt" \
    --gt-npz "unwrapping/synthetic/results/${PRESET}/ground_truth.npz" \
    --pixels-per-winding "${PPW}"

  srun python -u -m unwrapping.inr.strip_compare \
    --gt-npz "unwrapping/synthetic/results/${PRESET}/ground_truth.npz" \
    --pred-npz "${EVAL_DIR}/strip.npz" \
    --out-dir "${COMPARE_DIR}" \
    --n-panels 8 \
    --n-crops 4 \
    --crop-w 1200
done

echo "DONE $(date)"