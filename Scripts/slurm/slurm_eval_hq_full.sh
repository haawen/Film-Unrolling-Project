#!/bin/bash
# =============================================================================
# Eval-only re-run for HQ video presets at full 256 z-slices.
#
# Loads existing model_final.pt, skips training, runs surface_eval.py
# with no --max-slices so all 256 slices are rendered (256×455 px per frame).
#
# Usage:
#   sbatch Scripts/slurm/slurm_eval_hq_full.sh R1S     hq_video_4k
#   sbatch Scripts/slurm/slurm_eval_hq_full.sh R1S     hq_video_4k_outer_imperfect
#   sbatch Scripts/slurm/slurm_eval_hq_full.sh R1n     hq_video_4k_outer_imperfect
#   sbatch Scripts/slurm/slurm_eval_hq_full.sh R1PF1ws hq_video_4k_outer_imperfect
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --job-name=eval_hq
#SBATCH --output=logs/eval_hq_%x_%j.out
#SBATCH --error=logs/eval_hq_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=160G
#SBATCH --partition=a100-hourly

set -euo pipefail

VARIANT="${1:?usage: sbatch slurm_eval_hq_full.sh <variant> <preset>}"
PRESET="${2:?usage: sbatch slurm_eval_hq_full.sh <variant> <preset>}"

PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
SYN_DIR="${PROJECT_DIR}/unwrapping/synthetic/results/${PRESET}"

# Checkpoint is from the smoke run
CKPT_DIR="${PROJECT_DIR}/unwrapping/inr/results/R1_smoke_${PRESET}/${VARIANT}/train/model_final.pt"
OUT_DIR="${PROJECT_DIR}/unwrapping/inr/results/R1_eval256_${PRESET}/${VARIANT}/eval_synthetic"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

mkdir -p logs "${OUT_DIR}"

echo "============================================================"
echo "  HQ full-256 eval: ${VARIANT} / ${PRESET}"
echo "  ckpt : ${CKPT_DIR}"
echo "  data : ${SYN_DIR}"
echo "  out  : ${OUT_DIR}"
echo "  job  : ${SLURM_JOB_ID:-local}"
echo "  start: $(date)"
echo "============================================================"
cd "${PROJECT_DIR}"

# Attachment flag: R1S and R1PF1ws used emulsion; R1n also emulsion
ATTACH="emulsion"

srun python -u -m unwrapping.inr.surface_eval \
    --data-dir  "${SYN_DIR}" \
    --out-dir   "${OUT_DIR}" \
    --attachment "${ATTACH}" \
    --centerline-erode 1 \
    --ckpt      "${CKPT_DIR}" \
    --gt-npz    "${SYN_DIR}/ground_truth.npz"

echo "=== eval_hq_full ${VARIANT}/${PRESET} complete at $(date) ==="
