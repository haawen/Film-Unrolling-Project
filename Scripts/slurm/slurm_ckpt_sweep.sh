#!/bin/bash
# =============================================================================
# Checkpoint sweep: re-eval every saved intermediate checkpoint of an existing
# training run. Use to find the peak row_corr before discriminator-driven drift.
#
# Usage:
#   sbatch Scripts/slurm/slurm_ckpt_sweep.sh <variant> <preset>
#   e.g. sbatch Scripts/slurm/slurm_ckpt_sweep.sh R1D3 imperfect_4k
#
# Expects intermediate checkpoints at <train_dir>/ckpt_step*.pt
# (set --ckpt-every during training to produce these).
#
# Writes eval_synthetic_step{N}/eval_metrics.json per checkpoint and a
# summary CSV listing step → row_corr.
# =============================================================================
#SBATCH --job-name=ckpt_sweep
#SBATCH --partition=a100-hourly
#SBATCH --cluster=gmerlin7
#SBATCH --account=merlin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:55:00
#SBATCH --output=logs/ckpt_sweep_%j.out
#SBATCH --error=logs/ckpt_sweep_%j.err

set -euo pipefail

VARIANT="${1:?usage: $0 <variant> <preset>}"
PRESET="${2:?usage: $0 <variant> <preset>}"

PROJECT_DIR="/data/user/li_k1/M_thesis"
CONDA_ENV="nnunet"
SYN_DIR="${PROJECT_DIR}/unwrapping/synthetic/results/${PRESET}"
OUT_BASE="${PROJECT_DIR}/unwrapping/inr/results/R1_full_imperfect_4k/${VARIANT}"
TRAIN_DIR="${OUT_BASE}/train"
SUMMARY_CSV="${OUT_BASE}/ckpt_sweep_summary.csv"

if [ ! -d "${TRAIN_DIR}" ]; then
  echo "Train dir not found: ${TRAIN_DIR}" >&2
  exit 1
fi

if [ -f "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"

echo "step,row_corr,strip_rmse,median_shift" > "${SUMMARY_CSV}"

# Iterate over intermediate checkpoints in numeric order.
shopt -s nullglob
ckpts=( "${TRAIN_DIR}"/ckpt_step*.pt )
if [ ${#ckpts[@]} -eq 0 ]; then
  echo "No ckpt_step*.pt files found in ${TRAIN_DIR}" >&2
  exit 1
fi
echo "Found ${#ckpts[@]} intermediate checkpoints"

for ckpt in "${ckpts[@]}"; do
  fname="$(basename "${ckpt}")"
  step="${fname#ckpt_step}"
  step="${step%.pt}"
  EVAL_DIR="${OUT_BASE}/eval_synthetic_step${step}"
  echo ""
  echo "=== eval ${VARIANT} step ${step} ==="
  mkdir -p "${EVAL_DIR}"
  srun --exclusive -n1 python -u -m unwrapping.inr.surface_eval \
      --data-dir "${SYN_DIR}" \
      --out-dir  "${EVAL_DIR}" \
      --attachment emulsion \
      --centerline-erode 1 \
      --ckpt   "${ckpt}" \
      --gt-npz "${SYN_DIR}/ground_truth.npz"
  rc=$(python -c "import json; print(json.load(open('${EVAL_DIR}/eval_metrics.json'))['row_corr_mean'])")
  rmse=$(python -c "import json; print(json.load(open('${EVAL_DIR}/eval_metrics.json'))['strip_rmse'])")
  ms=$(python -c "import json; print(json.load(open('${EVAL_DIR}/eval_metrics.json'))['median_shift_px'])")
  echo "${step},${rc},${rmse},${ms}" >> "${SUMMARY_CSV}"
done

# Final eval at model_final.pt (10000 steps) for completeness.
final_ckpt="${TRAIN_DIR}/model_final.pt"
if [ -f "${final_ckpt}" ]; then
  EVAL_DIR="${OUT_BASE}/eval_synthetic"
  if [ -f "${EVAL_DIR}/eval_metrics.json" ]; then
    rc=$(python -c "import json; print(json.load(open('${EVAL_DIR}/eval_metrics.json'))['row_corr_mean'])")
    rmse=$(python -c "import json; print(json.load(open('${EVAL_DIR}/eval_metrics.json'))['strip_rmse'])")
    ms=$(python -c "import json; print(json.load(open('${EVAL_DIR}/eval_metrics.json'))['median_shift_px'])")
    echo "final,${rc},${rmse},${ms}" >> "${SUMMARY_CSV}"
  fi
fi

echo ""
echo "Summary CSV:"
cat "${SUMMARY_CSV}"
echo ""
echo "Sorted by row_corr desc:"
sort -t, -k2 -nr "${SUMMARY_CSV}" | head -5
