#!/bin/bash
# =============================================================================
# Eval-only render-interp comparison on an existing checkpoint.
#
# Re-renders an existing trained checkpoint under different intensity-
# interpolation modes (R1 cubic vs bilinear) to measure the rendering-side
# contribution to the SSIM ceiling.
#
# Usage:
#   sbatch Scripts/slurm/slurm_render_compare.sh <variant> <interp> [preset]
#     variant : matches the train dir under arch_full_<preset>/
#     interp  : bilinear | bicubic
#     preset  : default hq_video_4k_outer_imperfect
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --job-name=render_cmp
#SBATCH --output=logs/render_cmp_%x_%j.out
#SBATCH --error=logs/render_cmp_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=160G
#SBATCH --partition=a100-hourly
#SBATCH --time=00:45:00

set -euo pipefail

VARIANT="${1:?usage: sbatch slurm_render_compare.sh <variant> <interp_or_mt> [preset]}"
INTERP="${2:?usage: sbatch slurm_render_compare.sh <variant> <interp_or_mt> [preset]}"
PRESET="${3:-hq_video_4k_outer_imperfect}"

# INTERP may also be a multitap variant: mt<N>_<combine>[_<delta>]
# Examples: mt3_mean, mt5_mean, mt5_max, mt5_weighted, mt7_mean
MULTITAP_FLAGS=()
case "${INTERP}" in
  bilinear|bicubic)
    INTERP_CLI=(--render-interp "${INTERP}") ;;
  mt*)
    # parse "mt<N>_<combine>" possibly with "_<delta>" suffix.
    N=$(echo "${INTERP}" | sed -E 's/^mt([0-9]+)_.*/\1/')
    COMBINE=$(echo "${INTERP}" | sed -E 's/^mt[0-9]+_([a-z]+).*/\1/')
    DELTA=$(echo "${INTERP}" | sed -nE 's/^mt[0-9]+_[a-z]+_([0-9.]+)$/\1/p')
    DELTA=${DELTA:-4.0}
    MULTITAP_FLAGS=(--multitap-n "${N}" --multitap-combine "${COMBINE}"
                    --multitap-delta-px "${DELTA}")
    INTERP_CLI=()
    echo "  multitap: n=${N} combine=${COMBINE} delta=${DELTA}px" ;;
  *)
    echo "Unknown interp '${INTERP}'. Use: bilinear|bicubic|mt<N>_<combine>[_<delta>]"
    exit 1 ;;
esac

PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
SYN_DIR="${PROJECT_DIR}/unwrapping/synthetic/results/${PRESET}"
TRAIN_DIR="${PROJECT_DIR}/unwrapping/inr/results/arch_full_${PRESET}/${VARIANT}/train"
EVAL_DIR="${PROJECT_DIR}/unwrapping/inr/results/arch_full_${PRESET}/${VARIANT}/eval_${INTERP}"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
mkdir -p logs "${EVAL_DIR}"

if [ ! -f "${TRAIN_DIR}/model_final.pt" ]; then
  echo "ERROR: no checkpoint at ${TRAIN_DIR}/model_final.pt"
  exit 1
fi

echo "============================================================"
echo "  Render-compare: variant=${VARIANT}  interp=${INTERP}"
echo "  preset=${PRESET}"
echo "  ckpt=${TRAIN_DIR}/model_final.pt"
echo "  out=${EVAL_DIR}"
echo "  start: $(date)"
echo "============================================================"
cd "${PROJECT_DIR}"

srun python -u -m unwrapping.inr.surface_eval \
  --data-dir "${SYN_DIR}" \
  --out-dir  "${EVAL_DIR}" \
  --attachment emulsion \
  --centerline-erode 1 \
  --ckpt   "${TRAIN_DIR}/model_final.pt" \
  --gt-npz "${SYN_DIR}/ground_truth.npz" \
  --pixels-per-winding 1440 \
  "${INTERP_CLI[@]}" \
  "${MULTITAP_FLAGS[@]}"

echo "=== render-compare ${VARIANT} ${INTERP} done at $(date) ==="
