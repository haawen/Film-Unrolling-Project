#!/bin/bash
# =============================================================================
# Architecture-comparison full runs (10k steps, a100-daily).
#
# Companion to slurm_arch_smoke.sh — same recipe but a full-budget train+eval,
# parameterized by both the architecture variant and the supervision mode
# (R1S = supervised, R1n = pure self-supervised).
#
# Usage:
#   sbatch Scripts/slurm/slurm_arch_full.sh <variant> <mode> [preset]
#     variant: G1_grid | H1_hash | S2_wire | S3_finer
#     mode:    R1S | R1n
#     preset:  default hq_video_4k_outer_imperfect
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --job-name=arch_full
#SBATCH --output=logs/arch_full_%x_%j.out
#SBATCH --error=logs/arch_full_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=160G
#SBATCH --partition=a100-hourly
#SBATCH --time=00:55:00

set -euo pipefail

VARIANT="${1:?usage: sbatch slurm_arch_full.sh <variant> <mode> [preset]}"
MODE="${2:?usage: sbatch slurm_arch_full.sh <variant> <mode> [preset]}"
PRESET="${3:-hq_video_4k_outer_imperfect}"

PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
SYN_DIR="${PROJECT_DIR}/unwrapping/synthetic/results/${PRESET}"
SUFFIX="${W_STRIP_MSE:+_smse${W_STRIP_MSE}}"
SUFFIX="${SUFFIX//./}"  # strip dots for cleaner dir names
[ "${W_STRIP_MSE:-0.0}" = "0.0" ] && SUFFIX=""
OUT_BASE="${PROJECT_DIR}/unwrapping/inr/results/arch_full_${PRESET}/${VARIANT}_${MODE}${SUFFIX}"
TRAIN_DIR="${OUT_BASE}/train"
EVAL_DIR="${OUT_BASE}/eval_synthetic"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
mkdir -p logs "${TRAIN_DIR}" "${EVAL_DIR}"

STEPS=10000
RAMP=3000
BATCH=131072
LR="1e-4"

# --- Mode (supervised vs self-supervised) -----------------------------------
case "${MODE}" in
  R1S)  SUP_STEPS=${STEPS} ;;                       # supervised throughout
  R1n)  SUP_STEPS=0 ;;                              # pure self-supervised
  *)    echo "Unknown mode '${MODE}'. Use R1S or R1n."; exit 1 ;;
esac

W_STRIP_MSE="${W_STRIP_MSE:-0.0}"

COMMON_FLAGS=(
  --data-dir   "${SYN_DIR}"
  --out-dir    "${TRAIN_DIR}"
  --attachment emulsion
  --centerline-erode 1
  --supervised-steps "${SUP_STEPS}"
  --gt-npz     "${SYN_DIR}/ground_truth.npz"
  --steps      "${STEPS}"
  --ramp-steps "${RAMP}"
  --batch-size "${BATCH}"
  --use-distance-attach
  --w-attach 1.0
  --w-conformal 0.1
  --w-res-reg 0.0
  --w-strip-mse "${W_STRIP_MSE}"
  --log-every 100
  --ckpt-every 2000
)

# --- Architecture flags (must match slurm_arch_smoke.sh) --------------------
ARCH_FLAGS=()
case "${VARIANT}" in
  G1_grid)
    ARCH_FLAGS+=(
      --model-type grid
      --grid-n-levels 8 --grid-features 2
      --grid-base-u 64   --grid-base-z 16
      --grid-finest-u 8192 --grid-finest-z 256
      --grid-hidden 64   --grid-mlp-layers 2
      --encoder-lr-mult 100.0
    ) ;;
  H1_hash)
    ARCH_FLAGS+=(
      --model-type hash
      --hash-n-levels 16 --hash-features 2
      --hash-log2-size 19
      --hash-base-res 16 --hash-finest-res 8192
      --hash-hidden 64   --hash-mlp-layers 2
      --encoder-lr-mult 100.0
    ) ;;
  S2_wire)
    ARCH_FLAGS+=(
      --model-type wire
      --hidden-dim 256 --n-layers-mlp 4
      --wire-omega 20.0 --wire-sigma 10.0
    ) ;;
  S3_finer)
    ARCH_FLAGS+=(
      --model-type finer
      --hidden-dim 256 --n-layers-mlp 4
      --finer-omega 30.0 --finer-bias-k 5.0
    ) ;;
  *)
    echo "Unknown variant '${VARIANT}'. Valid: G1_grid H1_hash S2_wire S3_finer"
    exit 1 ;;
esac

echo "============================================================"
echo "  Arch full: ${VARIANT}  mode=${MODE}  preset=${PRESET}"
echo "  steps=${STEPS}  ramp=${RAMP}  sup=${SUP_STEPS}  lr=${LR}"
echo "  arch flags: ${ARCH_FLAGS[*]}"
echo "  out: ${OUT_BASE}"
echo "  start: $(date)"
echo "============================================================"
cd "${PROJECT_DIR}"

srun python -u -m unwrapping.inr.surface_train \
  --lr "${LR}" \
  "${COMMON_FLAGS[@]}" \
  "${ARCH_FLAGS[@]}"

echo ""
echo "--- Eval on synthetic (with GT metrics + SSIM) ---"
srun python -u -m unwrapping.inr.surface_eval \
  --data-dir "${SYN_DIR}" \
  --out-dir  "${EVAL_DIR}" \
  --attachment emulsion \
  --centerline-erode 1 \
  --ckpt   "${TRAIN_DIR}/model_final.pt" \
  --gt-npz "${SYN_DIR}/ground_truth.npz" \
  --pixels-per-winding 1440

echo "=== arch full ${VARIANT}_${MODE} done at $(date) ==="
