#!/bin/bash
# =============================================================================
# Architecture-comparison smoke runs (R1S supervised, hq_video_4k_outer_imperfect).
#
# Companion to .claude/docs/ss_unwrapping_plan.md
# (section "Architecture literature review — alternatives to hash").
#
# Goal: raise R1S SSIM from 0.50 (Fourier ceiling) toward 0.77 (oracle) by
# swapping the input encoding / activation. Tests one architecture per job;
# pick the winner on SSIM after eval.
#
# Variants:
#   F1_sigma40 — σ=40 Fourier (cheapest sanity, no arch change)
#   F1_sigma80 — σ=80 Fourier
#   G1_grid    — dense multi-resolution 2D feature grid (primary pick)
#   H1_hash    — Instant-NGP hash encoding (parallel reference)
#   S2_wire    — WIRE Gabor wavelet MLP
#   S3_finer   — FINER variable-periodic MLP
#
# Usage:
#   sbatch Scripts/slurm/slurm_arch_smoke.sh <variant> [preset]
# Default preset: hq_video_4k_outer_imperfect.
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --job-name=arch_sm
#SBATCH --output=logs/arch_smoke_%x_%j.out
#SBATCH --error=logs/arch_smoke_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=160G
#SBATCH --partition=a100-hourly

set -euo pipefail

VARIANT="${1:?usage: sbatch slurm_arch_smoke.sh <variant> [preset]}"
PRESET="${2:-hq_video_4k_outer_imperfect}"

PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
SYN_DIR="${PROJECT_DIR}/unwrapping/synthetic/results/${PRESET}"
OUT_BASE="${PROJECT_DIR}/unwrapping/inr/results/arch_smoke_${PRESET}/${VARIANT}"
TRAIN_DIR="${OUT_BASE}/train"
EVAL_DIR="${OUT_BASE}/eval_synthetic"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
mkdir -p logs "${TRAIN_DIR}" "${EVAL_DIR}"

# --- Common R1S (pure supervised) recipe ------------------------------------
STEPS=2000
RAMP=500
BATCH=131072
# SUP_STEPS=STEPS keeps us in supervised mode the entire run (R1S).
SUP_STEPS=${STEPS}

# Per-variant flags: each variant overrides only what it needs.
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
  --log-every 100
  --ckpt-every 1000
)

ARCH_FLAGS=()
LR="1e-4"
case "${VARIANT}" in
  F1_sigma40)
    ARCH_FLAGS+=(--model-type inr --sigma 40.0 --hidden-dim 256 --n-layers-mlp 4) ;;
  F1_sigma80)
    ARCH_FLAGS+=(--model-type inr --sigma 80.0 --hidden-dim 256 --n-layers-mlp 4) ;;
  G1_grid)
    # Encoder uses encoder_lr_mult × LR (default 100×) → grid params at 1e-2,
    # MLP head at 1e-4. This split is essential for grid-based INRs.
    ARCH_FLAGS+=(
      --model-type grid
      --grid-n-levels 8 --grid-features 2
      --grid-base-u 64   --grid-base-z 16
      --grid-finest-u 8192 --grid-finest-z 256
      --grid-hidden 64   --grid-mlp-layers 2
      --encoder-lr-mult 100.0
    ) ;;
  G1_finer_u)
    # 2× finer u resolution: 8192 → 16384. The strip has ≈143k effective px,
    # so even 16384 is still a 9× compression; might still be undersized.
    ARCH_FLAGS+=(
      --model-type grid
      --grid-n-levels 8 --grid-features 2
      --grid-base-u 64   --grid-base-z 16
      --grid-finest-u 16384 --grid-finest-z 256
      --grid-hidden 64   --grid-mlp-layers 2
      --encoder-lr-mult 100.0
    ) ;;
  G1_richer_feat)
    # 2× features per level (2 → 4). More expressive per-cell signal at the
    # cost of 2× encoder memory.
    ARCH_FLAGS+=(
      --model-type grid
      --grid-n-levels 8 --grid-features 4
      --grid-base-u 64   --grid-base-z 16
      --grid-finest-u 8192 --grid-finest-z 256
      --grid-hidden 64   --grid-mlp-layers 2
      --encoder-lr-mult 100.0
    ) ;;
  G1_deeper_mlp)
    # Wider+deeper MLP head: 64 → 128, 2 → 3 layers. Tests whether the head
    # is the bottleneck rather than the encoder.
    ARCH_FLAGS+=(
      --model-type grid
      --grid-n-levels 8 --grid-features 2
      --grid-base-u 64   --grid-base-z 16
      --grid-finest-u 8192 --grid-finest-z 256
      --grid-hidden 128   --grid-mlp-layers 3
      --encoder-lr-mult 100.0
    ) ;;
  G1_max)
    # All three knobs together: finer_u=16384, features=4, deeper head.
    # ≈4× encoder params, ≈4× head FLOPs. Upper-bound capacity test.
    ARCH_FLAGS+=(
      --model-type grid
      --grid-n-levels 8 --grid-features 4
      --grid-base-u 64   --grid-base-z 16
      --grid-finest-u 16384 --grid-finest-z 256
      --grid-hidden 128   --grid-mlp-layers 3
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
    echo "Unknown variant '${VARIANT}'. Valid: F1_sigma40 F1_sigma80 G1_grid H1_hash S2_wire S3_finer"
    exit 1 ;;
esac

echo "============================================================"
echo "  Architecture smoke: ${VARIANT}  (preset=${PRESET})"
echo "  steps=${STEPS}  ramp=${RAMP}  lr=${LR}  R1S supervised throughout"
echo "  arch flags: ${ARCH_FLAGS[*]}"
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

echo "=== arch smoke ${VARIANT} done at $(date) ==="
