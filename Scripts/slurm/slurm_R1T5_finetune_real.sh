#!/bin/bash
# =============================================================================
# Track 5 (Round 9) — fine-tune SS on real Mickey, init from a synthetic-
# supervised checkpoint (typically `mickey_geometry` R1S full).
#
# Usage:
#     INIT_FROM=$HOME/M_thesis/unwrapping/inr/results/R1_full_mickey_geometry/R1S/train/model_final.pt \
#         sbatch Scripts/slurm/slurm_R1T5_finetune_real.sh smoke
#     ... or `full`.
#
# Smoke = 1500 steps a100-hourly, real Mickey only (no GT eval available).
# Full  = 10000 steps a100-daily.
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --job-name=R1T5_FT
#SBATCH --output=logs/R1T5_FT_%j.out
#SBATCH --error=logs/R1T5_FT_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=128G

set -euo pipefail

MODE="${1:?usage: sbatch slurm_R1T5_finetune_real.sh <smoke|full>}"
PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
REAL_DIR="${PROJECT_DIR}/01_Mickey_3d"

if [ -z "${INIT_FROM:-}" ]; then
  echo "ERROR: INIT_FROM env var must point at a synthetic-trained model_final.pt" >&2
  exit 2
fi
if [ ! -f "${INIT_FROM}" ]; then
  echo "ERROR: INIT_FROM checkpoint not found: ${INIT_FROM}" >&2
  exit 2
fi

case "${MODE}" in
  smoke) TOTAL_STEPS=1500 ; RAMP=500  ;;
  full)  TOTAL_STEPS=10000; RAMP=2000 ;;
  *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac

# Track 4: optional patch-feature loss weight via env (W_PF=...).
W_PF="${W_PF:-0.0}"
PF_TAG=""
if awk "BEGIN{exit !(${W_PF} > 0)}"; then
  PF_TAG="_PF${W_PF}"
fi

# Step A: optional pose-tether weight via env (W_TETHER=...).
W_TETHER="${W_TETHER:-0.0}"
TETHER_TAG=""
if awk "BEGIN{exit !(${W_TETHER} > 0)}"; then
  TETHER_TAG="_T${W_TETHER}"
fi

# Step B: optional InfoNCE patch-feature weight (W_INFONCE=...).
W_INFONCE="${W_INFONCE:-0.0}"
INFONCE_TAG=""
if awk "BEGIN{exit !(${W_INFONCE} > 0)}"; then
  INFONCE_TAG="_N${W_INFONCE}"
fi

OUT_BASE="${PROJECT_DIR}/unwrapping/inr/results/R1T5_FT${PF_TAG}${TETHER_TAG}${INFONCE_TAG}_${MODE}"
TRAIN_DIR="${OUT_BASE}/train"
EVAL_DIR="${OUT_BASE}/eval_real"
mkdir -p logs "${TRAIN_DIR}" "${EVAL_DIR}"

if [ -f "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

echo "============================================================"
echo "  Track 5 fine-tune (real Mickey)  /  mode ${MODE}"
echo "============================================================"
echo "  init_from   : ${INIT_FROM}"
echo "  real        : ${REAL_DIR}"
echo "  out         : ${OUT_BASE}"
echo "  total_steps : ${TOTAL_STEPS}"
echo "  ramp_steps  : ${RAMP}"
echo "  w_patch_feat: ${W_PF}"
echo "  w_tether    : ${W_TETHER}"
echo "  w_infonce   : ${W_INFONCE}"
echo "  job id      : ${SLURM_JOB_ID}"
echo "  started     : $(date)"
echo "============================================================"

cd "${PROJECT_DIR}"

# Train: pure SS (R1n recipe) on real, init from synthetic-supervised ckpt.
# max-slices 20 caps GPU memory: real Mickey is 3063×3062 × 500 → would OOM.
echo ""
echo "--- Train: pure SS on real Mickey, init from synthetic ---"
srun python -u -m unwrapping.inr.surface_train \
    --data-dir   "${REAL_DIR}" \
    --out-dir    "${TRAIN_DIR}" \
    --max-slices 20 \
    --attachment emulsion \
    --use-distance-attach \
    --centerline-erode 1 \
    --supervised-steps 0 \
    --steps "${TOTAL_STEPS}" \
    --ramp-steps "${RAMP}" \
    --batch-size 131072 \
    --lr 1e-4 \
    --w-attach 1.0 \
    --w-conformal 0.1 \
    --w-res-reg 0.0 \
    --sigma 20.0 \
    --hidden-dim 256 \
    --n-layers-mlp 4 \
    --model-type inr \
    --log-every 50 \
    --init-from "${INIT_FROM}" \
    $( awk "BEGIN{exit !(${W_PF} > 0)}" && echo "--w-patch-feature-zcoh ${W_PF}" ) \
    $( awk "BEGIN{exit !(${W_TETHER} > 0)}" && echo "--w-pose-tether ${W_TETHER}" ) \
    $( awk "BEGIN{exit !(${W_INFONCE} > 0)}" && echo "--w-patch-feature-infonce ${W_INFONCE}" )

# Eval: render strip on real (no GT metrics — strip_score proxies + visual).
echo ""
echo "--- Eval: render strip on real Mickey ---"
srun python -u -m unwrapping.inr.surface_eval \
    --data-dir   "${REAL_DIR}" \
    --out-dir    "${EVAL_DIR}" \
    --max-slices 20 \
    --attachment emulsion \
    --centerline-erode 1 \
    --ckpt "${TRAIN_DIR}/model_final.pt"

# Run unsupervised strip-score proxies (z_coh, hf_energy, dyn_range).
echo ""
echo "--- Strip-score proxies ---"
srun python -u -m unwrapping.inr.strip_score \
    "${EVAL_DIR}/strip.npz" 2>&1 || \
    echo "(strip_score may not exist or have a different signature; skipping)"

echo ""
echo "=== R1T5 fine-tune ${MODE} complete at $(date) ==="
