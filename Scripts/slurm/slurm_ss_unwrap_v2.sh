#!/bin/bash
# =============================================================================
# SLURM Job Script — Self-Supervised 3D Unwrapping (v2: balanced losses)
# =============================================================================
# Changes from v1:
#   - w_boundary: 10 → 50 (prevent eikonal from collapsing u range)
#   - w_eikonal: 1.0 → 0.5 (less aggressive gradient regularization)
#   - Eikonal ramp: linear 0→full over 3000 steps after warmup (not step function)
#   - w_coverage: 0.1 → 1.0 (stronger anti-collapse)
#
# Usage:
#   sbatch Scripts/slurm/slurm_ss_unwrap_v2.sh
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-daily
#SBATCH --job-name=ss-unwrap-v2
#SBATCH --output=logs/ss_unwrap_v2_%j.out
#SBATCH --error=logs/ss_unwrap_v2_%j.err
#SBATCH --time=14:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
mkdir -p logs

DATA_DIR="${PROJECT_DIR}/01_Mickey_3d"
OUT_DIR="${PROJECT_DIR}/unwrapping/inr/results/ss_unwrap_v2"

# ─── Scratch setup ──────────────────────────────────────────────────────────
SCRATCH_DIR="/scratch/${USER}/ss_unwrap_v2_${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_DIR}/data"

cleanup() {
    echo ""
    echo "─── Cleanup ───────────────────────────────────────────────"
    rm -rf "${SCRATCH_DIR}"
    echo "  Scratch cleaned. Results in: ${OUT_DIR}"
    echo "─────────────────────────────────────────────────────────"
}
trap cleanup EXIT

echo "Copying data to /scratch..."
cp "${DATA_DIR}"/volume_*.h5 "${SCRATCH_DIR}/data/"
echo "  $(ls ${SCRATCH_DIR}/data/*.h5 | wc -l) files copied"

# ─── Activate conda ─────────────────────────────────────────────────────────
if [ -f "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

echo "============================================================"
echo "  Self-Supervised 3D Unwrapping v2 — Merlin7 A100"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Node:       ${SLURM_NODELIST}"
echo "  Started:    $(date)"
echo "  Changes:    eikonal ramp + stronger boundary/coverage"
echo "============================================================"

python -u -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
"

cd "${PROJECT_DIR}"
mkdir -p "${OUT_DIR}"

echo ""
echo "═══ Phase 1: Self-supervised training ═══"
srun python -u -m unwrapping.inr.ss_train \
    --data-dir "${SCRATCH_DIR}/data" \
    --out-dir "${OUT_DIR}" \
    --max-slices 200 \
    --n-fourier 512 \
    --sigma 10.0 \
    --hidden-dim 512 \
    --n-layers 6 \
    --steps 20000 \
    --batch-size 131072 \
    --n-pairs 16384 \
    --lr 1e-3 \
    --warmup-steps 2000 \
    --ramp-steps 3000 \
    --w-eikonal 0.5 \
    --w-boundary 50.0 \
    --w-order 1.0 \
    --w-coverage 1.0 \
    --log-every 200 \
    --save-every 2000

echo ""
echo "═══ Phase 2: Evaluation and strip generation ═══"
srun python -u -m unwrapping.inr.ss_eval \
    --data-dir "${SCRATCH_DIR}/data" \
    --model-dir "${OUT_DIR}" \
    --out-dir "${OUT_DIR}/eval" \
    --max-slices 200

echo ""
echo "============================================================"
echo "  Self-supervised unwrapping v2 complete at $(date)"
echo "============================================================"
