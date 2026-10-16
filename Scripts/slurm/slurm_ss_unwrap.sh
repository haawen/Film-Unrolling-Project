#!/bin/bash
# =============================================================================
# SLURM Job Script — Self-Supervised 3D Unwrapping
# =============================================================================
# No algorithmic UV labels. Model discovers the mapping from geometric losses:
#   eikonal (constant |∇u|), boundary anchoring, radial ordering, coverage.
#
# Data loading: ~2 min for 200 slices (center computed once, no pin_memory).
# Training: ~90 min for 20K steps with eikonal autograd.
# Total: ~2 hours (well within 14h limit).
#
# Usage:
#   sbatch Scripts/slurm/slurm_ss_unwrap.sh
# Submit from: /data/user/$USER/M_thesis
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-daily
#SBATCH --job-name=ss-unwrap
#SBATCH --output=logs/ss_unwrap_%j.out
#SBATCH --error=logs/ss_unwrap_%j.err
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
OUT_DIR="${PROJECT_DIR}/unwrapping/inr/results/ss_unwrap"

# ─── Scratch setup ──────────────────────────────────────────────────────────
SCRATCH_DIR="/scratch/${USER}/ss_unwrap_${SLURM_JOB_ID}"
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
echo "  Self-Supervised 3D Unwrapping — Merlin7 A100"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Node:       ${SLURM_NODELIST}"
echo "  Started:    $(date)"
echo "  Approach:   NO UV labels — eikonal + boundary + ordering"
echo "  Data:       200 slices (memory-safe)"
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
    --w-eikonal 1.0 \
    --w-boundary 10.0 \
    --w-order 1.0 \
    --w-coverage 0.1 \
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
echo "  Self-supervised unwrapping complete at $(date)"
echo "============================================================"
