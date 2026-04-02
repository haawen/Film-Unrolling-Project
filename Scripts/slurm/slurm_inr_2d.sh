#!/bin/bash
# =============================================================================
# SLURM Job Script — INR Training on 2D CT Slice (Merlin7 A100)
# =============================================================================
# Usage:
#   sbatch Scripts/slurm/slurm_inr_2d.sh                    # without seg
#   sbatch Scripts/slurm/slurm_inr_2d.sh --use-seg          # with seg
#
# Submit from: /data/user/$USER/M_thesis
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-hourly
#SBATCH --job-name=inr-2d
#SBATCH --output=logs/inr_2d_%j.out
#SBATCH --error=logs/inr_2d_%j.err
#SBATCH --time=00:55:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# ─── Parse arguments ────────────────────────────────────────────────────────
USE_SEG_FLAG="${1:-}"

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
mkdir -p logs

# Data files (2D slice — pick first available)
IMAGE="${PROJECT_DIR}/01_Mickey_hdf_subset/01_Mickey_Stitch_Stitch_Export_0835.h5"
PROBS="${PROJECT_DIR}/01_Mickey_hdf_subset/01_Mickey_Stitch_Stitch_Export_0835-image_Probabilities.h5"

# Output directory
if [ "$USE_SEG_FLAG" = "--use-seg" ]; then
    OUT_DIR="${PROJECT_DIR}/unwrapping/inr/results/2d_fourier_seg"
    SEG_ARGS="--use-seg --probs ${PROBS}"
    JOB_DESC="2D + seg"
else
    OUT_DIR="${PROJECT_DIR}/unwrapping/inr/results/2d_fourier_noseg"
    SEG_ARGS=""
    JOB_DESC="2D no-seg"
fi

# ─── Scratch setup ──────────────────────────────────────────────────────────
SCRATCH_DIR="/scratch/${USER}/inr_2d_${SLURM_JOB_ID}"
mkdir -p "${SCRATCH_DIR}"

cleanup() {
    echo ""
    echo "─── Cleanup ───────────────────────────────────────────────"
    if [ -d "${SCRATCH_DIR}/results" ]; then
        echo "  Copying results back to /data/user..."
        mkdir -p "${OUT_DIR}"
        rsync -a "${SCRATCH_DIR}/results/" "${OUT_DIR}/"
        echo "  Results saved to: ${OUT_DIR}"
    fi
    rm -rf "${SCRATCH_DIR}"
    echo "─────────────────────────────────────────────────────────"
}
trap cleanup EXIT

# Copy data to scratch for fast I/O
echo "Copying data to /scratch..."
cp "${IMAGE}" "${SCRATCH_DIR}/image.h5"
if [ -n "${SEG_ARGS}" ]; then
    cp "${PROBS}" "${SCRATCH_DIR}/probs.h5"
fi

# ─── Activate conda ─────────────────────────────────────────────────────────
if [ -f "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

# ─── Info ────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  INR Training — ${JOB_DESC} — Merlin7 A100"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Node:       ${SLURM_NODELIST}"
echo "  Arch:       fourier"
echo "  Seg head:   ${USE_SEG_FLAG:-disabled}"
echo "  Started:    $(date)"
echo "============================================================"

python -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
"

# ─── Build command ───────────────────────────────────────────────────────────
SCRATCH_IMAGE="${SCRATCH_DIR}/image.h5"
SCRATCH_PROBS="${SCRATCH_DIR}/probs.h5"

CMD="srun python -m unwrapping.inr.train \
    --mode 2d \
    --image ${SCRATCH_IMAGE} \
    --arch fourier \
    --hidden-dim 256 \
    --n-layers 4 \
    --n-fourier 256 \
    --sigma 10.0 \
    --steps 5000 \
    --batch-size 262144 \
    --lr 1e-3 \
    --out-dir ${SCRATCH_DIR}/results \
    --log-every 100 \
    --save-every 1000"

if [ "$USE_SEG_FLAG" = "--use-seg" ]; then
    CMD="${CMD} --use-seg --probs ${SCRATCH_PROBS} --seg-weight 0.1"
fi

echo ""
echo "Running: ${CMD}"
echo ""

cd "${PROJECT_DIR}"
eval ${CMD}

# ─── Evaluate ────────────────────────────────────────────────────────────────
echo ""
echo "Running evaluation..."
srun python -m unwrapping.inr.evaluate --model-dir "${SCRATCH_DIR}/results"

echo ""
echo "============================================================"
echo "  INR 2D Training complete at $(date)"
echo "============================================================"
