#!/bin/bash
# =============================================================================
# SLURM Job Script — MONAI Prediction (UNet3D + SwinUNETR)
# =============================================================================
# Generates validation predictions from best checkpoints for both models.
# Usage:
#   sbatch Scripts/slurm_predict_monai.sh
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-daily
#SBATCH --job-name=monai-predict
#SBATCH --output=logs/predict_monai_%j.out
#SBATCH --error=logs/predict_monai_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# ─── Project paths ───────────────────────────────────────────────────────────
PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
DATASET_NAME="Dataset502_MickeyScroll3D"

mkdir -p logs

# ─── Activate conda ─────────────────────────────────────────────────────────
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh"
else
    module purge 2>/dev/null
    module load anaconda 2>/dev/null || module load miniconda 2>/dev/null || true
fi
conda activate "${CONDA_ENV}"

# ─── System info ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "  MONAI Prediction — Merlin7"
echo "============================================================"
echo "  Job ID:   ${SLURM_JOB_ID}"
echo "  Node:     ${SLURM_NODELIST}"
echo "  Started:  $(date)"
echo "============================================================"

python -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
"

# Use persistent storage directly (inference is fast, no need for scratch)
export nnUNet_raw="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
export MONAI_RESULTS="${PROJECT_DIR}/monai_results"
export PROJECT_DIR="${PROJECT_DIR}"

# ─── Generate predictions ───────────────────────────────────────────────────
echo ""
echo "=== UNet3D predictions (fold 0) ==="
srun python "${PROJECT_DIR}/Scripts/predict_monai.py" \
    --model unet3d \
    --dataset "${DATASET_NAME}" \
    --fold 0 \
    --patch_size 16 192 192

echo ""
echo "=== SwinUNETR predictions (fold 0) ==="
srun python "${PROJECT_DIR}/Scripts/predict_monai.py" \
    --model swinunetr \
    --dataset "${DATASET_NAME}" \
    --fold 0 \
    --patch_size 32 192 192

echo ""
echo "============================================================"
echo "  Prediction complete at $(date)"
echo "============================================================"
