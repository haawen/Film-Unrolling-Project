#!/bin/bash
# =============================================================================
# SLURM Job — Fair comparison of all models on identical 3D validation data
# =============================================================================
# Prerequisites:
#   - nnU-Net 3D (Dataset502) trained
#   - nnU-Net 2D from 3D (Dataset503) trained
#   - MONAI UNet3D and SwinUNETR trained
#   - (Optional) nnU-Net 2D (Dataset501) for original 2D comparison
#
# Usage:
#   sbatch Scripts/slurm_fair_compare.sh
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-hourly
#SBATCH --job-name=fair-compare
#SBATCH --output=logs/fair_compare_%j.out
#SBATCH --error=logs/fair_compare_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
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

echo "============================================================"
echo "  Fair Comparison — All models on same 3D validation data"
echo "============================================================"
echo "  Job ID:   ${SLURM_JOB_ID}"
echo "  Node:     ${SLURM_NODELIST}"
echo "  Started:  $(date)"
echo "============================================================"

export PROJECT_DIR="${PROJECT_DIR}"
export nnUNet_raw="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_DIR}/nnUNet_data/nnUNet_results"
export MONAI_RESULTS="${PROJECT_DIR}/monai_results"

FAIR_DIR="${PROJECT_DIR}/fair_comparison"
SLICES_IN="${FAIR_DIR}/slices_input"
SLICES_OUT="${FAIR_DIR}/slices_output"

# ─── Step 1: Extract 2D slices from 3D val volumes (for original 2D model) ──
echo ""
echo "=== Step 1: Extracting 2D slices from 3D val volumes ==="
srun python "${PROJECT_DIR}/Scripts/fair_compare.py" --prepare --fold 0

# ─── Step 2: Run original nnU-Net 2D (Dataset501) on extracted slices ────────
echo ""
echo "=== Step 2: Running nnU-Net 2D (Dataset501) inference on val slices ==="
mkdir -p "${SLICES_OUT}"

srun nnUNetv2_predict \
    -i "${SLICES_IN}" \
    -o "${SLICES_OUT}" \
    -d 501 \
    -c 2d \
    -tr nnUNetTrainerProgress \
    -f 0 \
    --disable_tta

# ─── Step 3: Generate MONAI predictions (always regenerate for consistency) ──
echo ""
echo "=== Step 3: Generating MONAI predictions ==="
for MODEL in unet3d swinunetr; do
    MODEL_UPPER=$([ "$MODEL" = "unet3d" ] && echo "UNet3D" || echo "SwinUNETR")
    CKPT="${MONAI_RESULTS}/Dataset502_MickeyScroll3D/${MODEL_UPPER}/fold_0/checkpoints/best.pt"
    if [ -f "${CKPT}" ]; then
        echo "  Running predictions for ${MODEL_UPPER}..."
        srun python "${PROJECT_DIR}/Scripts/predict_monai.py" \
            --model "${MODEL}" --fold 0 --dataset Dataset502_MickeyScroll3D
    else
        echo "  ${MODEL_UPPER}: no checkpoint found, skipping."
    fi
done

# ─── Step 4: Reassemble and compare ─────────────────────────────────────────
echo ""
echo "=== Step 4: Reassembling predictions and running fair comparison ==="
srun python "${PROJECT_DIR}/Scripts/fair_compare.py" --compare --save --fold 0

echo ""
echo "============================================================"
echo "  Fair comparison complete at $(date)"
echo "  Results: ${FAIR_DIR}/fair_comparison_results.json"
echo "  Plots:   ${PROJECT_DIR}/visualizations/"
echo "============================================================"
