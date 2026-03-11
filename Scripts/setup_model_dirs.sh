#!/bin/bash
# =============================================================================
# Create output directories for all models on Merlin7 HPC
# =============================================================================
# Run once from your project root on the cluster:
#   bash Scripts/setup_model_dirs.sh
# =============================================================================

PROJECT_DIR="$HOME/M_thesis"
DATASET="Dataset501_MickeyScroll"

echo "Creating output directories for all models..."
echo "Project: ${PROJECT_DIR}"
echo ""

# ─── nnU-Net results (managed by nnU-Net, but ensure base exists) ────────────
mkdir -p "${PROJECT_DIR}/nnUNet_data/nnUNet_results/${DATASET}"
echo "  [OK] nnUNet_data/nnUNet_results/${DATASET}/"

# ─── MONAI results ──────────────────────────────────────────────────────────
for MODEL in UNet3D SwinUNETR; do
    for FOLD in 0 1 2 3 4; do
        mkdir -p "${PROJECT_DIR}/monai_results/${DATASET}/${MODEL}/fold_${FOLD}/checkpoints"
        mkdir -p "${PROJECT_DIR}/monai_results/${DATASET}/${MODEL}/fold_${FOLD}/validation"
    done
    echo "  [OK] monai_results/${DATASET}/${MODEL}/  (folds 0-4)"
done

# ─── Logs directory ──────────────────────────────────────────────────────────
mkdir -p "${PROJECT_DIR}/logs"
echo "  [OK] logs/"

# ─── Visualizations directory ────────────────────────────────────────────────
mkdir -p "${PROJECT_DIR}/visualizations"
echo "  [OK] visualizations/"

echo ""
echo "All directories created."
echo ""
echo "Model overview:"
echo "  1. nnU-Net 2D  → sbatch Scripts/slurm_train.sh [fold]"
echo "  2. nnU-Net 3D  → sbatch Scripts/slurm_train_3d.sh [fold]"
echo "  3. 3D U-Net    → sbatch Scripts/slurm_train_3dunet.sh [fold]"
echo "  4. Swin UNETR  → sbatch Scripts/slurm_train_swinunetr.sh [fold]"
