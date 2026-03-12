#!/bin/bash
# =============================================================================
# SLURM Job — nnU-Net preprocessing for Dataset502 (3D)
# =============================================================================
# Usage:  sbatch Scripts/slurm_preprocess_3d.sh
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-hourly
#SBATCH --job-name=preproc-502
#SBATCH --output=logs/preproc502_%j.out
#SBATCH --error=logs/preproc502_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:0

mkdir -p logs

PROJECT_DIR="$HOME/M_thesis"

export nnUNet_raw="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_DIR}/nnUNet_data/nnUNet_results"

# ─── Activate conda ─────────────────────────────────────────────────────────
if [ -f "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh"
else
    module purge 2>/dev/null
    module load anaconda 2>/dev/null || true
fi
conda activate nnunet

echo "============================================================"
echo "  nnU-Net Preprocessing — Dataset502_MickeyScroll3D"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  CPUs:       ${SLURM_CPUS_PER_TASK}"
echo "  Memory:     128G"
echo "  Started:    $(date)"
echo "============================================================"

# Install custom trainer first (needed for plan_and_preprocess to find it)
TRAINER_SRC="${PROJECT_DIR}/Scripts/custom_trainer.py"
NNUNET_PKG=$(python -c "import nnunetv2; print(nnunetv2.__path__[0])")
TRAINER_DST="${NNUNET_PKG}/training/nnUNetTrainer/variants/custom_trainer.py"
if [ -f "$TRAINER_SRC" ]; then
    cp "$TRAINER_SRC" "$TRAINER_DST"
    echo "  Custom trainer installed."
fi

echo ""
echo "Running plan_and_preprocess..."
srun nnUNetv2_plan_and_preprocess -d 502 --verify_dataset_integrity -c 3d_fullres --clean

echo ""
echo "============================================================"
echo "  Preprocessing complete at $(date)"
echo "============================================================"
echo ""
echo "Preprocessed data:"
ls -lh "${nnUNet_preprocessed}/Dataset502_MickeyScroll3D/"
