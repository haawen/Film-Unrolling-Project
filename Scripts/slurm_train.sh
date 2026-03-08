#!/bin/bash
# =============================================================================
# SLURM Job Script — nnU-Net v2 Training on Merlin7 (A100 GPU)
# =============================================================================
# Usage:
#   sbatch Scripts/slurm_train.sh              # Train all 5 folds
#   sbatch Scripts/slurm_train.sh 0            # Train fold 0 only
#   sbatch Scripts/slurm_train.sh 0 --c        # Continue training fold 0
#
# Submit from your project root: /data/user/$USER/M_thesis
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-daily
#SBATCH --job-name=nnunet-mickey
#SBATCH --output=logs/nnunet_%j.out
#SBATCH --error=logs/nnunet_%j.err
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --hint=multithread

# ─── Parse arguments ─────────────────────────────────────────────────────────
FOLD=${1:-"all"}
CONTINUE_FLAG=${2:-""}

# ─── Create log directory ────────────────────────────────────────────────────
mkdir -p logs

# ─── Project paths ───────────────────────────────────────────────────────────
PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"

# nnU-Net environment variables
export nnUNet_raw="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_DIR}/nnUNet_data/nnUNet_results"
export nnUNet_n_proc_DA=12

# ─── Activate conda environment ─────────────────────────────────────────────
# Merlin7 uses miniconda/anaconda in user home
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/.conda/etc/profile.d/conda.sh" ]; then
    source "$HOME/.conda/etc/profile.d/conda.sh"
else
    # Try module system as fallback
    module purge 2>/dev/null
    module load anaconda 2>/dev/null || module load miniconda 2>/dev/null || true
fi

conda activate "${CONDA_ENV}"

# ─── Verify GPU is available ────────────────────────────────────────────────
echo "============================================================"
echo "  nnU-Net v2 Training — Merlin7 A100"
echo "============================================================"
echo "  Job ID:       ${SLURM_JOB_ID}"
echo "  Node:         ${SLURM_NODELIST}"
echo "  GPUs:         ${SLURM_GPUS_ON_NODE:-1}"
echo "  CPUs:         ${SLURM_CPUS_PER_TASK}"
echo "  Partition:    ${SLURM_JOB_PARTITION}"
echo "  Fold:         ${FOLD}"
echo "  Continue:     ${CONTINUE_FLAG:-no}"
echo "  Started:      $(date)"
echo "============================================================"

python -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:     {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

# ─── Install custom trainer into nnU-Net package ────────────────────────────
TRAINER_SRC="${PROJECT_DIR}/Scripts/custom_trainer.py"
NNUNET_PKG=$(python -c "import nnunetv2; print(nnunetv2.__path__[0])")
TRAINER_DST="${NNUNET_PKG}/training/nnUNetTrainer/variants/custom_trainer.py"

if [ -f "$TRAINER_SRC" ]; then
    cp "$TRAINER_SRC" "$TRAINER_DST"
    echo "  Custom trainer installed: ${TRAINER_DST}"
fi

# ─── Dataset configuration ──────────────────────────────────────────────────
DATASET_ID=501
CONFIG="2d"
TRAINER="nnUNetTrainerProgress"

# ─── Run training ───────────────────────────────────────────────────────────
echo ""
echo "Starting training..."
echo "============================================================"

if [ "$FOLD" = "all" ]; then
    # Train all 5 folds sequentially
    for f in 0 1 2 3 4; do
        echo ""
        echo ">>> Training fold ${f}/4 ..."
        CMD="python -c \"from nnunetv2.run.run_training import run_training_entry; run_training_entry()\" \
            ${DATASET_ID} ${CONFIG} ${f} -tr ${TRAINER} --npz"

        if [ -n "$CONTINUE_FLAG" ]; then
            CMD="${CMD} --c"
        fi

        eval srun $CMD

        echo ">>> Fold ${f} finished at $(date)"
    done
else
    # Train single fold
    CMD="python -c \"from nnunetv2.run.run_training import run_training_entry; run_training_entry()\" \
        ${DATASET_ID} ${CONFIG} ${FOLD} -tr ${TRAINER} --npz"

    if [ -n "$CONTINUE_FLAG" ]; then
        CMD="${CMD} --c"
    fi

    eval srun $CMD
fi

echo ""
echo "============================================================"
echo "  Training complete at $(date)"
echo "  Results: ${nnUNet_results}"
echo "============================================================"
