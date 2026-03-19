 #!/bin/bash
# =============================================================================
# SLURM Job Script — nnU-Net v2 Training on Merlin7 (A100 GPU)
# =============================================================================
# Usage:
#   sbatch Scripts/slurm/slurm_train.sh              # Train all 5 folds
#   sbatch Scripts/slurm/slurm_train.sh 0            # Train fold 0 only
#   sbatch Scripts/slurm/slurm_train.sh 0 --c        # Continue training fold 0
#
# Submit from your project root: /data/user/$USER/M_thesis
#
# Storage policy (Merlin7 Code of Conduct):
#   - Preprocessed data is copied to /scratch at job start (fast local I/O)
#   - Training results are written to /scratch during training
#   - Results are copied back to /data/user at job end
#   - /scratch is cleaned up on exit
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
DATASET_NAME="Dataset501_MickeyScroll"

# Persistent data (on /data/user — for storage, not I/O during training)
HOME_RAW="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
HOME_PREPROCESSED="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
HOME_RESULTS="${PROJECT_DIR}/nnUNet_data/nnUNet_results"

# Scratch workspace (fast local I/O during training — Merlin7 policy)
SCRATCH_DIR="/scratch/${USER}/nnunet_${SLURM_JOB_ID}"
SCRATCH_RAW="${SCRATCH_DIR}/nnUNet_raw"
SCRATCH_PREPROCESSED="${SCRATCH_DIR}/nnUNet_preprocessed"
SCRATCH_RESULTS="${SCRATCH_DIR}/nnUNet_results"

# ─── Cleanup function (always runs, even on failure/cancellation) ────────────
cleanup() {
    echo ""
    echo "─── Cleanup ───────────────────────────────────────────────"
    # Copy results from scratch back to home before deleting
    if [ -d "${SCRATCH_RESULTS}/${DATASET_NAME}" ]; then
        echo "  Copying results from /scratch back to /data/user..."
        mkdir -p "${HOME_RESULTS}"
        rsync -a "${SCRATCH_RESULTS}/${DATASET_NAME}/" "${HOME_RESULTS}/${DATASET_NAME}/"
        echo "  Results saved to: ${HOME_RESULTS}/${DATASET_NAME}"
    fi
    # Clean up scratch (mandatory per Merlin7 Code of Conduct)
    if [ -d "${SCRATCH_DIR}" ]; then
        echo "  Cleaning up /scratch..."
        rm -rf "${SCRATCH_DIR}"
        echo "  Scratch cleaned."
    fi
    echo "─────────────────────────────────────────────────────────"
}
trap cleanup EXIT

# ─── Set up scratch workspace ────────────────────────────────────────────────
echo "Setting up /scratch workspace..."
mkdir -p "${SCRATCH_RAW}" "${SCRATCH_PREPROCESSED}" "${SCRATCH_RESULTS}"

# Copy data to scratch for fast I/O
if [ -d "${HOME_RAW}/${DATASET_NAME}" ]; then
    echo "  Copying raw dataset to /scratch..."
    rsync -a "${HOME_RAW}/${DATASET_NAME}/" "${SCRATCH_RAW}/${DATASET_NAME}/"
fi

if [ -d "${HOME_PREPROCESSED}/${DATASET_NAME}" ]; then
    echo "  Copying preprocessed data to /scratch..."
    rsync -a "${HOME_PREPROCESSED}/${DATASET_NAME}/" "${SCRATCH_PREPROCESSED}/${DATASET_NAME}/"
fi

# If continuing training, copy existing results to scratch
if [ -n "$CONTINUE_FLAG" ] && [ -d "${HOME_RESULTS}/${DATASET_NAME}" ]; then
    echo "  Copying existing results to /scratch (for --c)..."
    rsync -a "${HOME_RESULTS}/${DATASET_NAME}/" "${SCRATCH_RESULTS}/${DATASET_NAME}/"
fi

# Point nnU-Net at scratch directories
export nnUNet_raw="${SCRATCH_RAW}"
export nnUNet_preprocessed="${SCRATCH_PREPROCESSED}"
export nnUNet_results="${SCRATCH_RESULTS}"
export nnUNet_n_proc_DA=12

# ─── Activate conda environment ─────────────────────────────────────────────
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
echo "  Scratch:      ${SCRATCH_DIR}"
echo "  Started:      $(date)"
echo "============================================================"

python -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
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
    for f in 0 1 2 3 4; do
        echo ""
        echo ">>> Training fold ${f}/4 ..."
        CMD="python -c \"from nnunetv2.run.run_training import run_training_entry; run_training_entry()\" \
            ${DATASET_ID} ${CONFIG} ${f} -tr ${TRAINER} --npz"

        if [ -n "$CONTINUE_FLAG" ]; then
            CMD="${CMD} --c"
        fi

        eval srun $CMD

        # Copy intermediate results back after each fold
        echo "  Syncing fold ${f} results to /data/user..."
        mkdir -p "${HOME_RESULTS}"
        rsync -a "${SCRATCH_RESULTS}/${DATASET_NAME}/" "${HOME_RESULTS}/${DATASET_NAME}/"

        echo ">>> Fold ${f} finished at $(date)"
    done
else
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
echo "  Results will be copied to: ${HOME_RESULTS}"
echo "============================================================"
