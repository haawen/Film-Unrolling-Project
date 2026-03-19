#!/bin/bash
# =============================================================================
# SLURM Job — Create Dataset503 (2D slices from 3D), preprocess, and train
# =============================================================================
# Usage:
#   sbatch Scripts/slurm_train_2d_from_3d.sh              # Train fold 0
#   sbatch Scripts/slurm_train_2d_from_3d.sh 0 --c        # Continue fold 0
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-daily
#SBATCH --job-name=nnunet2d-from3d
#SBATCH --output=logs/nnunet2d_from3d_%j.out
#SBATCH --error=logs/nnunet2d_from3d_%j.err
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --hint=multithread

# ─── Parse arguments ─────────────────────────────────────────────────────────
FOLD=${1:-0}
CONTINUE_FLAG=${2:-""}

mkdir -p logs

# ─── Project paths ───────────────────────────────────────────────────────────
PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
DATASET_NAME="Dataset503_MickeyScroll2Dfrom3D"
DATASET_ID=503

# Persistent data
HOME_RAW="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
HOME_PREPROCESSED="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
HOME_RESULTS="${PROJECT_DIR}/nnUNet_data/nnUNet_results"

# Scratch workspace
SCRATCH_DIR="/scratch/${USER}/nnunet2d_from3d_${SLURM_JOB_ID}"
SCRATCH_RAW="${SCRATCH_DIR}/nnUNet_raw"
SCRATCH_PREPROCESSED="${SCRATCH_DIR}/nnUNet_preprocessed"
SCRATCH_RESULTS="${SCRATCH_DIR}/nnUNet_results"

# ─── Cleanup function ────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "─── Cleanup ───────────────────────────────────────────────"
    if [ -d "${SCRATCH_RESULTS}/${DATASET_NAME}" ]; then
        echo "  Copying results from /scratch back to /data/user..."
        mkdir -p "${HOME_RESULTS}"
        rsync -a "${SCRATCH_RESULTS}/${DATASET_NAME}/" "${HOME_RESULTS}/${DATASET_NAME}/"
        echo "  Results saved to: ${HOME_RESULTS}/${DATASET_NAME}"
    fi
    if [ -d "${SCRATCH_DIR}" ]; then
        echo "  Cleaning up /scratch..."
        rm -rf "${SCRATCH_DIR}"
        echo "  Scratch cleaned."
    fi
    echo "─────────────────────────────────────────────────────────"
}
trap cleanup EXIT

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

# ─── Install custom trainer ─────────────────────────────────────────────────
TRAINER_SRC="${PROJECT_DIR}/Scripts/custom_trainer.py"
NNUNET_PKG=$(python -c "import nnunetv2; print(nnunetv2.__path__[0])")
TRAINER_DST="${NNUNET_PKG}/training/nnUNetTrainer/variants/custom_trainer.py"
if [ -f "$TRAINER_SRC" ]; then
    cp "$TRAINER_SRC" "$TRAINER_DST"
    echo "  Custom trainer installed."
fi

echo "============================================================"
echo "  nnU-Net 2D (from 3D slices) — Merlin7 A100"
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

# ─── Step 1: Create Dataset503 if it doesn't exist ──────────────────────────
if [ ! -d "${HOME_RAW}/${DATASET_NAME}/imagesTr" ]; then
    echo ""
    echo "=== Creating Dataset503 (2D slices from 3D volumes) ==="
    export PROJECT_DIR="${PROJECT_DIR}"
    srun python "${PROJECT_DIR}/Scripts/create_2d_from_3d.py"
else
    echo "  Dataset503 already exists, skipping creation."
fi

# ─── Set up scratch ─────────────────────────────────────────────────────────
echo ""
echo "Setting up /scratch workspace..."
mkdir -p "${SCRATCH_RAW}" "${SCRATCH_PREPROCESSED}" "${SCRATCH_RESULTS}"

echo "  Copying raw dataset to /scratch..."
rsync -a "${HOME_RAW}/${DATASET_NAME}/" "${SCRATCH_RAW}/${DATASET_NAME}/"

if [ -d "${HOME_PREPROCESSED}/${DATASET_NAME}" ]; then
    echo "  Copying preprocessed data to /scratch..."
    rsync -a "${HOME_PREPROCESSED}/${DATASET_NAME}/" "${SCRATCH_PREPROCESSED}/${DATASET_NAME}/"
fi

if [ -n "$CONTINUE_FLAG" ] && [ -d "${HOME_RESULTS}/${DATASET_NAME}" ]; then
    echo "  Copying existing results to /scratch (for --c)..."
    rsync -a "${HOME_RESULTS}/${DATASET_NAME}/" "${SCRATCH_RESULTS}/${DATASET_NAME}/"
fi

export nnUNet_raw="${SCRATCH_RAW}"
export nnUNet_preprocessed="${SCRATCH_PREPROCESSED}"
export nnUNet_results="${SCRATCH_RESULTS}"
export nnUNet_n_proc_DA=12

# ─── Step 2: Preprocess (if not already done) ───────────────────────────────
if [ ! -d "${SCRATCH_PREPROCESSED}/${DATASET_NAME}/nnUNetPlans_2d" ]; then
    echo ""
    echo "=== Preprocessing Dataset503 ==="
    # Copy splits_final.json before preprocessing so nnU-Net uses our matched splits
    if [ -f "${HOME_RAW}/${DATASET_NAME}/splits_final.json" ]; then
        mkdir -p "${SCRATCH_PREPROCESSED}/${DATASET_NAME}"
        cp "${HOME_RAW}/${DATASET_NAME}/splits_final.json" \
           "${SCRATCH_PREPROCESSED}/${DATASET_NAME}/splits_final.json"
        echo "  Matched splits copied to preprocessed dir."
    fi
    srun nnUNetv2_plan_and_preprocess -d ${DATASET_ID} --verify_dataset_integrity -c 2d --clean

    # Copy preprocessed data back to /data for future runs
    echo "  Saving preprocessed data to /data/user..."
    mkdir -p "${HOME_PREPROCESSED}"
    rsync -a "${SCRATCH_PREPROCESSED}/${DATASET_NAME}/" "${HOME_PREPROCESSED}/${DATASET_NAME}/"
else
    echo "  Dataset503 already preprocessed."
fi

# ─── Step 3: Train ───────────────────────────────────────────────────────────
echo ""
echo "=== Training nnU-Net 2D on Dataset503 ==="
CONFIG="2d"
TRAINER="nnUNetTrainerProgress"

CMD="python -c \"from nnunetv2.run.run_training import run_training_entry; run_training_entry()\" \
    ${DATASET_ID} ${CONFIG} ${FOLD} -tr ${TRAINER} --npz"

if [ -n "$CONTINUE_FLAG" ]; then
    CMD="${CMD} --c"
fi

eval srun $CMD

echo ""
echo "============================================================"
echo "  Training complete at $(date)"
echo "  Results will be copied to: ${HOME_RESULTS}/${DATASET_NAME}"
echo "============================================================"
