#!/bin/bash
# =============================================================================
# SLURM Job Script — Swin UNETR (MONAI) Training on Merlin7 (A100 GPU)
# =============================================================================
# Usage:
#   sbatch Scripts/slurm_train_swinunetr.sh              # Train fold 0
#   sbatch Scripts/slurm_train_swinunetr.sh 0             # Train fold 0
#   sbatch Scripts/slurm_train_swinunetr.sh 0 --resume    # Resume fold 0
#
# PREREQUISITE: 3D NIfTI data in nnUNet_raw/Dataset501_MickeyScroll/
# Submit from your project root: /data/user/$USER/M_thesis
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-daily
#SBATCH --job-name=swinunetr-mickey
#SBATCH --output=logs/swinunetr_%j.out
#SBATCH --error=logs/swinunetr_%j.err
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --hint=multithread

# ─── Parse arguments ─────────────────────────────────────────────────────────
FOLD=${1:-0}
RESUME_FLAG=${2:-""}

# ─── Create log directory ────────────────────────────────────────────────────
mkdir -p logs

# ─── Project paths ───────────────────────────────────────────────────────────
PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
DATASET_NAME="Dataset502_MickeyScroll3D"

# Persistent data (on /data/user)
HOME_RAW="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
HOME_PREPROCESSED="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
HOME_RESULTS="${PROJECT_DIR}/monai_results"

# Scratch workspace (fast local I/O)
SCRATCH_DIR="/scratch/${USER}/swinunetr_${SLURM_JOB_ID}"
SCRATCH_RAW="${SCRATCH_DIR}/nnUNet_raw"
SCRATCH_PREPROCESSED="${SCRATCH_DIR}/nnUNet_preprocessed"
SCRATCH_RESULTS="${SCRATCH_DIR}/monai_results"

# ─── Cleanup function ───────────────────────────────────────────────────────
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

# ─── Set up scratch workspace ────────────────────────────────────────────────
echo "Setting up /scratch workspace..."
mkdir -p "${SCRATCH_RAW}" "${SCRATCH_PREPROCESSED}" "${SCRATCH_RESULTS}"

if [ -d "${HOME_RAW}/${DATASET_NAME}" ]; then
    echo "  Copying raw dataset to /scratch..."
    rsync -a "${HOME_RAW}/${DATASET_NAME}/" "${SCRATCH_RAW}/${DATASET_NAME}/"
fi

SPLITS_SRC="${HOME_PREPROCESSED}/${DATASET_NAME}/splits_final.json"
if [ -f "${SPLITS_SRC}" ]; then
    echo "  Copying splits_final.json to /scratch..."
    mkdir -p "${SCRATCH_PREPROCESSED}/${DATASET_NAME}"
    cp "${SPLITS_SRC}" "${SCRATCH_PREPROCESSED}/${DATASET_NAME}/splits_final.json"
    echo "  Splits file copied successfully."
else
    echo "  ERROR: splits_final.json not found at ${SPLITS_SRC}"
    echo "  Run nnU-Net preprocessing first to generate consistent splits."
    exit 1
fi

if [ -n "$RESUME_FLAG" ] && [ -d "${HOME_RESULTS}/${DATASET_NAME}/SwinUNETR" ]; then
    echo "  Copying existing results to /scratch (for resume)..."
    rsync -a "${HOME_RESULTS}/${DATASET_NAME}/SwinUNETR/" \
        "${SCRATCH_RESULTS}/${DATASET_NAME}/SwinUNETR/"
fi

# Point environment at scratch
export nnUNet_raw="${SCRATCH_RAW}"
export nnUNet_preprocessed="${SCRATCH_PREPROCESSED}"
export MONAI_RESULTS="${SCRATCH_RESULTS}"
export PROJECT_DIR="${PROJECT_DIR}"

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

# Ensure MONAI is installed
python -c "import monai" 2>/dev/null || pip install --quiet "monai[einops,nibabel]"

# ─── System info ─────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Swin UNETR (MONAI) Training — Merlin7 A100"
echo "============================================================"
echo "  Job ID:       ${SLURM_JOB_ID}"
echo "  Node:         ${SLURM_NODELIST}"
echo "  GPUs:         ${SLURM_GPUS_ON_NODE:-1}"
echo "  CPUs:         ${SLURM_CPUS_PER_TASK}"
echo "  Partition:    ${SLURM_JOB_PARTITION}"
echo "  Fold:         ${FOLD}"
echo "  Resume:       ${RESUME_FLAG:-no}"
echo "  Scratch:      ${SCRATCH_DIR}"
echo "  Started:      $(date)"
echo "============================================================"

python -c "
import torch, monai
print(f'  PyTorch:  {torch.__version__}')
print(f'  MONAI:    {monai.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# ─── Run training ───────────────────────────────────────────────────────────
# Swin UNETR needs img_size divisible by 32; use (16, 192, 192) for small Z
echo ""
echo "Starting Swin UNETR training..."
echo "============================================================"

CMD="srun python ${PROJECT_DIR}/Scripts/train_monai.py \
    --model swinunetr \
    --dataset ${DATASET_NAME} \
    --fold ${FOLD} \
    --epochs 250 \
    --batch_size 1 \
    --lr 1e-4 \
    --patch_size 32 256 256 \
    --val_interval 10 \
    --save_every 25 \
    --workers 4"

if [ "$RESUME_FLAG" = "--resume" ]; then
    CMD="${CMD} --resume"
fi

eval $CMD

echo ""
echo "============================================================"
echo "  Swin UNETR training complete at $(date)"
echo "  Results will be copied to: ${HOME_RESULTS}/${DATASET_NAME}/SwinUNETR"
echo "============================================================"
