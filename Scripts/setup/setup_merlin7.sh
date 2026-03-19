#!/bin/bash
# =============================================================================
# One-time setup script for nnU-Net on Merlin7
# =============================================================================
# Run this ONCE after transferring your project to Merlin7.
# It creates the conda environment and installs all dependencies.
#
# Usage:
#   ssh merlin7
#   cd ~/M_thesis
#   bash Scripts/setup/setup_merlin7.sh
# =============================================================================

set -e  # Exit on any error

PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"

echo "============================================================"
echo "  nnU-Net v2 — Merlin7 Environment Setup"
echo "============================================================"

# ─── 1. Ensure conda is available ───────────────────────────────────────────
echo ""
echo "[1/5] Checking conda..."

# Source conda shell hooks (needed for non-interactive shells / scripts)
CONDA_SH=""
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh" ]; then
    CONDA_SH="/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh"
fi

if [ -n "$CONDA_SH" ]; then
    source "$CONDA_SH"
elif ! command -v conda &> /dev/null; then
    # Try module system as last resort
    echo "  No local conda found. Loading system anaconda module..."
    module load anaconda 2>/dev/null || true
    # After module load, find and source conda.sh for activate support
    CONDA_PREFIX_FOUND=$(conda info --base 2>/dev/null)
    if [ -n "$CONDA_PREFIX_FOUND" ] && [ -f "${CONDA_PREFIX_FOUND}/etc/profile.d/conda.sh" ]; then
        source "${CONDA_PREFIX_FOUND}/etc/profile.d/conda.sh"
    fi
fi

if ! command -v conda &> /dev/null; then
    echo "  ERROR: conda not found. Please install Miniforge:"
    echo "    wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
    echo "    bash Miniforge3-Linux-x86_64.sh -b -p \$HOME/miniconda3"
    echo "    source \$HOME/miniconda3/etc/profile.d/conda.sh"
    echo "    conda init bash"
    exit 1
fi

echo "  conda: $(conda --version)"

# ─── 2. Create conda environment ────────────────────────────────────────────
echo ""
echo "[2/5] Creating conda environment '${CONDA_ENV}' (Python 3.12)..."

if conda env list | grep -q "^${CONDA_ENV} "; then
    echo "  Environment '${CONDA_ENV}' already exists. Skipping creation."
    echo "  (To recreate: conda env remove -n ${CONDA_ENV} && bash $0)"
else
    conda create -n "${CONDA_ENV}" python=3.12 -y
    echo "  Environment created."
fi

conda activate "${CONDA_ENV}"
echo "  Python: $(python --version)"
echo "  Pip: $(pip --version)"

# ─── 3. Install PyTorch with CUDA ───────────────────────────────────────────
echo ""
echo "[3/5] Installing PyTorch with CUDA 12.4..."

# A100 GPUs support CUDA 12.x; PyTorch cu124 is compatible
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

python -c "
import torch
print(f'  PyTorch {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
"

# ─── 4. Install nnU-Net v2 + dependencies ───────────────────────────────────
echo ""
echo "[4/5] Installing nnU-Net v2 and dependencies..."

pip install nnunetv2 h5py scikit-image tifffile tqdm

# Verify
python -c "
from importlib.metadata import version
print(f'  nnU-Net v2: {version(\"nnunetv2\")}')
"

# ─── 5. Create directory structure ──────────────────────────────────────────
echo ""
echo "[5/5] Setting up directory structure..."

mkdir -p "${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
mkdir -p "${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
mkdir -p "${PROJECT_DIR}/nnUNet_data/nnUNet_results"
mkdir -p "${PROJECT_DIR}/logs"

# Install custom trainer
NNUNET_PKG=$(python -c "import nnunetv2; print(nnunetv2.__path__[0])")
TRAINER_SRC="${PROJECT_DIR}/Scripts/custom_trainer.py"
TRAINER_DST="${NNUNET_PKG}/training/nnUNetTrainer/variants/custom_trainer.py"

if [ -f "$TRAINER_SRC" ]; then
    cp "$TRAINER_SRC" "$TRAINER_DST"
    echo "  Custom trainer installed."
fi

echo "  Directories created."

# ─── Done ────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Setup complete!"
echo "============================================================"
echo ""
echo "  Next steps:"
echo "    1. Transfer your data:"
echo "       scp -r nnUNet_data/nnUNet_raw/Dataset501_MickeyScroll \\"
echo "         merlin7:~/M_thesis/nnUNet_data/nnUNet_raw/"
echo ""
echo "    2. Run preprocessing (interactive GPU node):"
echo "       salloc --cluster=gmerlin7 --partition=a100-interactive --gres=gpu:1 --time=01:00:00"
echo "       conda activate nnunet"
echo "       export nnUNet_raw=~/M_thesis/nnUNet_data/nnUNet_raw"
echo "       export nnUNet_preprocessed=~/M_thesis/nnUNet_data/nnUNet_preprocessed"
echo "       export nnUNet_results=~/M_thesis/nnUNet_data/nnUNet_results"
echo "       python -c \"from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import plan_and_preprocess_entry; plan_and_preprocess_entry()\" -d 501 --verify_dataset_integrity -c 2d --clean"
echo ""
echo "    3. Submit training job:"
echo "       sbatch Scripts/slurm/slurm_train.sh 0     # single fold"
echo "       sbatch Scripts/slurm/slurm_train.sh        # all 5 folds"
echo ""
