#!/bin/bash
# =============================================================================
# SLURM Test Script — Quick nnU-Net Smoke Test on Merlin7 (A100 GPU)
# =============================================================================
# Runs a minimal 2-epoch training to verify:
#   - Conda environment works
#   - GPU is accessible
#   - Dataset is found and readable
#   - Custom trainer loads properly
#   - nnU-Net training loop starts and completes without errors
#
# Usage:
#   cd ~/M_thesis
#   sbatch Scripts/slurm_test.sh
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-hourly
#SBATCH --job-name=nnunet-test
#SBATCH --output=logs/nnunet_test_%j.out
#SBATCH --error=logs/nnunet_test_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# ─── Create log directory ────────────────────────────────────────────────────
mkdir -p logs

# ─── Project paths ───────────────────────────────────────────────────────────
PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"

# nnU-Net environment variables (persistent data stays in /data/user)
export nnUNet_raw="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_DIR}/nnUNet_data/nnUNet_results"
export nnUNet_n_proc_DA=4

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

# ─── Print environment info ─────────────────────────────────────────────────
echo "============================================================"
echo "  nnU-Net v2 — SMOKE TEST"
echo "============================================================"
echo "  Job ID:       ${SLURM_JOB_ID}"
echo "  Node:         ${SLURM_NODELIST}"
echo "  Partition:    ${SLURM_JOB_PARTITION}"
echo "  Started:      $(date)"
echo "============================================================"

echo ""
echo "[1/5] Checking Python & PyTorch..."
python -c "
import torch
print(f'  Python:   {__import__(\"sys\").version.split()[0]}')
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  ERROR: No GPU detected!')
    exit(1)
"
if [ $? -ne 0 ]; then echo "FAILED: PyTorch/CUDA check"; exit 1; fi

echo ""
echo "[2/5] Checking nnU-Net installation..."
python -c "
from importlib.metadata import version
print(f'  nnU-Net v2: {version(\"nnunetv2\")}')
import nnunetv2
print(f'  Package:    {nnunetv2.__path__[0]}')
"
if [ $? -ne 0 ]; then echo "FAILED: nnU-Net import"; exit 1; fi

echo ""
echo "[3/5] Checking dataset..."
python -c "
import os
raw = os.environ['nnUNet_raw']
ds = os.path.join(raw, 'Dataset501_MickeyScroll')
print(f'  Raw dir:    {raw}')
print(f'  Dataset:    {ds}')
print(f'  Exists:     {os.path.isdir(ds)}')
if os.path.isdir(ds):
    imgs = os.path.join(ds, 'imagesTr')
    lbls = os.path.join(ds, 'labelsTr')
    n_img = len(os.listdir(imgs)) if os.path.isdir(imgs) else 0
    n_lbl = len(os.listdir(lbls)) if os.path.isdir(lbls) else 0
    print(f'  Images:     {n_img}')
    print(f'  Labels:     {n_lbl}')
    dj = os.path.join(ds, 'dataset.json')
    print(f'  dataset.json: {os.path.isfile(dj)}')
else:
    print('  ERROR: Dataset not found! Transfer it first.')
    exit(1)
"
if [ $? -ne 0 ]; then echo "FAILED: Dataset check"; exit 1; fi

echo ""
echo "[4/5] Installing custom trainer..."
TRAINER_SRC="${PROJECT_DIR}/Scripts/custom_trainer.py"
NNUNET_PKG=$(python -c "import nnunetv2; print(nnunetv2.__path__[0])")
TRAINER_DST="${NNUNET_PKG}/training/nnUNetTrainer/variants/custom_trainer.py"
if [ -f "$TRAINER_SRC" ]; then
    cp "$TRAINER_SRC" "$TRAINER_DST"
    echo "  Custom trainer installed: ${TRAINER_DST}"
fi

# Verify the trainer can be imported
python -c "
from nnunetv2.training.nnUNetTrainer.variants.custom_trainer import nnUNetTrainerProgress
print(f'  Trainer class: {nnUNetTrainerProgress.__name__}')
print(f'  Default epochs: {nnUNetTrainerProgress.__init__.__code__.co_varnames}')
print('  Import OK!')
"
if [ $? -ne 0 ]; then echo "FAILED: Custom trainer import"; exit 1; fi

echo ""
echo "[5/5] Running 2-epoch training test (fold 0)..."
echo "============================================================"

# Monkey-patch run_training (not __init__) to limit to 2 epochs
python -c "
from nnunetv2.training.nnUNetTrainer.variants.custom_trainer import nnUNetTrainerProgress

_orig_run = nnUNetTrainerProgress.run_training
def _test_run(self):
    self.num_epochs = 2
    self.save_every = 1
    return _orig_run(self)
nnUNetTrainerProgress.run_training = _test_run

from nnunetv2.run.run_training import run_training_entry
run_training_entry()
" 501 2d 0 -tr nnUNetTrainerProgress --npz

EXIT_CODE=$?
echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  SMOKE TEST PASSED!"
    echo "  Training loop completed 2 epochs without errors."
    echo "  You can now safely run the full training:"
    echo "    sbatch Scripts/slurm_train.sh 0"
else
    echo "  SMOKE TEST FAILED (exit code: $EXIT_CODE)"
    echo "  Check logs/nnunet_test_${SLURM_JOB_ID}.err for details."
fi
echo "  Finished: $(date)"
echo "============================================================"

exit $EXIT_CODE
