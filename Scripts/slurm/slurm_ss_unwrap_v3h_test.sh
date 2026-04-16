#!/bin/bash
# =============================================================================
# Test v3h — contrastive angular diversity (w=500, alpha=5, aggressive)
# If w=100 isn't enough to overcome the radial basin, try w=500.
# At step 1000+ total loss is ~1200, so w=500 * 1.0 = 500 = 42% of total.
#
# Usage:
#   sbatch Scripts/slurm/slurm_ss_unwrap_v3h_test.sh
# =============================================================================

#SBATCH --cluster=gmerlin7
#SBATCH --partition=a100-hourly
#SBATCH --job-name=ss-v3h-t
#SBATCH --output=logs/ss_v3h_test_%j.out
#SBATCH --error=logs/ss_v3h_test_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

PROJECT_DIR="$HOME/M_thesis"
CONDA_ENV="nnunet"
mkdir -p logs

DATA_DIR="${PROJECT_DIR}/01_Mickey_3d"
OUT_DIR="${PROJECT_DIR}/unwrapping/inr/results/ss_unwrap_v3h_test"

if [ -f "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

echo "============================================================"
echo "  v3h TEST: contrastive angular, w=500 alpha=5"
echo "============================================================"
echo "  Job ID:     ${SLURM_JOB_ID}"
echo "  Started:    $(date)"
echo "============================================================"

python -u -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
if torch.cuda.is_available():
    print(f'  GPU:      {torch.cuda.get_device_name(0)}')
"

cd "${PROJECT_DIR}"
mkdir -p "${OUT_DIR}"

srun python -u -m unwrapping.inr.ss_train \
    --data-dir "${DATA_DIR}" \
    --out-dir "${OUT_DIR}" \
    --max-slices 5 \
    --n-fourier 512 \
    --sigma 10.0 \
    --hidden-dim 512 \
    --n-layers 6 \
    --steps 3000 \
    --batch-size 65536 \
    --n-pairs 8192 \
    --lr 1e-3 \
    --warmup-steps 500 \
    --ramp-steps 500 \
    --w-eikonal 0.5 \
    --w-boundary 50.0 \
    --w-order 1.0 \
    --w-coverage 1.0 \
    --w-angular 500.0 \
    --angular-alpha 5.0 \
    --w-z-smooth 0.1 \
    --log-every 100 \
    --save-every 500

echo ""
echo "=== v3h test complete at $(date) ==="
