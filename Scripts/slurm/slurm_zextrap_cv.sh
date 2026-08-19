#!/bin/bash
# Step 1 of the SPROCKET extension: can the walked winding geometry be
# extrapolated ~350 slices in z into the perforation bands, or must those bands be
# segmented and walked? Cross-validates on the ALREADY-DELIVERED full-density
# geometry (walk_full_final/matched_walks.npz) -- no new data, no labels.
#
#   sbatch Scripts/slurm/slurm_zextrap_cv.sh [holdout]
#
# CPU-only (default merlin7 cluster). ~2GB npz load dominates the runtime.
#SBATCH --job-name=zextrap_cv
#SBATCH --partition=hourly
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/zextrap_cv_%j.out

set -e
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd /data/user/li_k1/M_thesis
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

HOLD=${1:-350}

python -u Scripts/diag_z_extrap_cv.py \
    --geom unwrapping/inr/results/walk_full_final/matched_walks.npz \
    --out-dir unwrapping/inr/results/zextrap_cv \
    --holdout "$HOLD" \
    --fit-windows 100 200 400 \
    --degrees 0 1 2

echo "DONE"
