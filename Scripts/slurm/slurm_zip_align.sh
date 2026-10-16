#!/bin/bash
# Pin the OLD -> NEW-reconstruction transform (scale/rotation/centre) by aligning
# the DELIVERED winding curves to the uploaded zip slices, + the acceptance test.
#
#   sbatch Scripts/slurm/slurm_zip_align.sh
#
# CPU only (default merlin7 cluster). The 2 GB matched_walks.npz load dominates.
#SBATCH --job-name=zip_align
#SBATCH --partition=hourly
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/zip_align_%j.out

set -e
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd /data/user/li_k1/M_thesis
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

python -u Scripts/diag_zip_align_curves.py \
    --geom unwrapping/inr/results/walk_full_final/matched_walks.npz \
    --zip-dir 01_Mickey_sprockets \
    --out-dir unwrapping/inr/results/zip_align \
    --zip-z 480 500 520 1940 1960 \
    --s0 1.2515 --theta0 -63

echo "DONE"
