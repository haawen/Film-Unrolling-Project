#!/bin/bash
# Refine the OLD -> NEW transform around the coarse solution (mirror=True,
# theta~137, s~1.25) at high resolution, and settle the z-direction convention.
#
#   sbatch Scripts/slurm/slurm_zip_refine.sh
#
#SBATCH --job-name=zip_refine
#SBATCH --partition=hourly
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/zip_refine_%j.out

set -e
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd /data/user/li_k1/M_thesis
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# old z848 sits near the LOW end of the kept picture band, so under the
# same-direction assumption it pairs with zip ~z490 and under a flip with ~z1960.
python -u Scripts/diag_zip_refine.py \
    --old-dir 01_Mickey_hdf \
    --zip-dir 01_Mickey_sprockets \
    --out-dir unwrapping/inr/results/zip_refine \
    --old-z 848 --zip-z-same 490 --zip-z-flip 1960 \
    --theta 137 --theta-window 8 --theta-step 0.5 \
    --scales 1.235 1.265 0.005 \
    --work 2000

echo "DONE"
