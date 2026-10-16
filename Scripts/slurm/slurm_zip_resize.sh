#!/bin/bash
# Test whether the OLD stitch is just the NEW reconstruction RESIZED (+ maybe an
# axis flip): anti-aliased downsample, 8 array symmetries, raw-pixel NCC.
#
#   sbatch Scripts/slurm/slurm_zip_resize.sh
#
#SBATCH --job-name=zip_resize
#SBATCH --partition=hourly
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/zip_resize_%j.out

set -e
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd /data/user/li_k1/M_thesis
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

python -u Scripts/diag_zip_resize_match.py \
    --old-dir 01_Mickey_hdf \
    --zip-dir 01_Mickey_sprockets \
    --out-dir unwrapping/inr/results/zip_resize \
    --pairs 848:490 863:510 1988:1940 2003:1960 \
    --scale 1.2508 --rot-sweep

echo "DONE"
