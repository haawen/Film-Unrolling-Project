#!/bin/bash
# Is the OLD -> NEW frame relation just SCALE + an AXIS CONVENTION (transpose/flip)?
# Tests all 8 symmetries of the square, and FIRST runs the old-frame identity
# CONTROL (which must come out tight, else the metric is noise and the verdict void).
#
#   sbatch Scripts/slurm/slurm_zip_dihedral.sh
#
#SBATCH --job-name=zip_dihedral
#SBATCH --partition=hourly
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/zip_dihedral_%j.out

set -e
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd /data/user/li_k1/M_thesis
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

python -u Scripts/diag_zip_dihedral.py \
    --geom unwrapping/inr/results/walk_full_final/matched_walks.npz \
    --zip-dir 01_Mickey_sprockets \
    --out-dir unwrapping/inr/results/zip_dihedral \
    --zip-z 490 510 1940 1960 \
    --scale 1.2508 \
    --control-old-dir 01_Mickey_hdf \
    --control-z 848 863 1988 2003

echo "DONE"
