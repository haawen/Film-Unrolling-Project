#!/bin/bash
#SBATCH --job-name=unroll_cont
#SBATCH --partition=daily
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=/data/user/li_k1/M_thesis/unwrapping/inr/results/whole_scroll_continuous/slurm_%j.out

set -e
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd /data/user/li_k1/M_thesis

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p unwrapping/inr/results/whole_scroll_continuous

python -u -m unwrapping.inr.unroll_continuous \
    --ct-dir 01_Mickey_hdf \
    --seg-dir 01_Mickey_3d \
    --out-dir unwrapping/inr/results/whole_scroll_continuous \
    --start-xy 2145,1600 --end-xy 1200,178 \
    --smooth-per-pt 0.35 --n-rays 2880 --seam-exclude-deg 8 \
    --n-overlays 8

echo "DONE"
