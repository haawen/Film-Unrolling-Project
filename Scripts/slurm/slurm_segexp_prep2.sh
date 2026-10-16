#!/bin/bash
# Seg-sensitivity experiment, stage 1a-fix: copy dataset.json into the
# preprocessed dir (the transfer-learning recipe via move_plans does not place
# it there in nnU-Net 2.6.4) and run preprocess. GPU-free.
#   sbatch Scripts/slurm/slurm_segexp_prep2.sh
#SBATCH --cluster=gmerlin7
#SBATCH --job-name=segexp_prep2
#SBATCH --output=logs/segexp_prep2_%j.out
#SBATCH --error=logs/segexp_prep2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --partition=a100-hourly
#SBATCH --time=01:00:00

set -euo pipefail
PROJECT_DIR="$HOME/M_thesis"
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "${PROJECT_DIR}"
export nnUNet_raw="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_DIR}/nnUNet_data/nnUNet_results"
DID=510
DNAME="Dataset510_SynthClean"

cp -f "${nnUNet_raw}/${DNAME}/dataset.json" "${nnUNet_preprocessed}/${DNAME}/dataset.json"
echo "copied dataset.json -> preprocessed/${DNAME}"
echo "=== preprocess 3d_fullres ($(date)) ==="
nnUNetv2_preprocess -d ${DID} -plans_name nnUNetPlans -c 3d_fullres
echo "=== PREP2 DONE $(date) ==="
