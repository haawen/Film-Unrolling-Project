#!/bin/bash
# Seg-sensitivity experiment, stage 1a (GPU-free prep): build the chunked
# nnU-Net dataset from CLEAN, transfer Dataset502 plans, preprocess. Runs on
# a100-hourly (higher priority -> starts sooner) so the deterministic stages
# are validated while the GPU training job waits in a100-daily.
#   sbatch Scripts/slurm/slurm_segexp_prep.sh
#SBATCH --cluster=gmerlin7
#SBATCH --job-name=segexp_prep
#SBATCH --output=logs/segexp_prep_%j.out
#SBATCH --error=logs/segexp_prep_%j.err
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
mkdir -p logs
export nnUNet_raw="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_DIR}/nnUNet_data/nnUNet_results"

DID=510
DNAME="Dataset510_SynthClean"
RAW="${nnUNet_raw}/${DNAME}"
CLEAN="${PROJECT_DIR}/unwrapping/synthetic/results/hq_video_4k_outer"

echo "=== [1/4] build chunked dataset from CLEAN ($(date)) ==="
python -u Scripts/build_synth_nnunet_dataset.py \
    --syn-dir "${CLEAN}" --mode train --chunk 24 \
    --prefix Synth3D --raw-dir "${RAW}"

echo "=== [2/4] extract fingerprint ($(date)) ==="
nnUNetv2_extract_fingerprint -d ${DID} --verify_dataset_integrity

echo "=== [3/4] move Dataset502 plans -> ${DID} ($(date)) ==="
nnUNetv2_move_plans_between_datasets -s 502 -t ${DID} -sp nnUNetPlans -tp nnUNetPlans

echo "=== [4/4] preprocess 3d_fullres ($(date)) ==="
nnUNetv2_preprocess -d ${DID} -plans_name nnUNetPlans -c 3d_fullres

echo "=== PREP DONE $(date) ==="
