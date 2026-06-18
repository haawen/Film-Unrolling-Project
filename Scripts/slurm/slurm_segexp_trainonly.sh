#!/bin/bash
# Seg-sensitivity experiment, stage 1b (GPU train only): transfer-learn from
# Dataset502 fold-0 weights on the already-preprocessed Dataset510. Intended to
# run with --dependency=afterok:<prep job> so it starts the moment prep is done
# and an a100-daily GPU frees.
#   sbatch --dependency=afterok:<PREPJOB> Scripts/slurm/slurm_segexp_trainonly.sh
#SBATCH --cluster=gmerlin7
#SBATCH --job-name=segexp_tr
#SBATCH --output=logs/segexp_tr_%j.out
#SBATCH --error=logs/segexp_tr_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --partition=a100-daily
#SBATCH --time=23:59:00

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
PRETRAINED="${nnUNet_results}/Dataset502_MickeyScroll3D/nnUNetTrainerProgress__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
EPOCHS_TR="nnUNetTrainer_100epochs"

echo "=== transfer-learn (fold 0, ${EPOCHS_TR}, pretrained from 502) ($(date)) ==="
nnUNetv2_train ${DID} 3d_fullres 0 -tr ${EPOCHS_TR} \
    -pretrained_weights "${PRETRAINED}" --npz

echo "=== TRAIN DONE $(date) ==="
echo "ckpt: ${nnUNet_results}/Dataset510_SynthClean/${EPOCHS_TR}__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
