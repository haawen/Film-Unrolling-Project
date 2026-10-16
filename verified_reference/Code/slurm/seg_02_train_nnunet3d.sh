#!/bin/bash
# STAGE 1b - train the selected segmentation model: nnU-Net v2, 3d_fullres,
# Dataset502, fold 0 (20 training / 5 held-out sub-volumes), custom trainer
# nnUNetTrainerProgress (= default nnU-Net trainer + progress bar + periodic
# checkpoints; 250 epochs). Run seg_01_preprocess.sh first.
#
#   sbatch Code/slurm/seg_02_train_nnunet3d.sh          # fold 0 (thesis)
#   sbatch Code/slurm/seg_02_train_nnunet3d.sh 0 --c    # continue an interrupted run
#
# Writes to $WORK/nnUNet_results (NOT the shipped model in Data/2_trained_models).
# ~20 h on one A100-80GB.
#
#SBATCH --clusters=gmerlin7
#SBATCH --partition=a100-daily
#SBATCH --gres=gpu:1
#SBATCH --job-name=seg_train_3d
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=72G
#SBATCH --output=work/logs/%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/Code/slurm/_activate.sh"
export nnUNet_results="$WORK/nnUNet_results"
mkdir -p "$nnUNet_results"

# register the custom trainer with the installed nnU-Net package
cp -v Code/segmentation/nnunet_custom_trainer.py \
      "$(python -c 'import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))')/training/nnUNetTrainer/variants/custom_trainer.py"

FOLD=${1:-0}; shift || true
nnUNetv2_train 502 3d_fullres "$FOLD" -tr nnUNetTrainerProgress --npz "$@"
echo "=== trained model in $nnUNet_results/Dataset502_MickeyScroll3D ==="
