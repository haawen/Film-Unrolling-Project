#!/bin/bash
# STAGE 1c - train the four comparison models of the segmentation table.
#
#   sbatch Code/slurm/seg_03_train_baselines.sh nnunet2d_matched   # Dataset503, 2d
#   sbatch Code/slurm/seg_03_train_baselines.sh nnunet2d_orig      # Dataset501, 2d
#   sbatch Code/slurm/seg_03_train_baselines.sh unet3d             # MONAI UNet3D
#   sbatch Code/slurm/seg_03_train_baselines.sh swinunetr          # MONAI SwinUNETR
#
# Preprocess first: seg_01_preprocess.sh 502 501 503  (MONAI reads Dataset502's
# splits_final.json from $nnUNet_preprocessed; without it train_monai.py silently
# falls back to its own K-fold split).
# Outputs go to $WORK/nnUNet_results and $WORK/monai_results.
#
#SBATCH --clusters=gmerlin7
#SBATCH --partition=a100-daily
#SBATCH --gres=gpu:1
#SBATCH --job-name=seg_train_baseline
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=work/logs/%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/Code/slurm/_activate.sh"
export nnUNet_results="$WORK/nnUNet_results"
export MONAI_RESULTS="$WORK/monai_results"
MODEL=${1:?usage: seg_03_train_baselines.sh nnunet2d_matched|nnunet2d_orig|unet3d|swinunetr}

cp Code/segmentation/nnunet_custom_trainer.py \
   "$(python -c 'import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))')/training/nnUNetTrainer/variants/custom_trainer.py"

case "$MODEL" in
  nnunet2d_matched) nnUNetv2_train 503 2d 0 -tr nnUNetTrainerProgress --npz ;;
  nnunet2d_orig)    nnUNetv2_train 501 2d 0 -tr nnUNetTrainerProgress --npz ;;
  unet3d|swinunetr)
      test -f "$nnUNet_preprocessed/Dataset502_MickeyScroll3D/splits_final.json" \
          || { echo "run seg_01_preprocess.sh first (needs Dataset502 splits_final.json)"; exit 1; }
      python -u Code/segmentation/train_monai.py --model "$MODEL" --fold 0 --epochs 250 ;;
  *) echo "unknown model $MODEL"; exit 1 ;;
esac
