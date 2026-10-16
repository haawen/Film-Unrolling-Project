#!/bin/bash
# STAGE 1d - evaluate the segmentation models on the held-out sub-volumes
# (thesis segmentation table: Dice / IoU / precision / recall, macro over classes).
#
#   sbatch Code/slurm/seg_04_evaluate.sh                 # the SHIPPED models
#   MODELS_FROM=work sbatch Code/slurm/seg_04_evaluate.sh   # your retrained models
#
# Steps (as in the thesis): (1) cut the 3D held-out volumes into 2D slices for the
# original 2D model, (2) nnU-Net 2D (Dataset501) inference on those slices,
# (3) MONAI inference, (4) reassemble + score everything on identical voxels.
# The nnU-Net 3D and matched-2D predictions are the ones nnU-Net writes into
# fold_0/validation/ at the end of training (shipped with the models).
# Output: $FAIR_DIR/fair_comparison_results.json + figures/.
#
#SBATCH --clusters=gmerlin7
#SBATCH --partition=a100-hourly
#SBATCH --gres=gpu:1
#SBATCH --job-name=seg_evaluate
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=work/logs/%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/Code/slurm/_activate.sh"
if [ "${MODELS_FROM:-shipped}" = work ]; then
    export nnUNet_results="$WORK/nnUNet_results"
    export MONAI_RESULTS="$WORK/monai_results"
else
    # predict_monai.py writes its predictions next to the checkpoint: use a copy
    rm -rf "$WORK/monai_results_eval"; cp -r "$MONAI_RESULTS" "$WORK/monai_results_eval"
    export MONAI_RESULTS="$WORK/monai_results_eval"
fi
# predict_monai.py reads the fold definition from here
mkdir -p "$nnUNet_preprocessed/Dataset502_MickeyScroll3D"
cp Data/1_segmentation_training/splits/Dataset502_MickeyScroll3D_splits_final.json \
   "$nnUNet_preprocessed/Dataset502_MickeyScroll3D/splits_final.json"
cp Code/segmentation/nnunet_custom_trainer.py \
   "$(python -c 'import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))')/training/nnUNetTrainer/variants/custom_trainer.py"

echo "=== [1/4] 2D slices of the held-out volumes ==="
python -u Code/segmentation/fair_compare.py --prepare --fold 0
echo "=== [2/4] nnU-Net 2D (Dataset501) on those slices ==="
mkdir -p "$FAIR_DIR/slices_output"
nnUNetv2_predict -i "$FAIR_DIR/slices_input" -o "$FAIR_DIR/slices_output" \
    -d 501 -c 2d -tr nnUNetTrainerProgress -f 0 --disable_tta
echo "=== [3/4] MONAI predictions ==="
for M in unet3d swinunetr; do
    python -u Code/segmentation/predict_monai.py --model "$M" --fold 0 --dataset Dataset502_MickeyScroll3D
done
echo "=== [4/4] reassemble + score ==="
python -u Code/segmentation/fair_compare.py --compare --save --fold 0
echo "=== results: $FAIR_DIR/fair_comparison_results.json ==="
