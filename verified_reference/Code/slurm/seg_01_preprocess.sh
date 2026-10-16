#!/bin/bash
# STAGE 1a - nnU-Net planning + preprocessing (CPU) for the segmentation datasets.
#
#   sbatch Code/slurm/seg_01_preprocess.sh            # Dataset502 (3D, the selected model)
#   sbatch Code/slurm/seg_01_preprocess.sh 501 503    # also the 2D baselines
#
# Dataset503 (2D slices of Dataset502) is not shipped (22 GB, derived): it is
# built here first if requested. After preprocessing, the exact fold definitions
# used in the thesis are copied in (splits_final.json), so training reproduces the
# reported train/test split.
#
#SBATCH --job-name=seg_preprocess
#SBATCH --partition=daily
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=work/logs/%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/Code/slurm/_activate.sh"

DATASETS=("${@:-502}")
declare -A CFG=([501]=2d [502]=3d_fullres [503]=2d)
declare -A NAME=([501]=Dataset501_MickeyScroll [502]=Dataset502_MickeyScroll3D [503]=Dataset503_MickeyScroll2Dfrom3D)

for D in "${DATASETS[@]}"; do
    if [ "$D" = 503 ] && [ ! -d "$nnUNet_raw/${NAME[503]}" ]; then
        echo "=== building Dataset503 (2D slices of Dataset502) ==="
        python -u Code/segmentation/build_dataset503_2d_from_3d.py
    fi
    echo "=== plan + preprocess Dataset$D (${CFG[$D]}) ==="
    nnUNetv2_plan_and_preprocess -d "$D" --verify_dataset_integrity -c "${CFG[$D]}"
    cp -v "Data/1_segmentation_training/splits/${NAME[$D]}_splits_final.json" \
          "$nnUNet_preprocessed/${NAME[$D]}/splits_final.json"
done
echo "=== preprocessing done ==="
