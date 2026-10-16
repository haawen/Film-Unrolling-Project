#!/bin/bash
# STAGE 2 - segment a whole CT scan with the trained nnU-Net 3D model and write
# walk-ready, compact segmentation chunks.
#
#   sbatch Code/slurm/seg_05_segment_scan.sh                   # picture band z470-1969, 75 chunks
#   sbatch Code/slurm/seg_05_segment_scan.sh 470 1589 <out>    # sub-range / custom out dir
#
# Per batch of N chunks (bounded disk use, ~8 GB of temporaries per chunk):
#   CT slices -> 20-slice chunks (key 'volume') -> NIfTI -> nnUNetv2_predict
#   (--save_probabilities --disable_tta, exactly as in the thesis) -> softmax h5
#   (oriented to the CT; guards the nnU-Net H<->W transpose) -> uint8 one-hot
#   (compact_segmentation.py, verified voxel-identical argmax) -> temporaries deleted.
# Output: <out>/volume_{z0}-{z1}_Probabilities.h5 (default: $WORK/segmentation).
# The shipped Data/3_full_width_ct_scan/segmentation_nnunet3d_picture_band was produced this way.
#
#SBATCH --clusters=gmerlin7
#SBATCH --partition=a100-daily
#SBATCH --gres=gpu:1
#SBATCH --job-name=seg_scan
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=work/logs/%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/Code/slurm/_activate.sh"
CT="$D_SCAN/ct_slices"
Z0=${1:-470}; Z1=${2:-1969}
OUT=${3:-$WORK/segmentation}
PER=${CHUNKS_PER_BATCH:-4}
TMP="$WORK/tmp_seg_${SLURM_JOB_ID:-$$}"
mkdir -p "$OUT"

for ((b0 = Z0; b0 <= Z1; b0 += 20 * PER)); do
    b1=$(( b0 + 20 * PER - 1 )); (( b1 > Z1 )) && b1=$Z1
    echo "=== z$b0-$b1 ==="
    rm -rf "$TMP"; mkdir -p "$TMP/pairs" "$TMP/nii" "$TMP/pred"
    python -u Code/segmentation/ct_slices_to_chunks.py --in-dir "$CT" --out-dir "$TMP/pairs" \
        --chunk-z 20 --z-start "$b0" --z-end "$b1"
    python -u Code/segmentation/chunks_to_nifti.py --in-dir "$TMP/pairs" --out-dir "$TMP/nii" --prefix Scan
    nnUNetv2_predict -i "$TMP/nii" -o "$TMP/pred" -d 502 -c 3d_fullres \
        -tr nnUNetTrainerProgress -f 0 --save_probabilities --disable_tta
    python -u Code/segmentation/nnunet_softmax_to_h5.py --nnunet-out "$TMP/pred" \
        --pairs-dir "$TMP/pairs" --prefix Scan
    python -u Code/segmentation/compact_segmentation.py --in-dir "$TMP/pairs" --out-dir "$OUT" --ct-dir "$CT"
    rm -rf "$TMP"
done
ls "$OUT" | wc -l
echo "=== walk-ready segmentation in $OUT ==="
