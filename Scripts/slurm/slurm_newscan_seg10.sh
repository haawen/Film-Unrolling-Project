#!/bin/bash
# Segment TEN 20-slice chunks spread over the WHOLE new scan, so the walk can be
# fit-checked away from the perforation rows.
#
# Everything so far was judged on z500-519 alone, which sits ~40 slices above the
# LOW perf row (z ~200-460) -- i.e. the hardest part of the film, and not
# representative. 01_Mickey_sprockets holds all 2361 slices z40-2400 contiguous
# (54 GB), so no export is needed; these chunks come straight off it.
#
# z picks, against the measured z structure (Scripts/diag_zip_zsurvey.py):
#   <60 packaging | 75-200 margin | 200-460 LOW PERF | 470-1975 picture band
#   | 1990-2250 HIGH PERF | 2260-2350 margin | >2360 packaging
#   250  low perf row
#   560 780 1000 1220 1440 1660 1880   picture band, evenly spread
#   2060 high perf row
#   2200 margin
#
# Stage 2 (the walk + fit overlays + miss census) is slurm_newscan_walk10.sh,
# submit with --dependency=afterok:<this job> so the GPU is not held during the
# ~2.5 h of CPU walking.
#
#   sbatch Scripts/slurm/slurm_newscan_seg10.sh
#
#SBATCH --clusters=gmerlin7
#SBATCH --job-name=newscan_seg10
#SBATCH --output=/data/user/li_k1/M_thesis/logs/newscan_seg10_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --partition=a100-daily
#SBATCH --time=04:00:00

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export nnUNet_raw="$PROJ/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="$PROJ/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="$PROJ/nnUNet_data/nnUNet_results"

BASE="$PROJ/newscan_z10"
PAIRS="$BASE/pairs"
rm -rf "$BASE"; mkdir -p "$PAIRS" "$BASE/nii"

Z0S="250 560 780 1000 1220 1440 1660 1880 2060 2200"

echo "=== [1/3] build 10 native chunks ==="
for Z0 in $Z0S; do
  Z1=$((Z0 + 19))
  python -u Scripts/newscan_seg_test.py chunks \
      --in-dir 01_Mickey_sprockets --out-dir "$PAIRS" --z0 "$Z0" --z1 "$Z1" --scale 1.0
done
ls -la "$PAIRS"

echo "=== [2/3] hdf5 -> nifti ==="
python -u Scripts/hdf5_to_nifti_for_predict.py \
    --in-dir "$PAIRS" --out-dir "$BASE/nii" --prefix NewScan

echo "=== [3/3] predict WITH probabilities, then softmax -> h5 ==="
nnUNetv2_predict -i "$BASE/nii" -o "$BASE/pred" \
    -d 502 -c 3d_fullres -tr nnUNetTrainerProgress -f 0 \
    --save_probabilities --disable_tta
python -u Scripts/nnunet_softmax_to_h5_probs.py \
    --nnunet-out "$BASE/pred" --pairs-dir "$PAIRS" --prefix NewScan

echo "=== DONE ==="
ls -la "$PAIRS"
