#!/bin/bash
# WALK TEST on the NEW scan, one 20-slice picture-band chunk (zip z500-519) -- the
# same slices whose segmentation transfer was already verified.
#
# Stages: predict WITH probabilities (the walk needs softmax, the transfer test only
# produced labels) -> softmax->h5 (this also orients seg to the CT, guarding the
# H<->W transpose that once corrupted a whole render) -> measure the SEAM on the
# segmented slices -> v13 walk with --inspect-anchors (FIT CHECK ONLY, no render).
#
# All spacing-keyed thresholds are scaled by 1.2508 (new scan is that much finer:
# spacing 26px vs 21, film 22 vs 18, emulsion 5 vs 4 -- measured, not assumed).
# Render knob --transverse-px 1 per request (tw1, not tw3).
#
#   sbatch Scripts/slurm/slurm_newscan_walktest.sh
#
#SBATCH --clusters=gmerlin7
#SBATCH --job-name=newscan_walktest
#SBATCH --output=/data/user/li_k1/M_thesis/logs/newscan_walktest_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --partition=a100-hourly
#SBATCH --time=01:00:00

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export nnUNet_raw="$PROJ/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="$PROJ/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="$PROJ/nnUNet_data/nnUNet_results"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

PAIRS="$PROJ/newscan_walktest/pairs"
OUT="$PROJ/unwrapping/inr/results/newscan_walk_test"
rm -rf "$PROJ/newscan_walktest" "$OUT"; mkdir -p "$PAIRS" "$OUT"

echo "=== [1/5] build native chunk z500-519 ==="
python -u Scripts/newscan_seg_test.py chunks \
    --in-dir 01_Mickey_sprockets --out-dir "$PAIRS" --z0 500 --z1 519 --scale 1.0

echo "=== [2/5] hdf5 -> nifti ==="
python -u Scripts/hdf5_to_nifti_for_predict.py \
    --in-dir "$PAIRS" --out-dir "$PROJ/newscan_walktest/nii" --prefix NewScan

echo "=== [3/5] predict WITH probabilities ==="
nnUNetv2_predict -i "$PROJ/newscan_walktest/nii" -o "$PROJ/newscan_walktest/pred" \
    -d 502 -c 3d_fullres -tr nnUNetTrainerProgress -f 0 \
    --save_probabilities --disable_tta
python -u Scripts/nnunet_softmax_to_h5_probs.py \
    --nnunet-out "$PROJ/newscan_walktest/pred" --pairs-dir "$PAIRS" --prefix NewScan
ls -la "$PAIRS"

echo "=== [4/5] SEAM on the segmented slices (must NOT assume the old 60deg) ==="
python -u Scripts/diag_seam_seg.py --seg-dir "$PAIRS" | tee "$OUT/seam.log"
SEAM=$(grep '^SEAM_DEG=' "$OUT/seam.log" | tail -1 | cut -d= -f2)
echo "  -> using --seam-deg $SEAM"

echo "=== [5/5] v13 walk, fit check only (--inspect-anchors) ==="
python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 \
    --ct-dir 01_Mickey_sprockets \
    --seg-dir "$PAIRS" \
    --out-dir "$OUT" \
    --slice-frac 1.0 \
    --seam-deg "$SEAM" \
    --smooth-px 0.75 --snap-max 10 --step 2.5 --coast-max 50 \
    --film-min-thick 7.5 --jump-max 5 --snap-accept 7.5 --max-dr 2.5 \
    --recenter-search 7.5 --recenter-sigma 25 \
    --min-sep 7.5 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
    --z-heal-px 7.5 --min-coverage 8 \
    --transverse-px 1 \
    --inspect-anchors --inspect-max 6

echo "=== DONE. fit images in $OUT ==="
ls -la "$OUT"
