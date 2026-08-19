#!/bin/bash
# DENSE (100%) walk of the new scan's PICTURE BAND, one batch of 15 chunks.
#
#   sbatch Scripts/slurm/slurm_newscan_dense_batch.sh <batch 0..4>
#
# Picture band only (z470-1969, 75 chunks x 20 slices = 1500 anchors). The perf
# rows and margins are deliberately excluded: walking them was measured to fail
# outright (z250/z2060/z2200 -> 44k-57k columns vs ~259k, arc-stretch up to
# +21880%), because the perforations break emulsion continuity past --coast-max
# and the margin has no picture emulsion to snap to at all. Their z still gets
# rendered later, from the nearest picture-band geometry (v13's renderer clamps
# outside the anchor range), which is how the sprockets came out clean last time.
#
# WHY BATCHED: /data has ~327 GB free and a 20-slice chunk's softmax is ~3.2 GB,
# so all 75 at once would need ~243 GB of probability maps plus a similar volume
# of nnU-Net's own intermediates. Each batch segments 15 chunks (~48 GB), walks
# them into a cache, then DELETES the probability maps. Peak stays ~100 GB.
#
# WHY seg AND walk in one GPU job: the two stages live on different clusters
# (predict needs gmerlin7, the walk is pure CPU), and Slurm rejects a
# cross-cluster --dependency, so splitting them would mean babysitting 10 jobs.
# This holds a GPU during the CPU walk -- accepted deliberately for restartability.
#
# --n-centers 6 per batch x 5 batches = 30 find_spool_center calls over the stack,
# which is the standing spool-centre policy. They dominate the wall time here
# (~623 s per call on these 3738x3769 slices, NOT the ~2.5 min quoted for the old
# 3063^2 ones).
#
#SBATCH --clusters=gmerlin7
#SBATCH --job-name=nsdense
#SBATCH --output=/data/user/li_k1/M_thesis/logs/nsdense_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --partition=a100-daily
#SBATCH --time=12:00:00

set -euo pipefail
B=${1:?usage: sbatch slurm_newscan_dense_batch.sh <batch 0..4>}
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export nnUNet_raw="$PROJ/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="$PROJ/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="$PROJ/nnUNet_data/nnUNet_results"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

PER=15                      # chunks per batch
Z_START=470                 # picture band starts ~470
BASE="$PROJ/newscan_dense/b${B}"
PAIRS="$BASE/pairs"
OUT="$PROJ/unwrapping/inr/results/newscan_dense/b${B}"
rm -rf "$BASE"; mkdir -p "$PAIRS" "$BASE/nii" "$OUT"

echo "=== batch $B: chunks $((B*PER))..$((B*PER+PER-1)) ==="
for i in $(seq $((B*PER)) $((B*PER+PER-1))); do
  Z0=$((Z_START + 20*i)); Z1=$((Z0+19))
  echo "  build z${Z0}-${Z1}"
  python -u Scripts/newscan_seg_test.py chunks \
      --in-dir 01_Mickey_sprockets --out-dir "$PAIRS" --z0 "$Z0" --z1 "$Z1" --scale 1.0
done
df -h /data | tail -1

echo "=== nifti + predict ==="
python -u Scripts/hdf5_to_nifti_for_predict.py \
    --in-dir "$PAIRS" --out-dir "$BASE/nii" --prefix NewScan
nnUNetv2_predict -i "$BASE/nii" -o "$BASE/pred" \
    -d 502 -c 3d_fullres -tr nnUNetTrainerProgress -f 0 \
    --save_probabilities --disable_tta
python -u Scripts/nnunet_softmax_to_h5_probs.py \
    --nnunet-out "$BASE/pred" --pairs-dir "$PAIRS" --prefix NewScan
# nnU-Net's own outputs are as large as the h5 softmax; drop them now.
rm -rf "$BASE/pred" "$BASE/nii"
df -h /data | tail -1

echo "=== dense walk (slice-frac 1.0), seam 353, thresholds x1.2508 ==="
python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 \
    --ct-dir 01_Mickey_sprockets \
    --seg-dir "$PAIRS" \
    --out-dir "$OUT" \
    --slice-frac 1.0 \
    --n-centers 6 \
    --seam-deg 353.0 \
    --smooth-px 0.75 --snap-max 10 --step 2.5 --coast-max 50 \
    --film-min-thick 7.5 --jump-max 5 --snap-accept 7.5 --max-dr 2.5 \
    --recenter-search 7.5 --recenter-sigma 25 \
    --min-sep 7.5 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
    --z-heal-px 7.5 --min-coverage 100 \
    --transverse-px 3 --max-infill 3 --radial-robust \
    --seam-exclude-deg 0.3 \
    --inspect-anchors --inspect-max 3

echo "=== drop the probability maps, keep the cache ==="
ls -la "$OUT"
rm -rf "$PAIRS"
df -h /data | tail -1
echo "=== batch $B DONE ==="
