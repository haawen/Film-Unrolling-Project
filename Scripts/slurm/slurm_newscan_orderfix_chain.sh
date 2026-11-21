#!/bin/bash
# Render + stabilise the new scan WITH --enforce-radial-order (the w5<->w6 track-
# swap fix behind the end-of-film left-band tear), reproducing the chain behind
# compare/gt_vs_newscan_affine.mp4 with only the fix flag added:
#   0. back up the current render/wholeroll.npy (render/ keeps only the latest)
#   1. slurm_newscan_render.sh orderfix b0 --transverse-px 4 --enforce-radial-order
#      -> render/wholeroll.npy + videos/film_orderfix.mp4
#   2. frame_match --no-fliplr (the new scan's mirror; see doc s12) -> frame_match_orderfix/
#   3. stabilize_affine --mode affine on film_stabilized_gtsync.mp4 -> film_affine.mp4
# Fit check that approved the flag: diag/order_fix3 (doc s11.5).
#
#   sbatch Scripts/slurm/slurm_newscan_orderfix_chain.sh
#
#SBATCH --job-name=nsofix
#SBATCH --partition=daily
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/nsofix_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
cd "$PROJ"
R="$PROJ/unwrapping/inr/results/newscan"
BK="$R/diag/order_fix3/wholeroll_pre_orderfix.npy"
[ -f "$BK" ] || cp -v "$R/render/wholeroll.npy" "$BK"

echo "=== [1/3] render $(date) ==="
bash Scripts/slurm/slurm_newscan_render.sh orderfix b0 --transverse-px 4 --enforce-radial-order

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
export PYTHONPATH="$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
GT="$PROJ/data/h265_1080p.mp4"
O="$R/frame_match_orderfix"

echo "=== [2/3] frame_match --no-fliplr $(date) ==="
python -u -m unwrapping.eval.frame_match --strip "$R/render/wholeroll.npy" --gt "$GT" \
    --z-crop 0.203,0.80 --pitch 1130 --export-video --montage --video --fps 25 \
    --no-fliplr --out-dir "$O" 2>&1 | tail -60

echo "=== [3/3] stabilize_affine $(date) ==="
python -u -m unwrapping.eval.stabilize_affine \
    --in "$O/film_stabilized_gtsync.mp4" --gt "$GT" --mode affine \
    --out "$O/film_affine.mp4"

ls -la "$O" "$R/videos"
echo "=== nsofix DONE $(date) ==="
