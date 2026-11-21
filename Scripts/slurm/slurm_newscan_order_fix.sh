#!/bin/bash
# FIT CHECK for --enforce-radial-order (the w5<->w6 track-swap fix), no render:
#   1. v13 --inspect-anchors on the render's merged cache with the film_safe flags
#      + --enforce-radial-order -> matched_walks.npz + the swap report (= a scan
#      of the WHOLE roll for order violations).
#   2. diag_winding_jump.py at the SAME azimuths/slices as the pre-fix run
#      (diag/winding_jump/) for a like-for-like before/after.
#   3. inspect_walk_hires: zooms at the break slices + full-only overlays at 9
#      anchors spread over z (standing fit-check rule).
#
#   sbatch Scripts/slurm/slurm_newscan_order_fix.sh
#
#SBATCH --job-name=nsorder
#SBATCH --partition=daily
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/nsorder_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export PYTHONPATH="$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

R="$PROJ/unwrapping/inr/results/newscan"
D="$R/diag/${OUT:-order_fix}"   # OUT=order_fix2 etc. keeps earlier runs
mkdir -p "$D/hires" "$D/hires_full"
ln -sf "$R/render/walk_anchors.npz" "$D/walk_anchors.npz"

echo "=== [1/3] v13 inspect + --enforce-radial-order $(date) ==="
python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 \
    --ct-dir 01_Mickey_sprockets --seg-dir "$PROJ/newscan_low/pairs" \
    --out-dir "$D" --use-cached-anchors \
    --seam-deg 353.0 --ref-mode track \
    --min-sep 13 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
    --z-heal-px 7.5 --min-coverage 300 --no-gap-infill \
    --transverse-px 3 --seam-exclude-deg 0.3 --seam-bridge \
    --enforce-radial-order \
    --inspect-anchors --inspect-max 10

echo "=== [2/3] diag_winding_jump (same views as pre-fix) $(date) ==="
python -u Scripts/diag_winding_jump.py --matched "$D/matched_walks.npz" \
    --raw-cache "$D/walk_anchors.npz" --ct-dir 01_Mickey_sprockets --out-dir "$D" \
    --az-deg 297.5,272.5,222.5

echo "=== [3/3] inspect_walk_hires $(date) ==="
python -u -m unwrapping.inr.inspect_walk_hires --cache-dir "$D" \
    --ct-dir 01_Mickey_sprockets --anchors 230,270,275,282,290,320 \
    --wedge-az-deg 304.5,279.5,229.5 --wedge-deg 12 --out-dir "$D/hires"
python -u -m unwrapping.inr.inspect_walk_hires --cache-dir "$D" \
    --ct-dir 01_Mickey_sprockets --anchors 0,30,200,380,560,740,920,1100,1280,1450 \
    --full-only --out-dir "$D/hires_full"

echo "=== [4/4] compare vs the pre-fix geometry $(date) ==="
python -u Scripts/diag_order_fix_compare.py \
    --old "$R/diag/winding_jump/matched_walks.npz" --new "$D/matched_walks.npz"
python -u Scripts/diag_kink_compare.py \
    --old "$R/diag/winding_jump/matched_walks.npz" --new "$D/matched_walks.npz"

ls -la "$D" "$D/hires" "$D/hires_full"
echo "=== DONE $(date) ==="
