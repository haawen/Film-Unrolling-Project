#!/bin/bash
# Diagnose the left-band break (strip z~743-753, innermost windings, video frames
# ~198-214) in the DELIVERED geometry, without re-rendering:
#   1. v13 --inspect-anchors on the render's merged cache (b0+b1+b2+b3+tail_dense,
#      exactly the film_safe/tw2/tw4 flags) -> matched_walks.npz, then exits
#      BEFORE the CT render. The cache is symlinked; cached mode only reads it.
#   2. diag_winding_jump.py -> per-winding dR/dT vs a clean z790-830 reference,
#      raw-walk table, targeted overlays at the worst azimuths.
#   3. inspect_walk_hires full + zoom overlays at slices across the break.
#
#   sbatch Scripts/slurm/slurm_newscan_diag_jump.sh
#
#SBATCH --job-name=nsjump
#SBATCH --partition=daily
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/nsjump_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export PYTHONPATH="$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

R="$PROJ/unwrapping/inr/results/newscan"
D="$R/diag/winding_jump"
mkdir -p "$D/hires"
ln -sf "$R/render/walk_anchors.npz" "$D/walk_anchors.npz"

if [ ! -f "$D/matched_walks.npz" ]; then
echo "=== [1/3] v13 inspect (render flags of film_safe) $(date) ==="
python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 \
    --ct-dir 01_Mickey_sprockets --seg-dir "$PROJ/newscan_low/pairs" \
    --out-dir "$D" --use-cached-anchors \
    --seam-deg 353.0 --ref-mode track \
    --min-sep 13 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
    --z-heal-px 7.5 --min-coverage 300 --no-gap-infill \
    --transverse-px 3 --seam-exclude-deg 0.3 --seam-bridge \
    --inspect-anchors --inspect-max 10
fi

echo "=== [2/3] diag_winding_jump $(date) ==="
python -u Scripts/diag_winding_jump.py --matched "$D/matched_walks.npz" \
    --raw-cache "$D/walk_anchors.npz" --ct-dir 01_Mickey_sprockets --out-dir "$D"

echo "=== [3/3] inspect_walk_hires $(date) ==="
source "$D/hires_args.sh"
echo "anchors $ANCHORS  wedges(rel seam) $WEDGES"
python -u -m unwrapping.inr.inspect_walk_hires --cache-dir "$D" \
    --ct-dir 01_Mickey_sprockets --anchors "$ANCHORS" --wedge-az-deg "$WEDGES" \
    --wedge-deg 12 --out-dir "$D/hires"

ls -la "$D" "$D/hires"
echo "=== DONE $(date) ==="
