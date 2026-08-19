#!/bin/bash
# WALK TEST 6 on the NEW scan — the ONLY change vs test 5 is the SEAM: 353.0 deg
# instead of the 133.5 that tests 2-5 forced.
#
# 133.5 came from the pooled gap-count profile and was always weak (dip 30.12 vs
# a 30.1-36.2 baseline). Three independent measurements now put the seam at ~353:
#
#   1. Step detector on the pooled OUTER-BOUNDARY profile r_out(phi)
#      (Scripts/diag_seam_outer.py, job 7848427): +22.2 px -- about one 26 px
#      winding pitch -- at az 353.0, SNR 5.2x, and per-slice 352.25-353.00 on
#      5 consecutive slices (z506-538). Nothing comparable anywhere else.
#   2. The outermost walked winding w34 (r 1707) TERMINATES at abs 355.1 deg;
#      every interior winding spans the full 2pi-2eps.
#   3. Direct look at the CT (hires/anchor_010_z510_full.png cropped at az 353,
#      r 1700): the outermost band ends bluntly there, and the eps=0.3 walk runs
#      straight PAST the end onto the layer below -- the textbook wrong-seam
#      crossing. At az 128 (the other r_out feature) the band is continuous, so
#      that one is eccentricity, not a film end.
#
# The failure the wrong seam produced: on z500-506 the walk degenerates into a
# radial zigzag at abs ~136 deg (149/116/75/72/17/11/8 jumps >8 px per anchor,
# median phi 2-3 deg, i.e. right at the forced seam), decaying to 0 by z507.
#
# PREDICTIONS if 353 is right: (a) no winding reports a short span except the
# inner tongue, (b) the ~136 deg zigzag is gone, (c) arc-columns' within-turn
# stretch drops from the +20% max test 5 reported toward the old scan's +-5%.
#
#   sbatch Scripts/slurm/slurm_newscan_walktest6.sh
#
#SBATCH --job-name=newscan_walk6
#SBATCH --partition=hourly
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/newscan_walk6_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

PAIRS="$PROJ/newscan_walktest/pairs"
OUT="$PROJ/unwrapping/inr/results/newscan_walk_seam353"
mkdir -p "$OUT"

SEAM=353.0

# Thresholds are test 5's = the v13 canonical ones scaled by 1.2508 for this
# scan's finer sampling (26 px pitch vs 21). --seam-exclude-deg 0.3 is the
# canonical v12/v13 value and is NOT the argparse default (6.0).
echo "=== v13 walk, fit check only, eps=0.3, seam $SEAM (measured) ==="
python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 \
    --ct-dir 01_Mickey_sprockets \
    --seg-dir "$PAIRS" \
    --out-dir "$OUT" \
    --slice-frac 1.0 \
    --n-centers 3 \
    --seam-deg "$SEAM" \
    --smooth-px 0.75 --snap-max 10 --step 2.5 --coast-max 50 \
    --film-min-thick 7.5 --jump-max 5 --snap-accept 7.5 --max-dr 2.5 \
    --recenter-search 7.5 --recenter-sigma 25 \
    --min-sep 7.5 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
    --z-heal-px 7.5 --min-coverage 8 \
    --transverse-px 1 --max-infill 3 --radial-robust \
    --seam-exclude-deg 0.3 \
    --inspect-anchors --inspect-max 6

echo
echo "=== where does each winding terminate now? (expect: only the tongue) ==="
python -u Scripts/diag_seam_terminus.py --npz "$OUT/matched_walks.npz" --anchors 20

echo
echo "=== FIT CHECK overlays: AT the seam (0 rel) and opposite it ==="
python -u -m unwrapping.inr.inspect_walk_hires \
    --cache-dir "$OUT" --ct-dir 01_Mickey_sprockets \
    --anchors 0,10,19 --wedge-az-deg 0,180,90,270 \
    --out-dir "$OUT/hires"

ls -la "$OUT" "$OUT/hires"
