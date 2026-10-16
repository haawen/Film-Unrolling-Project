#!/bin/bash
# WALK TEST on the NEW scan, stage 2: reuse the already-predicted probabilities in
# newscan_walktest/pairs (so NO GPU needed) -> pooled-profile SEAM -> v13 walk with
# --inspect-anchors (fit check only, no render).
#
# Why a rerun: job 192665 was going to hit its 1h wall in the seam stage.
# find_spool_center costs ~2.5 min per call on these 3738x3769 slices (the 63s in
# the notes was measured on the old 3063^2 ones), and that version centred EVERY
# slice. Now 3 centres per chunk, interpolated.
#
# Also: the per-slice seam detector is unusable on this scan (-59.5, -68.5, -102.0,
# -71.0, +89.0, +136.0 deg on consecutive slices with identical centres), so the
# seam now comes from the POOLED gap-count profile over all slices.
#
#   sbatch Scripts/slurm/slurm_newscan_walktest2.sh
#
#SBATCH --job-name=newscan_walk5
#SBATCH --partition=daily
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/newscan_walk5_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

PAIRS="$PROJ/newscan_walktest/pairs"
OUT="$PROJ/unwrapping/inr/results/newscan_walk_eps03"
mkdir -p "$OUT"
ls -la "$PAIRS"


# Seam measured by job 7772879 (pooled gap-count profile over the 20 slices).
# CAVEAT: weak evidence -- dip 30.12 vs baseline 33.03 with the profile spanning
# 30.1-36.2, so the minimum is not cleanly separated from noise. If the fit looks
# wrong, the seam is suspect #1: replace the gap-count detector with a step
# detector on the pooled OUTER-BOUNDARY profile r_out(phi) (the outermost winding
# TERMINATES at the seam, giving a sharp ~26px step on a smooth curve).
SEAM=133.50

# WALK TEST 5: the ONLY change vs test 4 is --seam-exclude-deg 0.3, the CANONICAL
# v12/v13 value (see results/v13_arclen_diag/README.md; the argparse DEFAULT is 6.0,
# which test 4 silently used -> a 12deg untraced wedge instead of 0.6deg, i.e. 20x
# too wide, which is what the user spotted in the fit overlays).
#
# DOUBLE DUTY as a seam probe: with only 0.3deg of blind spot the walks are forced
# THROUGH the physical seam, so if 133.5 is wrong they will cross onto neighbouring
# windings at the REAL seam azimuth -- and where they break localises it. On the old
# scan eps=0.3 is safe only because 60deg is independently verified.
echo "=== v13 walk, fit check only, eps=0.3 (canonical), seam $SEAM forced ==="
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

echo "=== DONE. fit images in $OUT ==="
ls -la "$OUT"
