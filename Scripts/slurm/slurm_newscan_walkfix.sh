#!/bin/bash
# Walk-parameter variants on the FAILING range only (z1590-1969).
#
# Measured failure curve (Scripts/diag_walk_quality.py on the dense run): radial
# jumps per anchor are 0.1-0.6 through z1070-1569 and then climb 3.0 -> 7.7 ->
# 12.6 -> 16.1 by z1869, i.e. a 30-100x rise starting z1570-1620. That is exactly
# where the render tears and where the overlays show radial jogs.
#
# Cause (from the CT): the bright emulsion line the walk snaps to FADES toward the
# film edge. By z1640 the bands are nearly uniform grey. The walk's thresholds were
# tuned on the picture band's strong emulsion, so as contrast dies it snaps to
# noise or to a neighbouring band instead of coasting.
#
# So every variant here moves the same way: TRUST CONTINUITY, DISTRUST WEAK
# EVIDENCE. Lower --snap-accept (only snap to a confident target, else coast),
# higher --coast-max (glide further through dead stretches), lower --max-dr (cap
# how far the radius may move per step).
#   V0 baseline    7.5 / 50  / 2.5   (what produced the artifact)
#   V1             5.0 / 120 / 1.5
#   V2             4.0 / 200 / 1.2   + smooth-px 1.5
#   V3             7.5 / 120 / 1.5   (continuity levers only, snap unchanged)
# V3 isolates whether the snap threshold matters at all, so a win by V1/V2 can be
# attributed rather than assumed.
#
# --slice-frac 0.25 and --n-centers 1: this is a RANKING run, not a deliverable.
# The metric is per-anchor so fewer anchors is fine, and the centre moves <1px
# over 380 slices.
#
#   sbatch Scripts/slurm/slurm_newscan_walkfix.sh
#
#SBATCH --job-name=walkfix
#SBATCH --partition=daily
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/walkfix_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

PAIRS="$PROJ/newscan_fix/pairs"
R="$PROJ/unwrapping/inr/results/newscan_fix"
mkdir -p "$R"

run () {                    # name snap_accept coast_max max_dr smooth
  local NAME=$1 SA=$2 CM=$3 DR=$4 SM=$5
  local OUT="$R/$NAME"
  rm -rf "$OUT"; mkdir -p "$OUT"
  echo
  echo "################## $NAME: snap-accept $SA coast-max $CM max-dr $DR smooth $SM ##################"
  python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 \
      --ct-dir 01_Mickey_sprockets --seg-dir "$PAIRS" --out-dir "$OUT" \
      --slice-frac 0.25 --n-centers 1 --seam-deg 353.0 \
      --smooth-px "$SM" --snap-max 10 --step 2.5 --coast-max "$CM" \
      --film-min-thick 7.5 --jump-max 5 --snap-accept "$SA" --max-dr "$DR" \
      --recenter-search 7.5 --recenter-sigma 25 \
      --min-sep 13 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
      --z-heal-px 7.5 --min-coverage 20 \
      --transverse-px 3 --max-infill 3 --radial-robust \
      --seam-exclude-deg 0.3 \
      --inspect-anchors --inspect-max 4 \
    || { echo "  $NAME FAILED"; return 0; }
  echo "--- $NAME walk quality ---"
  python -u Scripts/diag_walk_quality.py --npz "$OUT/matched_walks.npz" --zbin 25 \
    || echo "  scoring failed"
}

run V0_baseline 7.5 50  2.5 0.75
run V1_conserv  5.0 120 1.5 0.75
run V2_strict   4.0 200 1.2 1.5
run V3_contonly 7.5 120 1.5 0.75

echo
echo "=== summary: baseline line from each variant ==="
grep -H "baseline (first" "$PROJ/logs/walkfix_${SLURM_JOB_ID}.out" || true
