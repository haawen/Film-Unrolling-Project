#!/bin/bash
# CAN THE PERFORATION BANDS BE WALKED?
#
# Everything about the perf-locked stabilisation hinges on this. The sprockets
# cannot register the picture while the perf bands are EXTRAPOLATED: the two
# rows' measured motion does not agree with each other (across-film corr -0.07,
# along-film -0.23 on the linearly-extrapolated render2), so each row is
# reporting its own band's reconstruction error rather than the film. The only
# way out is to WALK those bands from real segmentation.
#
# The ten-chunk test recorded that the walk "fails" at z250 (low perf) and z2060
# (high perf) because the perf gaps exceed --coast-max. That reads like a
# property of the data, but it is arithmetic: coast_max * step = 50 * 2.5 =
# 125 px of arc, while a 16 mm perforation is 1.27 mm = 188 px of film along its
# length. The walk was being asked to cross a gap 1.5x its allowance.
#
# Two levers, tested factorially on both perf-band chunks:
#   --coast-max   50 (the control that failed) / 120 (300 px) / 200 (500 px)
#   --walk-class  emulsion (fragmentary in these bands) / film (base, found
#                 cleanly with the holes punched through it)
#
# PASS/FAIL IS OBJECTIVE, no eyeballing needed -- from the ten-chunk run, a good
# picture-band chunk gives ~259,000 columns, ~34 windings and arc-stretch med
# +2.3..+3.3%, while the failures gave 44,000-57,000 columns with arc-stretch
# med +123..+470%. z1000 is included as a KNOWN-GOOD control so the numbers can
# be read against a chunk from the same run.
#
#   sbatch Scripts/slurm/slurm_perfband_coast.sh
#
#SBATCH --job-name=perfband_coast
#SBATCH --partition=daily
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/perfband_coast_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

PAIRS="$PROJ/newscan_z10/pairs"
BASE="$PROJ/unwrapping/inr/results/perfband_coast"
mkdir -p "$BASE"
SEAM=353.0

# An explicit variant list, not the full 3x2 factorial: each walk is tens of
# minutes and 13 of them would not fit the wall clock. These 7 still separate
# the two levers (z250 varies coast alone, then class alone) and confirm the
# high row, with a known-good picture-band chunk to read the numbers against.
#   z250  = low perf row, z2060 = high perf row, z1000 = picture band control
VARIANTS="
250:50:emulsion
250:120:emulsion
250:120:film
250:200:film
2060:120:film
2060:200:film
1000:50:emulsion
"

for V in $VARIANTS; do
  Z0="${V%%:*}"; REST="${V#*:}"; CM="${REST%%:*}"; WC="${REST#*:}"
  Z1=$((Z0 + 19))
  ONE="$PROJ/newscan_z10/one_coast_z${Z0}"
  if [ ! -d "$ONE" ]; then
    mkdir -p "$ONE"
    ln -s "$PAIRS/volume_$(printf %04d $Z0)-$(printf %04d $Z1)"*.h5 "$ONE/"
  fi
  OUT="$BASE/z${Z0}_cm${CM}_${WC}"
  rm -rf "$OUT"; mkdir -p "$OUT"
  echo
  echo "############ z${Z0}-${Z1}  coast-max=${CM}  walk-class=${WC} ############"
  python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 \
      --ct-dir 01_Mickey_sprockets \
      --seg-dir "$ONE" \
      --out-dir "$OUT" \
      --slice-frac 1.0 \
      --n-centers 1 \
      --seam-deg "$SEAM" \
      --smooth-px 0.75 --snap-max 10 --step 2.5 \
      --coast-max "$CM" --walk-class "$WC" \
      --film-min-thick 7.5 --jump-max 5 --snap-accept 7.5 --max-dr 2.5 \
      --recenter-search 7.5 --recenter-sigma 25 \
      --min-sep 7.5 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
      --z-heal-px 7.5 --min-coverage 8 \
      --transverse-px 1 --max-infill 3 --radial-robust \
      --seam-exclude-deg 0.3 \
      --inspect-anchors --inspect-max 3 \
    || { echo "WALK FAILED at z$Z0 cm$CM $WC"; continue; }
done

echo
echo "=================== SUMMARY ==================="
echo "Read against the ten-chunk run: GOOD = ~259,000 cols, ~34 windings,"
echo "arc-stretch med +2.3..+3.3%.  FAILED = 44,000-57,000 cols, med +123..+470%."
grep -E '^############|whole roll:|arc-columns:' \
     "/data/user/li_k1/M_thesis/logs/perfband_coast_${SLURM_JOB_ID}.out" || true
echo
ls -d "$BASE"/*/
