#!/bin/bash
# Stage 2: walk each of the ten scattered chunks SEPARATELY and draw a full-slice
# fit overlay for each, plus a CT-referenced missed-winding census.
#
# Separately, not one run over all ten: the chunks are ~220 slices apart, and the
# z-sequential tracker is built for contiguous z. Pooling them would make the
# overlays show TRACKING failure across the gaps rather than WALK quality, which
# is what the fit check is for. Each chunk therefore gets its own cache, its own
# centre, and its own overlay -- directly comparable to the z500-519 one already
# judged.
#
# Seam 353.0: measured three independent ways (r_out step detector +22.2 px SNR
# 5.2x at 353.0; outermost walked winding terminating at 355.1; and the CT itself
# at az 353 r 1700 showing the band end). The 133.5 used by tests 2-5 was wrong
# and produced a radial zigzag at abs ~136 on z500-506.
# --seam-exclude-deg 0.3 is the canonical v12/v13 value, NOT the argparse 6.0.
#
#   sbatch --dependency=afterok:<seg10 jobid> Scripts/slurm/slurm_newscan_walk10.sh
#
#SBATCH --job-name=newscan_walk10
#SBATCH --partition=daily
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/newscan_walk10_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

PAIRS="$PROJ/newscan_z10/pairs"
BASE="$PROJ/unwrapping/inr/results/newscan_z10"
mkdir -p "$BASE"
SEAM=353.0
Z0S="250 560 780 1000 1220 1440 1660 1880 2060 2200"

for Z0 in $Z0S; do
  Z1=$((Z0 + 19))
  OUT="$BASE/z${Z0}"
  ONE="$PROJ/newscan_z10/one_z${Z0}"
  rm -rf "$OUT" "$ONE"; mkdir -p "$OUT" "$ONE"
  # --seg-dir globs a whole directory, so give each walk a dir holding only its
  # own chunk (symlinks, no copy of the ~2 GB probability maps).
  ln -s "$PAIRS/volume_$(printf %04d $Z0)-$(printf %04d $Z1)"*.h5 "$ONE/"

  echo
  echo "################## z${Z0}-${Z1} ##################"
  python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 \
      --ct-dir 01_Mickey_sprockets \
      --seg-dir "$ONE" \
      --out-dir "$OUT" \
      --slice-frac 1.0 \
      --n-centers 1 \
      --seam-deg "$SEAM" \
      --smooth-px 0.75 --snap-max 10 --step 2.5 --coast-max 50 \
      --film-min-thick 7.5 --jump-max 5 --snap-accept 7.5 --max-dr 2.5 \
      --recenter-search 7.5 --recenter-sigma 25 \
      --min-sep 7.5 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
      --z-heal-px 7.5 --min-coverage 8 \
      --transverse-px 1 --max-infill 3 --radial-robust \
      --seam-exclude-deg 0.3 \
      --inspect-anchors --inspect-max 3 || { echo "WALK FAILED at z$Z0"; continue; }

  echo "--- full-slice fit overlay (mid anchor) ---"
  python -u -m unwrapping.inr.inspect_walk_hires \
      --cache-dir "$OUT" --ct-dir 01_Mickey_sprockets \
      --anchors 10 --wedge-az-deg 0,180 \
      --out-dir "$OUT/hires" || echo "  overlay failed at z$Z0"

  echo "--- missed-winding census vs the CT ---"
  python -u Scripts/diag_missed_windings.py \
      --npz "$OUT/matched_walks.npz" --ct-dir 01_Mickey_sprockets \
      --anchors 0,10,19 || echo "  census failed at z$Z0"
done

echo
echo "=== ALL DONE ==="
find "$BASE" -name "*.png" | sort
