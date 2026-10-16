#!/bin/bash
# STAGE 3 - EMULSION WALK on one z-range of the picture band.
# The final render merges five walks (each a separate job; run them in parallel):
#
#   for B in b0 b1 b2 b3 tail; do sbatch Code/slurm/walk_01_walk_range.sh $B; done
#
#   b0  z470-769   b1  z770-1069   b2  z1070-1369   b3  z1370-1669    (15 chunks each)
#   tail z1590-1969 (19 chunks, the full-density re-walk of the upper band)
#
# Every slice of the segmentation is walked. The spacing-keyed
# thresholds are scaled to this scan's 26 px winding spacing. The seam MUST be
# forced to the measured 353 deg (param_01_seam_and_picture_band.sh): the automatic
# seam detector fails on this scan. Perforation rows / margins are not walked (the
# perforations break the emulsion; their geometry is clamped from the nearest
# picture-band slice at render time).
# Output: $WORK/walks/<B>/{walk_anchors.npz, matched_walks.npz, insp_*.png montage}
# The shipped caches Data/3_full_width_ct_scan/walk_caches/<B>.npz are these walks.
#
#SBATCH --job-name=walk_range
#SBATCH --partition=daily
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=work/logs/%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/Code/slurm/_activate.sh"
B=${1:?usage: walk_01_walk_range.sh b0|b1|b2|b3|tail}
SEG_ALL=${SEG:-$D_SCAN/segmentation_nnunet3d_picture_band}
case "$B" in
  b0) Z0=470;  Z1=769  ;;  b1) Z0=770;  Z1=1069 ;;
  b2) Z0=1070; Z1=1369 ;;  b3) Z0=1370; Z1=1669 ;;
  tail) Z0=1590; Z1=1969 ;;
  *) echo "unknown range $B"; exit 1 ;;
esac
OUT="$WORK/walks/$B"; LNK="$OUT/seg_links"
rm -rf "$OUT"; mkdir -p "$LNK"
# the walker takes every chunk in --seg-dir: link just this range
for f in "$SEG_ALL"/volume_*_Probabilities.h5; do
    z=$(basename "$f" | sed -E 's/volume_0*([0-9]+)-.*/\1/')
    if [ "$z" -ge "$Z0" ] && [ "$z" -le "$Z1" ]; then ln -s "$f" "$LNK/"; fi
done
echo "$B: $(ls "$LNK" | wc -l) chunks z$Z0-$Z1"

COMMON="--ct-dir $D_SCAN/ct_slices --seg-dir $LNK --out-dir $OUT --seam-deg 353.0 \
  --smooth-px 0.75 --snap-max 10 --step 2.5 --coast-max 50 --film-min-thick 7.5 --jump-max 5 \
  --snap-accept 7.5 --max-dr 2.5 --recenter-search 7.5 --recenter-sigma 25 \
  --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 --z-heal-px 7.5 \
  --transverse-px 3 --max-infill 3 --radial-robust --seam-exclude-deg 0.3"
if [ "$B" = tail ]; then
    python -u -m virtual_unrolling.emulsion_walk.unroll_whole_roll $COMMON \
        --n-centers 8 --min-sep 13 --min-coverage 20 --inspect-anchors --inspect-max 6
else
    python -u -m virtual_unrolling.emulsion_walk.unroll_whole_roll $COMMON \
        --n-centers 6 --min-sep 7.5 --min-coverage 100 --inspect-anchors --inspect-max 3
fi
python -u -m virtual_unrolling.emulsion_walk.centre_track_report --zbin 50 --caches "$OUT/walk_anchors.npz"
echo "=== walk $B done: $OUT ==="
