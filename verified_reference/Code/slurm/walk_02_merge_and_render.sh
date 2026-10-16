#!/bin/bash
# STAGE 4 - merge the five walks, render the strip (2361 rows = every CT slice
# z40-2400, incl. both perforation rows) and cut the plain film video (no GT).
#
#   sbatch Code/slurm/walk_02_merge_and_render.sh          # FINAL model (shipped walk caches)
#   sbatch Code/slurm/walk_02_merge_and_render.sh early    # EARLY version (requested video 1a)
#   WALKS_FROM=work sbatch Code/slurm/walk_02_merge_and_render.sh   # your walk_01 outputs
#
# Merge: b0 | b1 from where b0 ends | b2 | b3 up to z1589 | tail z1590-1949
# (z1950+ is already inside the upper perforation transition). Outside z470-1949
# the renderer clamps to the nearest walked slice (perforation rows + margins).
# final: strip columns on uniform film ARC LENGTH, 4-tap transverse average across
#        the ~5 px emulsion.
#        Pitch fixed to 1130 px: the automatic estimate on this strip lands on
#        1139 px (226 cells, framing creeps), while the same estimator gives
#        1130.1 px on the 2- and 3-tap renders of identical geometry.
# early: the same walk without the two later refinements: columns on a uniform
#        AZIMUTH grid (--no-arc-columns) and one sample on the emulsion centre line
#        (--transverse-px 1); automatic pitch (azimuth columns change the length).
# Output: $WORK/<final_render|early_version>/{walk_anchors.npz, wholeroll.npy, film_<variant>.mp4}
#
#SBATCH --job-name=render
#SBATCH --partition=daily
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --output=work/logs/%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/Code/slurm/_activate.sh"
VARIANT=${1:-final}
case "$VARIANT" in
  final) OUT="$WORK/final_render";  RENDER="--transverse-px 4";                   PITCH="--pitch 1130" ;;
  early) OUT="$WORK/early_version"; RENDER="--no-arc-columns --transverse-px 1"; PITCH="" ;;
  *) echo "usage: walk_02_merge_and_render.sh [final|early]"; exit 1 ;;
esac
mkdir -p "$OUT"
if [ "${WALKS_FROM:-shipped}" = work ]; then
    K="$WORK/walks"; C0=$K/b0/walk_anchors.npz; C1=$K/b1/walk_anchors.npz
    C2=$K/b2/walk_anchors.npz; C3=$K/b3/walk_anchors.npz; CT_=$K/tail/walk_anchors.npz
else
    K="$D_SCAN/walk_caches"; C0=$K/b0.npz; C1=$K/b1.npz; C2=$K/b2.npz; C3=$K/b3.npz; CT_=$K/tail_dense.npz
fi
B1_FROM=$(python -c "import numpy as np; print(int(np.load('$C0', allow_pickle=True)['z_anchor'].max()) + 1)")
echo "=== [1/3] merge walks (b1 starts at z$B1_FROM) ==="
python -u -m virtual_unrolling.emulsion_walk.merge_walk_caches --out "$OUT/walk_anchors.npz" \
    --refit-centers --caches "$C0" "$C1" "$C2" "$C3" "$CT_" \
    --z-range - "${B1_FROM}:" - :1589 1590:1949
python -u -m virtual_unrolling.emulsion_walk.centre_track_report --zbin 100 --caches "$OUT/walk_anchors.npz"

echo "=== [2/3] render the strip ($VARIANT) ==="
# --seg-dir is required by the argument parser even with --use-cached-anchors
python -u -m virtual_unrolling.emulsion_walk.unroll_whole_roll \
    --ct-dir "$D_SCAN/ct_slices" --seg-dir "$D_SCAN/segmentation_nnunet3d_picture_band" \
    --out-dir "$OUT" --use-cached-anchors --seam-deg 353.0 \
    --min-sep 13 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
    --z-heal-px 7.5 --min-coverage 300 \
    $RENDER --seam-exclude-deg 0.3 --seam-bridge

echo "=== [3/3] plain film video (no ground truth) ==="
python -u -m virtual_unrolling.video.make_film_video "$OUT/wholeroll.npy" "$OUT/film_$VARIANT.mp4" \
    $PITCH --invert-phase --reverse --invert --rotate --height 720
echo "=== done: $OUT ==="
