#!/bin/bash
# THE new-scan render. One script, one output directory -- re-run it rather than
# making a new path per attempt. Layout (all under unwrapping/inr/results/newscan):
#   walks/    per-range walk caches   (b0..b4, low_sparse, tail_dense, ...)
#   render/   current strip + z_index + overview, OVERWRITTEN each run
#   videos/   mp4s, kept, named by tag
#   diag/     diagnostics
#
#   sbatch Scripts/slurm/slurm_newscan_render.sh [tag] [low-band cache name]
#
# The low-band cache is selectable because it is the one slot that has bitten us:
# swapping the dense b0 for the quarter-density low_sparse put flat bright blocks
# into z679-873 at ~5.2s (measured: 3 frames vs 1 edge-noise frame in the b0
# render), almost certainly the winding tracker breaking across a junction between
# caches of very different anchor density. b1's trim is derived from whatever the
# low cache actually ends at, so the two never overlap.
#
# WHICH CACHES AND WHY (each --z-range trims an overlap; interp1d needs strictly
# increasing z, so no cache may repeat another's z):
#
#   low_sparse   z520-779   NOT b0. b0 was walked from z470, which
#                           diag_band_edges measures as still perforation
#                           (gaps/ring 11.1 vs ~5.0 in the picture band), and its
#                           find_spool_center samples were taken on that
#                           material: b0's centre track spans 92 px with 46
#                           anchors 94 px off a robust line. low_sparse is walked
#                           from z520 and its centres sit 0.38 px off, joining
#                           b1 to within 0.9 px. It is 65 anchors against b0's
#                           300, i.e. ~4-slice gaps -- acceptable (the roll
#                           drifts << 1 px/slice) and a far smaller error than a
#                           94 px centre.
#   b1 780:      z780-1069  trimmed so it does not overlap low_sparse
#   b2           z1070-1369
#   b3 :1589     z1370-1589 trimmed below tail_dense
#   tail_dense   z1590-1949 the full-density re-walk. CUT AT 1949, NOT 1969:
#                           z1969 is already inside the upper perforation band
#                           (bands visibly dashing in the overlay; radial jumps
#                           2-9 at z1964-1969 against 0.0 through z1949).
#                           z1950-2400 renders from clamped z1949 geometry.
#
# Below z520 and above z1949 the renderer clamps to the nearest anchor, which is
# what puts usable geometry under the perforation rows and margins.
#
# --seg-dir is REQUIRED by argparse even with --use-cached-anchors (the loader
# only takes the else branch when the cache is missing) -- any real directory does.
# --seam-exclude-deg 0.3 is LOAD-BEARING and is NOT the argparse default of 6.0;
# at 6.0 the walk throws away a 12 deg wedge instead of 0.6 deg.
#
# --no-gap-infill: gap-infill FABRICATES a winding wherever the radius gap between
# two kept windings exceeds 1.5x the median, by blending the two neighbours
# (_blend). It has no emulsion behind it. The first lowfix render inserted exactly
# one, which accounted for the whole apparent gain over v0fix (264948 - 258887 =
# 6061 cols ~ one winding, 233 cells vs 229) and rendered as the pale blocky
# tearing at ~5 s. v0fix inserted 0 and had no such tearing. Off until a dropped
# track is shown NOT to be sitting in the gap -- the new "tracks dropped" report
# prints exactly that, and if a real track is in there the fix is a lower
# --min-coverage, not a synthetic winding.
#
#SBATCH --job-name=nsrender
#SBATCH --partition=daily
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/nsrender_%j.out

set -euo pipefail
TAG=${1:-lowfix}
LOW=${2:-low_sparse}
shift 2 2>/dev/null || true
EXTRA=("$@")          # appended LAST to the walk command, so they override
if [ ${#EXTRA[@]} -gt 0 ]; then echo "extra render args: ${EXTRA[*]}"; fi
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

R="$PROJ/unwrapping/inr/results/newscan"
W="$R/walks"; OUT="$R/render"
mkdir -p "$OUT" "$R/videos"

B1_FROM=$(python - "$W/$LOW/walk_anchors.npz" <<'PY'
import sys, numpy as np
print(int(np.load(sys.argv[1], allow_pickle=True)["z_anchor"].max()) + 1)
PY
)
echo "low-band cache: $LOW  ->  b1 trimmed to start at z$B1_FROM"
python -u Scripts/merge_walk_caches.py \
    --out "$OUT/walk_anchors.npz" --refit-centers \
    --caches "$W/$LOW/walk_anchors.npz" \
             "$W/b1/walk_anchors.npz" \
             "$W/b2/walk_anchors.npz" \
             "$W/b3/walk_anchors.npz" \
             "$W/tail_dense/walk_anchors.npz" \
    --z-range - "${B1_FROM}:" - :1589 1590:1949

python -u Scripts/diag_centers.py --zbin 100 --caches "$OUT/walk_anchors.npz"

python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 \
    --ct-dir 01_Mickey_sprockets --seg-dir "$PROJ/newscan_low/pairs" \
    --out-dir "$OUT" --use-cached-anchors \
    --seam-deg 353.0 --ref-mode track \
    --min-sep 13 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
    --z-heal-px 7.5 --min-coverage 300 --no-gap-infill \
    --transverse-px 3 --seam-exclude-deg 0.3 --seam-bridge \
    "${EXTRA[@]}"

python -u -m unwrapping.inr.make_film_video \
    "$OUT/wholeroll.npy" "$R/videos/film_${TAG}.mp4" \
    --phase-lock global --invert-phase --reverse --invert --rotate --height 720

ls -la "$OUT" "$R/videos"
echo "=== render $TAG DONE ==="
