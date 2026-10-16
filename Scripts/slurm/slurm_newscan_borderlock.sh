#!/bin/bash
# BORDER-LOCKED STABILISATION of the current best new-scan render.
#
# Runs the chain that [[perf-locked-stabilization]] established, on the new strip:
#   detect_border  -> border.npz (per-frame picture-border track)
#   border_render  -> a video locked ACROSS the film to the film's own printed
#                     border, GT-FREE
#   border_stillness -> the arbiter
#
# WHY THE BORDER AND NOT THE SPROCKETS, even though this scan has both perf rows:
# our own perforations are CLOSED on both axes. The perf bands are rendered from
# geometry CLAMPED outside the walked anchor range, so they do not carry the
# picture's geometry. The held-out test settles it -- the two perf rows sit on
# opposite edges of the SAME film, so real motion would make them agree, and they
# correlate +0.06..+0.30 along and -0.07..+0.31 across, with the edge-to-edge
# difference sd LARGER than either edge's own scatter. The picture border lies
# INSIDE the walked band and passes that same test (-0.45 raw / -0.65 smoothed:
# anti-correlated = the picture breathes, which is the real defect).
#
# FRAMING CONSTANTS ARE LOAD-BEARING, do not "clean them up":
#   --along-shift 0.156   the render is otherwise mis-framed by ~0.128 pitch, and
#                         that error silently survives the whole stabilisation
#                         chain (stabilize_horizontal zero-medians its offsets and
#                         border_render subtracts the median again). It is also
#                         what made affine look catastrophic once -- a systematic
#                         offset larger than affine's per-block search range.
#   --across-shift 0.0
#   --across-scale 1.0    do NOT chase GT's aperture box: it is "the darkest
#                         column inboard of the perforation", measures 10.75mm vs
#                         the ISO aperture's 10.26, and chasing it cut 4.4% off
#                         the left of the picture. Our detected border puts the
#                         edges on the frame edges by construction.
#
# STAGE A ONLY (GT-free). The GT-anchored along-film translations that took the
# old flagship from 69% to 91% of GT are NOT reused here: the existing
# along_offsets_aligned.npy is (228,2), tied to render1's cut and its geometry,
# while this strip is a different render at 229 cells. They must be regenerated
# against this strip, which is stage B.
#
#   sbatch Scripts/slurm/slurm_newscan_borderlock.sh
#
#SBATCH --job-name=borderlock
#SBATCH --partition=daily
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/borderlock_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

R="$PROJ/unwrapping/inr/results/newscan"
STRIP="$R/render/wholeroll.npy"
BDIR="$R/border"; ODIR="$R/stabilized"
mkdir -p "$BDIR" "$ODIR"

echo "=== [1/3] detect the printed picture border ==="
python -u -m unwrapping.eval.detect_border "$STRIP" --out-dir "$BDIR"

echo "=== [2/3] border-locked render (across-film lock, GT-free) ==="
COMMON="--border $BDIR/border.npz --out-dir $ODIR --along phaselock \
        --smooth-frames 0.8 --reverse --invert --height 720 \
        --along-shift 0.156 --across-shift 0.0 --across-scale 1.0"
python -u -m unwrapping.eval.border_render "$STRIP" $COMMON \
    --margin 0.02 --video film_borderlock_m02.mp4
python -u -m unwrapping.eval.border_render "$STRIP" $COMMON \
    --margin 0.0  --video film_borderlock_m00.mp4

echo "=== [3/3] stillness: temporal-median sharpness, central 70% ==="
# Measured on an INNER crop at a common size, or the metric is flattered by a
# lock holding its own crop still, and full-frame it ranks a MIS-FRAMED render
# first because the framing error drags a bright perforation into the crop.
python -u -m unwrapping.eval.border_stillness \
    --videos "$ODIR/film_borderlock_m02.mp4" "$R/videos/film_safe.mp4" \
    --labels borderlock safe_baseline \
    --margins 0.15 0.15 --size 720 720 --out-dir "$ODIR"

ls -la "$BDIR" "$ODIR"
echo "=== borderlock DONE ==="
