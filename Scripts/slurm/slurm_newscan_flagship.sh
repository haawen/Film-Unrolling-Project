#!/bin/bash
# STAGE B: maximum stabilisation of the new-scan render, GT allowed.
#
# Runs after slurm_newscan_borderlock.sh (stage A, GT-free across-film lock).
# Ladder, in increasing stillness and decreasing faithfulness:
#   1. border-locked          across-film locked to the film's own printed
#                             border. GT-free. (stage A)
#   2. + along-film offsets   GT-anchored TRANSLATION only -- slides our pixels,
#                             never reprojects. On the old flagship this was
#                             worth 69% -> 91% of GT median sharpness.
#   3. + affine               scale/rotation/shear fitted to GT. Still our
#                             pixels (no dense reprojection), reached 118% of GT.
#
# GT IS THE PERF-LOCKED GT, NOT THE RAW SCAN. Raw GT wanders 21.7 px across-film
# -- it is not a still target, and warping onto it previously wrecked the
# across-film lock. gt_perflock.py re-cut it on its OWN real perforations
# (4 per frame, full affine) taking its picture-border stillness from
# across 21.74 -> 4.31 px. `--gt-crop 0,0,1,1` because that video is ALREADY
# cropped to the picture; the 0.235,0.12,0.81,0.88 default would crop it twice.
#
# OFFSET PLUMBING IS HANDLED INSIDE border_render -- do not pre-process the .npy.
# It reverses frame order itself (the render uses --reverse, so frame i is cell
# n-1-i), derives the strip-px-per-offset-px scale from its own geometry, and
# pads to length. Sign is +1: np.rot90 makes video row = nC-1-along, so a window
# shift of +delta moves content DOWN, the same direction as nd_shift(+y)
# (verified empirically: +1 scored 6.56, -1 gave 11.02).
#
# Framing constants carried verbatim from the tuned recipe -- a 0.128-pitch
# framing error survives the whole chain silently (stabilize_horizontal
# zero-medians its offsets and border_render subtracts the median again) and is
# what once made affine look catastrophic, its per-block search being smaller
# than the systematic offset.
#
#   sbatch Scripts/slurm/slurm_newscan_flagship.sh
#
#SBATCH --job-name=flagship
#SBATCH --partition=daily
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/flagship_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

R="$PROJ/unwrapping/inr/results/newscan"
STRIP="$R/render/wholeroll.npy"
BDIR="$R/border"; ODIR="$R/stabilized"
GT="$PROJ/unwrapping/eval/results/stabilization/gt/gt_perflock_picture.mp4"
mkdir -p "$ODIR"
[ -f "$BDIR/border.npz" ] || { echo "missing border.npz -- run stage A first"; exit 1; }
[ -f "$GT" ] || { echo "missing perf-locked GT"; exit 1; }

COMMON="--border $BDIR/border.npz --out-dir $ODIR --along phaselock \
        --smooth-frames 0.8 --reverse --invert --height 720 \
        --along-shift 0.156 --across-shift 0.0 --across-scale 1.0"

echo "=== [1/4] GT-anchored along-film offsets (translation only) ==="
python -u -m unwrapping.eval.stabilize_horizontal \
    --in "$ODIR/film_borderlock_m02.mp4" --out "$ODIR/_scratch_h.mp4" \
    --gt "$GT" --gt-crop 0,0,1,1 --axes v \
    --save-offsets "$ODIR/along_offsets.npy"

echo "=== [2/4] fold them into the sampling -> FLAGSHIP ==="
python -u -m unwrapping.eval.border_render "$STRIP" $COMMON \
    --along-offsets "$ODIR/along_offsets.npy" --along-offsets-sign +1 \
    --margin 0.02 --video FLAGSHIP_borderlock.mp4

echo "=== [3/4] affine on top -> stabilised VIEWING COPY ==="
python -u -m unwrapping.eval.stabilize_affine \
    --in "$ODIR/FLAGSHIP_borderlock.mp4" --gt "$GT" --gt-crop 0,0,1,1 \
    --mode affine --out "$ODIR/FLAGSHIP_borderlock_affine.mp4"

echo "=== [4/4] arbiter: temporal-median sharpness, central 70%, common size ==="
# Inner crop and a common size, or the metric is flattered by a lock holding its
# own crop still, and full-frame it ranks a MIS-FRAMED render first (its framing
# error drags a bright perforation into the crop = free contrast).
python -u -m unwrapping.eval.border_stillness \
    --videos "$R/videos/film_safe.mp4" \
             "$ODIR/film_borderlock_m02.mp4" \
             "$ODIR/FLAGSHIP_borderlock.mp4" \
             "$ODIR/FLAGSHIP_borderlock_affine.mp4" \
             "$GT" \
    --labels baseline borderlock flagship flagship_affine GT_perflock \
    --margins 0.15 0.15 0.15 0.15 0.15 \
    --size 720 720 --out-dir "$ODIR"

rm -f "$ODIR/_scratch_h.mp4"
ls -la "$ODIR"
echo "=== flagship DONE ==="
