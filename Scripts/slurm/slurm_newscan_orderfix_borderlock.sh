#!/bin/bash
# The BORDER-LOCK chain behind compare/gt_vs_newscan_affine.mp4 (= stabilized/
# FLAGSHIP_borderlock_affine.mp4), re-run VERBATIM on the --enforce-radial-order
# strip, into NEW dirs so the delivered outputs stay untouched:
#   stage A (slurm_newscan_borderlock.sh): detect_border -> border_render m02
#   stage B (slurm_newscan_flagship.sh):   stabilize_horizontal --gt -> border_render
#            with along offsets -> stabilize_affine
# Flags are copied exactly, INCLUDING stage B's stabilize_affine without
# --no-gt-trim (doc s12: that mis-pairs the perf-locked GT) -- kept on purpose so
# the ONLY difference from the delivered video is the track-swap fix.
# Run after slurm_newscan_orderfix_chain.sh has rendered render/wholeroll.npy:
#   sbatch --dependency=afterok:<chain job> Scripts/slurm/slurm_newscan_orderfix_borderlock.sh
#
#SBATCH --job-name=nsofixbl
#SBATCH --partition=daily
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/nsofixbl_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export PYTHONPATH="$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

R="$PROJ/unwrapping/inr/results/newscan"
STRIP=${STRIP:-$R/render/wholeroll.npy}   # override to run the chain on a post-processed strip
SFX=${SFX:-}   # e.g. _tw4 -- keeps each strip variant in its own dirs
BDIR="$R/border_orderfix$SFX"; ODIR="$R/stabilized_orderfix$SFX"
GT="$PROJ/unwrapping/eval/results/stabilization/gt/gt_perflock_picture.mp4"
mkdir -p "$BDIR" "$ODIR"
grep -q "radial-order:" "$R/render/render_cmd.log" || { echo "render/ is not an order-fix render"; exit 1; }

COMMON="--border $BDIR/border.npz --out-dir $ODIR --along phaselock \
        --smooth-frames 0.8 --reverse --invert --height 720 \
        --along-shift 0.156 --across-shift 0.0 --across-scale 1.0"
# Intensity-only post-processed strips (band_correct): frame them EXACTLY like their
# unprocessed source -- PHASE_STRIP gives the phase lock's pitch/phase, BORDER_FROM
# the detected border -- so the A/B differs in brightness only.
[ -n "${PHASE_STRIP:-}" ] && COMMON="$COMMON --phase-strip $PHASE_STRIP"

if [ -n "${BORDER_FROM:-}" ]; then
    echo "=== [A1] reuse border $BORDER_FROM ==="
    cp -v "$BORDER_FROM" "$BDIR/border.npz"
else
    echo "=== [A1] detect_border $(date) ==="
    python -u -m unwrapping.eval.detect_border "$STRIP" --out-dir "$BDIR"
fi
echo "=== [A2] border_render m02 $(date) ==="
python -u -m unwrapping.eval.border_render "$STRIP" $COMMON \
    --margin 0.02 --video film_borderlock_m02.mp4

echo "=== [B1] GT along-film offsets $(date) ==="
python -u -m unwrapping.eval.stabilize_horizontal \
    --in "$ODIR/film_borderlock_m02.mp4" --out "$ODIR/_scratch_h.mp4" \
    --gt "$GT" --gt-crop 0,0,1,1 --axes v \
    --save-offsets "$ODIR/along_offsets.npy"
echo "=== [B2] FLAGSHIP render $(date) ==="
python -u -m unwrapping.eval.border_render "$STRIP" $COMMON \
    --along-offsets "$ODIR/along_offsets.npy" --along-offsets-sign +1 \
    --margin 0.02 --video FLAGSHIP_borderlock.mp4
echo "=== [B3] affine $(date) ==="
python -u -m unwrapping.eval.stabilize_affine \
    --in "$ODIR/FLAGSHIP_borderlock.mp4" --gt "$GT" --gt-crop 0,0,1,1 \
    --mode affine --out "$ODIR/FLAGSHIP_borderlock_affine.mp4"

echo "=== arbiter vs the delivered flagship $(date) ==="
python -u -m unwrapping.eval.border_stillness \
    --videos "$R/stabilized/FLAGSHIP_borderlock_affine.mp4" \
             "$ODIR/FLAGSHIP_borderlock_affine.mp4" "$GT" \
    --labels delivered_affine orderfix_affine GT_perflock \
    --margins 0.15 0.15 0.15 --size 720 720 --out-dir "$ODIR"

rm -f "$ODIR/_scratch_h.mp4"
ls -la "$BDIR" "$ODIR"
echo "=== nsofixbl DONE $(date) ==="
