#!/bin/bash
#SBATCH --job-name=bordfinal
#SBATCH --partition=daily
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/user/li_k1/M_thesis/unwrapping/eval/results/bordfinal_%j.out

# Final border-locked render: framing fixed against GT's aperture, plus the
# GT-anchored along-film offsets folded into the strip sampling.
#
# Framing comes in two parts and BOTH are needed:
#   --along-shift 0.156 = 0.128 (measured against GT's aperture on the first
#     aligned render) + 0.028 (the residual that render still had).  The folded
#     offsets cannot supply this: stabilize_horizontal normalises its own
#     offsets to zero median, and border_render subtracts the median again, so
#     a CONSTANT framing error passes through the whole chain untouched.  That
#     is why the previous flagship kept its 0.128-pitch error despite carrying
#     a GT-anchored correction.
#   --across-shift 0.012 --across-scale 0.941 puts our window on GT's aperture,
#     removing the ~4 % overshoot that showed a perforation at one edge.
#
# NOOFF is the control: identical framing, no folded offsets, so the offsets'
# contribution is measurable rather than assumed.

set -e
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd /data/user/li_k1/M_thesis

STRIP=unwrapping/inr/results/newscan_render1/wholeroll.npy
BORDER=unwrapping/inr/results/newscan_border/border.npz
OFF=unwrapping/eval/results/stabilization/border/along_offsets_aligned.npy
OUT=unwrapping/eval/results/stabilization/renders/final
mkdir -p "$OUT"

FRAME="--along-shift 0.156 --across-shift 0.012 --across-scale 0.941"
COMMON="--border $BORDER --out-dir $OUT --along phaselock --smooth-frames 0.8 \
        --reverse --invert --height 720 $FRAME"
FOLD="--along-offsets $OFF --along-offsets-sign +1"

echo "=== FINAL margin 0.04 (border visible) ==="
python -m unwrapping.eval.border_render "$STRIP" $COMMON $FOLD \
    --margin 0.04 --video film_final_m04.mp4

echo "=== FINAL margin 0 (exact aperture) ==="
python -m unwrapping.eval.border_render "$STRIP" $COMMON $FOLD \
    --margin 0.0 --video film_final_m00.mp4

echo "=== CONTROL: same framing, no folded offsets ==="
python -m unwrapping.eval.border_render "$STRIP" $COMMON \
    --margin 0.0 --video film_nooff_m00.mp4

ls -la "$OUT"
echo DONE
