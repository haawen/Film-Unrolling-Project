#!/bin/bash
#SBATCH --job-name=final2
#SBATCH --partition=daily
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/user/li_k1/M_thesis/unwrapping/eval/results/final2_%j.out

# TASK B: fix the over-cropped left border.
#
# The previous render used --across-shift 0.012 --across-scale 0.941, chasing
# GT's measured aperture box.  Working back through border_render's own
# parameterisation, that put OUR detected picture edges at output x = -0.044 and
# +1.019 -- i.e. it cut 4.4 % off the left of the picture and 1.9 % off the
# right, which is the asymmetric left-side loss the user saw.
#
# TWO mistakes produced it:
#  1. SIGN. frame_align_check suggests the same sign for both axes, but np.rot90
#     reverses the ALONG axis and not the ACROSS one, so across takes the
#     opposite sign.  The +0.012 pushed the window further right, the way it was
#     already wrong.
#  2. Chasing GT's aperture box at all.  That box is "the darkest column inboard
#     of the perforation" in GT's perf-locked mean and measures 10.75 mm, wider
#     than the ISO 10.26 mm aperture, so it is not obviously the printed picture
#     edge.  OUR detected border is the film's own printed border and is the
#     thing the whole border-lock is built on.
# => across-shift 0, across-scale 1.0: the detected picture edges land exactly
#    on the frame edges, by construction.
#
# ALONG framing keeps 0.156; that axis' sign was verified with an opposite-sign
# control render (-0.128 gave -28.3 % where +0.128 gave -2.8 %).

set -e
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd /data/user/li_k1/M_thesis

STRIP=unwrapping/inr/results/newscan_render1/wholeroll.npy
BORDER=unwrapping/inr/results/newscan_border/border.npz
OFF=unwrapping/eval/results/stabilization/border/along_offsets_aligned.npy
OUT=unwrapping/eval/results/stabilization/renders/final2
mkdir -p "$OUT"

COMMON="--border $BORDER --out-dir $OUT --along phaselock --smooth-frames 0.8 \
        --reverse --invert --height 720 \
        --along-shift 0.142 --across-shift 0.0 --across-scale 1.0 \
        --along-offsets $OFF --along-offsets-sign +1"

echo "=== margin 0: exactly the detected picture, borders at the frame edges ==="
python -m unwrapping.eval.border_render "$STRIP" $COMMON \
    --margin 0.0 --video film_final2_m00.mp4

echo "=== margin 0.02: thin sliver so the border stays visible as proof ==="
python -m unwrapping.eval.border_render "$STRIP" $COMMON \
    --margin 0.02 --video film_final2_m02.mp4

ls -la "$OUT"
echo DONE
