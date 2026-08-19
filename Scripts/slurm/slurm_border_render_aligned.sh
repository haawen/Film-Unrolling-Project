#!/bin/bash
#SBATCH --job-name=bordalign
#SBATCH --partition=daily
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/user/li_k1/M_thesis/unwrapping/eval/results/bordalign_%j.out

# Re-render the border-locked film with the framing corrected against GT's
# aperture, which is fixed to GT's OWN (real, scanned) perforations.
#
# The previous render framed a complete picture but sat 0.128 pitch off along
# the film -- the frame line landed 13 % up from the bottom edge with a sliver
# of the next frame below it -- and ran ~4 % wider than the aperture across the
# film, which is what put a perforation at one edge.  A 0.128-pitch offset is
# ~92 px of a 720-row frame, far beyond the +-36 px per-block search the affine
# stabiliser uses, so every block locked onto the wrong feature.  That, not the
# affine method, is why the affine output was "severely wrong".
#
# V3 renders the opposite along-shift purely as a sign check.

set -e
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd /data/user/li_k1/M_thesis

STRIP=unwrapping/inr/results/newscan_render1/wholeroll.npy
BORDER=unwrapping/inr/results/newscan_border/border.npz
OUT=unwrapping/eval/results/stabilization/renders/aligned
mkdir -p "$OUT"

COMMON="--border $BORDER --out-dir $OUT --along phaselock --smooth-frames 0.8 \
        --reverse --invert --height 720"

echo "=== V1 aligned, margin 0.04 (border visible) ==="
python -m unwrapping.eval.border_render "$STRIP" $COMMON \
    --along-shift 0.128 --across-scale 0.960 --across-shift -0.016 \
    --margin 0.04 --video film_aligned_m04.mp4

echo "=== V2 aligned, margin 0 (exact aperture; affine input) ==="
python -m unwrapping.eval.border_render "$STRIP" $COMMON \
    --along-shift 0.128 --across-scale 0.960 --across-shift -0.016 \
    --margin 0.0 --video film_aligned_m00.mp4

echo "=== V3 sign check: opposite along-shift ==="
python -m unwrapping.eval.border_render "$STRIP" $COMMON \
    --along-shift -0.128 --across-scale 0.960 --across-shift -0.016 \
    --margin 0.04 --video film_signcheck_m04.mp4

ls -la "$OUT"
echo DONE
