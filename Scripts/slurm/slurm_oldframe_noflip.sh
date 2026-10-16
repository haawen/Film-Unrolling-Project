#!/bin/bash
#SBATCH --job-name=oldfrnf
#SBATCH --partition=daily
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/data/user/li_k1/M_thesis/unwrapping/eval/results/oldframe_noflip_%j.out

# TASK A: run the OLD scan's unrolling->affine chain on the NEW full-width strip.
#
# The old chain (frame_match.py -> stabilize_affine.py) produced the affine
# videos that worked, and its frame is 822x720 (aspect 1.142) because
# frame_match sizes its cells to the GT crop 0.235,0.12,0.81,0.88 -- NOT the
# 1.347 aspect of the full film frame.  So "cut our video exactly like the old
# one" means running this chain, which cuts and GT-locks each cell itself.
#
# --z-crop is the one thing that must change for the new scan: its default
# 0.05,0.95 spans almost the whole strip, and the new strip is full-width, so
# that would include BOTH perforation rows.  0.1985,0.8431 is the detected
# picture band (strip rows 468.7-1990.6 of 2361), recovered from
# border_track.npz by undoing the shift/scale it stores post-adjustment.
# --pitch is passed explicitly rather than auto-fitted.

set -e
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd /data/user/li_k1/M_thesis

STRIP=unwrapping/inr/results/newscan_render1/wholeroll.npy
OUT=unwrapping/eval/results/stabilization/oldframe_noflip
mkdir -p "$OUT"

echo "=== frame_match on the new strip, old-scan framing ==="
python -m unwrapping.eval.frame_match \
    --strip "$STRIP" --gt data/h265_1080p.mp4 --out-dir "$OUT" \
    --z-crop 0.1985,0.8431 --pitch 1130.06 --no-fliplr \
    --export-video --montage --video

echo "=== affine on it, with the OLD gt-crop that frame_match used ==="
python -m unwrapping.eval.stabilize_affine \
    --in "$OUT/film_stabilized.mp4" --gt data/h265_1080p.mp4 \
    --gt-crop 0.235,0.12,0.81,0.88 --mode affine --extra-crop 0 \
    --out "$OUT/film_oldframe_noflip_affine.mp4"

ls -la "$OUT"
echo DONE
