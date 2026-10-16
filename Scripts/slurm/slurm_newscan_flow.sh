#!/bin/bash
# STAGE C: the last rungs of the GT-anchored stabilisation ladder for the new
# full-width scan. Runs after slurm_newscan_flagship.sh (stage B).
#
#   stage A  border lock                 GT-free, across-film
#   stage B  + GT along-film translation FLAGSHIP_borderlock.mp4   (our pixels)
#            + affine (scale/rot/shear)  FLAGSHIP_borderlock_affine.mp4
#   stage C  + block-flow "dense"        coarse non-rigid, 10x10 grid
#            + dense Farneback flow      per-pixel, the stillest rung
#
# EVERYTHING IN THIS STAGE IS A VIEWING COPY. Affine still slides/scales OUR
# pixels; dense and flow REPROJECT them onto GT's geometry, so any residual
# unrolling distortion is hidden rather than fixed. No fidelity metric (SSIM,
# grad_corr, DISTS, frame_match) may be reported on a stage-C output -- it has
# been warped toward the very reference such a metric would score it against.
# Label figures accordingly. The honest fix stays geometric consistency in the
# emulsion walk.
#
# GT IS THE PERF-LOCKED GT (gt_perflock_picture.mp4), already cut to the picture
# and already leader-trimmed, hence BOTH `--gt-crop none` and `--no-gt-trim`:
# cropping twice eats the picture, trimming twice shifts the frame pairing by a
# few frames. stabilize_flow only grew --no-gt-trim today; without it this stage
# would have silently paired frame i with GT frame i+k.
#
# --extra-crop 0.02 (default 0.05) because stage B's affine already took 4% off
# every edge; the chained default would compound to ~9%.
#
#   sbatch Scripts/slurm/slurm_newscan_flow.sh
#
#SBATCH --job-name=nsflow
#SBATCH --partition=hourly
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/nsflow_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

R="$PROJ/unwrapping/inr/results/newscan"
ODIR="$R/stabilized"
GT="$PROJ/unwrapping/eval/results/stabilization/gt/gt_perflock_picture.mp4"
GTARGS="--gt $GT --gt-crop none --no-gt-trim"

[ -f "$ODIR/FLAGSHIP_borderlock.mp4" ] || { echo "missing stage B output"; exit 1; }
[ -f "$ODIR/FLAGSHIP_borderlock_affine.mp4" ] || { echo "missing stage B affine"; exit 1; }
[ -f "$GT" ] || { echo "missing perf-locked GT"; exit 1; }

echo "=== [1/4] block-flow (coarse non-rigid) on the translation-locked cut ==="
python -u -m unwrapping.eval.stabilize_affine \
    --in "$ODIR/FLAGSHIP_borderlock.mp4" $GTARGS \
    --mode dense --grid 10 --iters 3 --extra-crop 0.02 \
    --out "$ODIR/FLAGSHIP_borderlock_dense.mp4"

echo "=== [2/4] dense Farneback flow on the AFFINE cut (affine + flow) ==="
python -u -m unwrapping.eval.stabilize_flow \
    --in "$ODIR/FLAGSHIP_borderlock_affine.mp4" $GTARGS \
    --iters 2 --extra-crop 0.02 \
    --out "$ODIR/FLAGSHIP_borderlock_affine_flow.mp4"

echo "=== [3/4] dense Farneback flow straight on the translation-locked cut ==="
# Flow subsumes affine in principle, so this is the control: if it matches the
# affine+flow chain, the affine rung is redundant here.
python -u -m unwrapping.eval.stabilize_flow \
    --in "$ODIR/FLAGSHIP_borderlock.mp4" $GTARGS \
    --iters 2 --extra-crop 0.02 \
    --out "$ODIR/FLAGSHIP_borderlock_flow.mp4"

echo "=== [4/4] arbiter: temporal-median sharpness, central 70%, common size ==="
# Same inner crop and common size for every entry. NOT film_safe.mp4 -- it is
# full-width while these are picture-cropped, and mixing the two crops is the
# documented way to get a meaningless ranking.
python -u -m unwrapping.eval.border_stillness \
    --videos "$ODIR/film_borderlock_m02.mp4" \
             "$ODIR/FLAGSHIP_borderlock.mp4" \
             "$ODIR/FLAGSHIP_borderlock_affine.mp4" \
             "$ODIR/FLAGSHIP_borderlock_dense.mp4" \
             "$ODIR/FLAGSHIP_borderlock_flow.mp4" \
             "$ODIR/FLAGSHIP_borderlock_affine_flow.mp4" \
             "$GT" \
    --labels borderlock flagship flagship_affine flagship_dense \
             flagship_flow flagship_affine_flow GT_perflock \
    --margins 0.15 0.15 0.15 0.15 0.15 0.15 0.15 \
    --size 720 720 --out-dir "$ODIR"

ls -la "$ODIR"
echo "=== nsflow DONE ==="
