#!/bin/bash
# DENSE run, final stage: merge the five batch caches (1500 anchors, 100% of the
# picture band) and render the WHOLE stack off them.
#
#   sbatch Scripts/slurm/slurm_newscan_dense_render.sh
#
# The render covers all 2361 slices even though only z470-1969 was walked: v13
# fits interp1d per winding over z_anchor with fill_value=(X[0], X[-1]), so the
# perf rows and margins reuse the nearest picture-band geometry. That is exactly
# how the sprockets came out clean in the 160-anchor render, and it is why the
# failed perf-row walks were never needed.
#
# --min-coverage 400 of 1500 anchors (~27%): real windings sit near 1500, and a
# track that only exists in one batch tops out at ~300. Same logic that dropped
# the z780 bore phantom in the sparse render.
#
#SBATCH --job-name=nsdense_render
#SBATCH --partition=daily
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/nsdense_render_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

R="$PROJ/unwrapping/inr/results"
OUT="$R/newscan_dense/full"
mkdir -p "$OUT"

echo "=== merge the five batch caches ==="
python -u Scripts/merge_walk_caches.py \
    --out "$OUT/walk_anchors.npz" \
    --caches "$R/newscan_dense/b0/walk_anchors.npz" \
             "$R/newscan_dense/b1/walk_anchors.npz" \
             "$R/newscan_dense/b2/walk_anchors.npz" \
             "$R/newscan_dense/b3/walk_anchors.npz" \
             "$R/newscan_dense/b4/walk_anchors.npz"

echo
echo "=== render all 2361 slices ==="
# --seg-dir is REQUIRED by argparse even though --use-cached-anchors means it is
# never read. --seam-exclude-deg 0.3 is LOAD-BEARING (argparse default is 6.0).
python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 \
    --ct-dir 01_Mickey_sprockets \
    --seg-dir "$PROJ/newscan_z10/pairs" \
    --out-dir "$OUT" \
    --use-cached-anchors \
    --seam-deg 353.0 \
    --ref-mode track \
    --min-sep 7.5 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
    --z-heal-px 7.5 --min-coverage 400 \
    --transverse-px 3 \
    --seam-exclude-deg 0.3 --seam-bridge

echo
echo "=== video (new-scan flags: --rotate AND --invert-phase are both required) ==="
python -u -m unwrapping.inr.make_film_video \
    "$OUT/wholeroll.npy" "$OUT/film_dense.mp4" \
    --phase-lock global --invert-phase --reverse --invert --rotate --height 720

ls -la "$OUT"
