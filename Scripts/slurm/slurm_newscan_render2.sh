#!/bin/bash
# RENDER 2 of the full-width new scan: identical to newscan_render1 EXCEPT that
# the geometry outside the walk anchors is linearly extrapolated instead of held.
#
# Why: render1's anchors only cover the picture band (z500-1899), so v13's
# `fill_value=(X[0], X[-1])` rendered BOTH perforation rows on picture-band
# geometry. The holes come out displaced by 4-44 px in a way uncorrelated with
# the picture band's own motion (r = +0.03 against the frame lines), which is
# precisely what stops the sprockets working as a registration fiducial --
# locking to them moves the picture by a wobble the picture does not have.
# Scripts/diag_z_extrap_cv.py already cross-validated the fix: 350 slices out,
# linear gives 2.0 px against hold's 3.6 px.
#
# SAFETY: --z-extrap only changes z OUTSIDE the anchor range. The picture band
# is bit-for-bit identical to render1, so this cannot degrade the picture; it can
# only change the perf rows and margins.
#
# Reuses render1's merged anchor cache -- no re-merge, no new walks.
#
#   sbatch Scripts/slurm/slurm_newscan_render2.sh
#
#SBATCH --job-name=newscan_render2
#SBATCH --partition=daily
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/newscan_render2_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

R="$PROJ/unwrapping/inr/results"
OUT="$R/newscan_render2"
mkdir -p "$OUT"
cp "$R/newscan_render1/walk_anchors.npz" "$OUT/walk_anchors.npz"

# Flags are render1's verbatim plus --z-extrap linear. --seam-exclude-deg 0.3 is
# LOAD-BEARING and is NOT the argparse default (6.0); --seg-dir is required by
# argparse even though --use-cached-anchors means it is never read.
python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 \
    --ct-dir 01_Mickey_sprockets \
    --seg-dir "$PROJ/newscan_z10/pairs" \
    --out-dir "$OUT" \
    --use-cached-anchors \
    --seam-deg 353.0 \
    --ref-mode track \
    --min-sep 7.5 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
    --z-heal-px 7.5 --min-coverage 40 \
    --transverse-px 3 \
    --seam-exclude-deg 0.3 --seam-bridge \
    --z-extrap linear --z-extrap-span 500

echo
echo "=== detect perforations on the new strip ==="
python -u -m unwrapping.eval.detect_perfs "$OUT/wholeroll.npy" \
    --out-dir "$R/newscan_perfs2"

ls -la "$OUT" "$R/newscan_perfs2"
