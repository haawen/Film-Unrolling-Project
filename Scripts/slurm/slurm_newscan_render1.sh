#!/bin/bash
# FIRST RENDER of the new scan — whole stack, from the walks we already have.
#
# Merge the EIGHT clean picture-band caches (z500 from newscan_walk_seam353 plus
# z560/780/1000/1220/1440/1660/1880 from newscan_z10) into one anchor cache and
# render every slice of 01_Mickey_sprockets (2361, z40-2400) off it. No new walks.
#
# The three chunks that failed -- z250 (low perf), z2060 (high perf), z2200
# (margin) -- are simply LEFT OUT. v13's renderer fits interp1d per winding with
# fill_value=(X[0], X[-1]), so z<500 reuses the z500 geometry and z>1899 reuses
# z1899: "the nearest working z", which is what was asked for. The perf rows will
# therefore be sampled along picture-band paths -- expected to be wrong in detail
# for the outer windings, but enough of the sprocket structure should survive for
# image registration, which is the point of rendering them at all.
#
# --min-coverage 40: real windings appear in all ~160 anchors; the z780 bore
# phantom (a circular track at r~300 on the CT's ring artefacts) exists in that
# one chunk only, so it lands at ~20 and gets dropped. That is the merge fixing
# the phantom for free.
#
# Safe to merge across the ~220-slice gaps: the outermost winding sits at
# r 1700/1698/1699/1696/1693/1690/1691 across z560..z1880 = ~2-3 px of drift per
# gap, against a 26 px pitch and --match-tol-px 15.
#
#   sbatch Scripts/slurm/slurm_newscan_render1.sh
#
#SBATCH --job-name=newscan_render1
#SBATCH --partition=daily
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/newscan_render1_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

R="$PROJ/unwrapping/inr/results"
OUT="$R/newscan_render1"
mkdir -p "$OUT"

echo "=== [1/3] merge the clean picture-band caches ==="
python -u Scripts/merge_walk_caches.py \
    --out "$OUT/walk_anchors.npz" \
    --caches "$R/newscan_walk_seam353/walk_anchors.npz" \
             "$R/newscan_z10/z560/walk_anchors.npz" \
             "$R/newscan_z10/z780/walk_anchors.npz" \
             "$R/newscan_z10/z1000/walk_anchors.npz" \
             "$R/newscan_z10/z1220/walk_anchors.npz" \
             "$R/newscan_z10/z1440/walk_anchors.npz" \
             "$R/newscan_z10/z1660/walk_anchors.npz" \
             "$R/newscan_z10/z1880/walk_anchors.npz"

echo
echo "=== [2/3] render every slice of the stack ==="
# Thresholds = the x1.2508-scaled v13 set used for every new-scan walk.
# --seam-exclude-deg 0.3 is LOAD-BEARING and is NOT the argparse default (6.0).
# --seg-dir is REQUIRED by argparse even though --use-cached-anchors means it is
# never read (the loader takes the `else` branch only when the cache is absent).
# Point it at a real dir so the parser is satisfied.
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
    --seam-exclude-deg 0.3 --seam-bridge

echo
echo "=== [3/3] video ==="
# Film is a NEGATIVE -> --invert; arc runs inner->outer so play --reverse.
# Let the pitch auto-detect on this first look rather than forcing the old
# scan's 905 -- this reconstruction is 1.245x finer, so its pitch is its own.
python -u -m unwrapping.inr.make_film_video \
    "$OUT/wholeroll.npy" "$OUT/film_newscan1.mp4" \
    --phase-lock global --reverse --invert --height 720

ls -la "$OUT"
