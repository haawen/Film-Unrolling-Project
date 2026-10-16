#!/bin/bash
# STAGE 0 - measure the two scan-specific parameters every walk/render depends on.
# CT only (no segmentation, no GPU). Re-derives the values hard-coded in the job
# scripts; it does not have to be run to reproduce the results.
#
#   sbatch Code/slurm/param_01_seam_and_picture_band.sh
#
# 1. SEAM AZIMUTH (-> --seam-deg 353 in every walk/render).
#    seam_from_outer_boundary: the film's outer end terminates at the seam, so the
#    outermost film radius r_out(phi) drops by one winding pitch (~26 px) at a single
#    azimuth on an otherwise smooth curve; the same is reported for the inner end.
#    Measured on picture-band slices z482-538 (below ~z470 the outer boundary is a
#    perforation edge, not the film end). The centre only adds a smooth cos(phi)
#    term, so the median spool centre of the shipped b0 walk is sufficient.
#    Cross-check from any walk: seam_from_walk_termini --npz <walk>/matched_walks.npz
#    (the outermost winding terminates at the seam).
#    The legacy automatic detector (common/seam.py) is unreliable on this scan.
# 2. PICTURE-BAND Z LIMITS (-> which slices are segmented/walked, z-crop in frame_match).
#    picture_band_limits: fraction of the film annulus covered by film and the number
#    of air gaps per radius ring (perforation holes) per slice. Full perforation
#    z<=460 and z>=1935; clean picture band z600-1930; walked range z470-1949.
# Output: $WORK/scan_parameters/{seam_profiles.npz, *.log}
#
#SBATCH --job-name=scan_params
#SBATCH --partition=daily
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=work/logs/%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/Code/slurm/_activate.sh"
OUT="$WORK/scan_parameters"; mkdir -p "$OUT"
P="python -u -m virtual_unrolling.scan_parameters"

CXY=$(python -c "import numpy as np; d=np.load('$D_SCAN/walk_caches/b0.npz', allow_pickle=True); \
print(f'{np.median(d[\"cx_a\"]):.1f},{np.median(d[\"cy_a\"]):.1f}')")
echo "=== 1. seam from the outer/inner film boundary (centre $CXY) ==="
$P.seam_from_outer_boundary --ct-dir "$D_SCAN/ct_slices" \
    --z 482,490,498,506,514,522,530,538 --pitch-px 26 --center-xy "$CXY" \
    --save-npz "$OUT/seam_profiles.npz" | tee "$OUT/seam.log"

echo "=== 2. picture-band limits: coarse survey, then both transitions ==="
$P.picture_band_limits --ct-dir "$D_SCAN/ct_slices" --z 100 2350 --step 50 | tee "$OUT/band_coarse.log"
$P.picture_band_limits --ct-dir "$D_SCAN/ct_slices" --z 420 700 --step 5   | tee "$OUT/band_low.log"
$P.picture_band_limits --ct-dir "$D_SCAN/ct_slices" --z 1850 2050 --step 5 | tee "$OUT/band_high.log"
echo "=== done: $OUT ==="
