#!/bin/bash
# SEAM, take 3 — a step detector on the pooled OUTER-BOUNDARY profile r_out(phi).
#
# Why a new detector: the gap-count one (_detect_seam_angle) is unusable on the
# new scan (-59.5..+136 deg on consecutive slices with identical centres) and its
# pooled version gives 133.5 deg on a dip of 30.12 vs a 30.1-36.2 baseline = not
# separated from noise. Scripts/diag_seam_outer.py instead reads the film's OUTER
# END, which drops r_out by a whole winding pitch (~26 px new, ~21 px old) at one
# azimuth on an otherwise smooth curve.
#
# VALIDATION FIRST: it runs on the OLD scan, where the seam is independently
# known to be 60 deg. If it does not recover 60 there, ignore what it says about
# the new scan.
#
# Run 1 (job 7846304) failed on a hand-rolled Otsu that dropped the N factor from
# the between-class variance -> threshold ~30000, film 0.1%, everything downstream
# noise. Now using the same otsu() as Scripts/diag_zip_zsurvey.py, plus a printed
# film-fraction sanity flag, plus the centre taken from the walk cache instead of
# the mask centroid (the centroid is dragged around by the roll's eccentricity).
#
#   sbatch Scripts/slurm/slurm_seam_outer.sh
#
#SBATCH --job-name=seam_outer
#SBATCH --partition=hourly
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/seam_outer_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
OUT="$PROJ/unwrapping/inr/results/seam_outer"
mkdir -p "$OUT"

echo "################ CONTROL: OLD SCAN (true seam = 60 deg) ################"
python -u Scripts/diag_seam_outer.py \
    --ct-dir 01_Mickey_hdf \
    --n-slices 15 --pitch-px 21 --expect-deg 60 \
    --center-from-cache "$PROJ/unwrapping/inr/results/walk_full_final" \
    --save-npz "$OUT/oldscan_profiles.npz"

echo
echo "################ NEW SCAN, picture band (seam UNKNOWN) ################"
# z 482-538 only: the picture band starts ~470 (below that is the low perf row,
# where the outermost boundary is a perforation edge, not a film end).
python -u Scripts/diag_seam_outer.py \
    --ct-dir 01_Mickey_sprockets \
    --z 482,490,498,506,514,522,530,538 --pitch-px 26 \
    --center-from-cache "$PROJ/unwrapping/inr/results/newscan_walk_eps03" \
    --save-npz "$OUT/newscan_profiles.npz"

ls -la "$OUT"
