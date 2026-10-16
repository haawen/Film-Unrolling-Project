#!/bin/bash
# VALIDATE the winding-terminus seam probe on the OLD scan (seam known = 60 deg),
# then read it on the new scan.
#
# Context: the r_out step detector (job 7848427) found a +22.2 px step (~1 winding
# pitch) at az 353.0 deg SNR 5.2x on the new scan -- the textbook film-end
# signature -- but its OLD-scan control showed no step at all (the old roll's outer
# envelope is a smooth once-per-turn ellipse, step noise 2 px), so on its own it is
# unvalidated. This second, independent probe reads the azimuth where the
# OUTERMOST walked winding stops. On the new scan that gave abs 355.1 deg,
# agreeing with 353.0. If it also recovers ~60 on the old scan, the pair is
# trustworthy and the new scan's seam is ~353-355, NOT the 133.5 currently forced.
#
#   sbatch Scripts/slurm/slurm_seam_terminus.sh
#
#SBATCH --job-name=seam_term
#SBATCH --partition=hourly
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/seam_term_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"

echo "######## CONTROL: OLD SCAN walk_full_final (true seam = 60 deg) ########"
python -u Scripts/diag_seam_terminus.py \
    --npz "$PROJ/unwrapping/inr/results/walk_full_final/matched_walks.npz" \
    --anchors 20 --expect-deg 60

echo
echo "######## CONTROL 2: OLD SCAN walk_dense_v13 (same roll, 500 anchors) ########"
python -u Scripts/diag_seam_terminus.py \
    --npz "$PROJ/unwrapping/inr/results/walk_dense_v13/matched_walks.npz" \
    --anchors 20 --expect-deg 60

echo
echo "######## NEW SCAN newscan_walk_eps03 (seam forced 133.5) ########"
python -u Scripts/diag_seam_terminus.py \
    --npz "$PROJ/unwrapping/inr/results/newscan_walk_eps03/matched_walks.npz" \
    --anchors 20
