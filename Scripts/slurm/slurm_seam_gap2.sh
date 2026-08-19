#!/bin/bash
# SEAM PROBE via collisions. With --seam-exclude-deg 0.3 the walk is forced THROUGH
# the physical seam, so a wrong seam angle makes adjacent windings converge/swap at
# the REAL seam azimuth. Runs on the new-scan eps=0.3 walk and on the delivered
# old-scan walk (true seam 60 deg) as the control.
#
#   sbatch Scripts/slurm/slurm_seam_gap2.sh
#
#SBATCH --job-name=seam_gap2
#SBATCH --partition=hourly
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/seam_gap2_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"

echo "######## NEW SCAN eps=0.3 (seam forced 133.5, UNVERIFIED) ########"
python -u Scripts/diag_seam_gap.py \
    --npz "$PROJ/unwrapping/inr/results/newscan_walk_eps03/matched_walks.npz" \
    --eps-deg 0.3 --anchors 8 --min-sep 8

echo
echo "######## NEW SCAN eps=6.0, same walk otherwise (for the A/B) ########"
python -u Scripts/diag_seam_gap.py \
    --npz "$PROJ/unwrapping/inr/results/newscan_walk_test/matched_walks.npz" \
    --eps-deg 6.0 --anchors 8 --min-sep 8

echo
echo "######## OLD SCAN CONTROL walk_full_final (seam 60, eps 6.0) ########"
python -u Scripts/diag_seam_gap.py \
    --npz "$PROJ/unwrapping/inr/results/walk_full_final/matched_walks.npz" \
    --eps-deg 6.0 --anchors 8 --min-sep 7
