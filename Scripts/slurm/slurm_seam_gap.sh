#!/bin/bash
# How wide is the seam gap really, and where do the walks actually stop?
# Runs Scripts/diag_seam_gap.py on the NEW-scan walk test and on the delivered
# OLD-scan 100% walk as a control (its seam is independently known = 60 deg).
#
#   sbatch Scripts/slurm/slurm_seam_gap.sh
#
#SBATCH --job-name=seam_gap
#SBATCH --partition=hourly
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=180G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/seam_gap_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"

echo "############ NEW SCAN (newscan_walk_test, seam forced 133.5) ############"
python -u Scripts/diag_seam_gap.py \
    --npz "$PROJ/unwrapping/inr/results/newscan_walk_test/matched_walks.npz" \
    --anchors 8

echo
echo "############ OLD SCAN CONTROL (walk_full_final, seam 60) ############"
python -u Scripts/diag_seam_gap.py \
    --npz "$PROJ/unwrapping/inr/results/walk_full_final/matched_walks.npz" \
    --anchors 8
