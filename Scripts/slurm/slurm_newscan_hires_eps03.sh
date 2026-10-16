#!/bin/bash
# Full-resolution FIT CHECK overlays for the eps=0.3 new-scan walk (the canonical
# --seam-exclude-deg; walk test 4 wrongly used the 6.0 default).
#
# The walk's own --inspect-anchors only leaves a coarse tiled montage (it deletes
# the per-anchor PNGs), which is too small to judge whether a line sits on the
# emulsion. This draws each anchor at full dpi plus zoom wedges.
#
# Wedges include 135 deg ON PURPOSE: that is the forced seam (133.5), where the
# montage showed a radial splay on 2 of 6 anchors -- the thing to judge.
#
#   sbatch Scripts/slurm/slurm_newscan_hires_eps03.sh
#
#SBATCH --job-name=hires_eps03
#SBATCH --partition=hourly
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/hires_eps03_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"

python -u -m unwrapping.inr.inspect_walk_hires \
    --cache-dir "$PROJ/unwrapping/inr/results/newscan_walk_eps03" \
    --ct-dir 01_Mickey_sprockets \
    --anchors 0,3,10,19 \
    --wedge-az-deg 135,45,225,315 \
    --out-dir "$PROJ/unwrapping/inr/results/newscan_walk_eps03/hires"

ls -la "$PROJ/unwrapping/inr/results/newscan_walk_eps03/hires"
