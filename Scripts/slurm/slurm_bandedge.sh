#!/bin/bash
# Two questions raised by the z470 overlay, answered before any re-segmentation.
#
# 1. WHERE DOES THE PICTURE BAND ACTUALLY START (AND END)?
#    CLAUDE.md says z470-1975 and slurm_newscan_dense_batch.sh used Z_START=470
#    on that basis. The z470 walk overlay is plainly a perforation slice: dashed
#    film bands, radial jogs, zigzag scribbles on the inner turns. So the recorded
#    edge is wrong and every chunk built from it is contaminated -- including the
#    batch's first find_spool_center sample, which is why b0's centre came back
#    cx 1812 against ~1904 for its neighbours.
#    Scanned in fine steps at both edges, plus a coarse sweep of the whole stack
#    for context.
#
# 2. IS THE SPOOL-CENTRE TRACK CLEAN, PER CACHE?
#    A centre error displaces a strip column by dx*sin(phi) - dy*cos(phi),
#    independent of radius, so it is confined in z and oscillates once per turn:
#    exactly the shape of an artifact that occupies a band of the video's
#    horizontal axis. Dumped for every cache that fed the v0fix render, plus the
#    merged v0fix cache itself (which had --refit-centers applied) so the repair
#    can be checked rather than assumed.
#
#   sbatch Scripts/slurm/slurm_bandedge.sh
#
#SBATCH --job-name=bandedge
#SBATCH --partition=hourly
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/bandedge_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"

echo "############ 1. picture-band edges from the CT ############"
echo "--- coarse sweep over the whole stack ---"
python -u Scripts/diag_band_edges.py --ct-dir 01_Mickey_sprockets --z 100 2350 --step 50
echo
echo "--- fine sweep, LOW edge ---"
python -u Scripts/diag_band_edges.py --ct-dir 01_Mickey_sprockets --z 420 700 --step 5
echo
echo "--- fine sweep, HIGH edge ---"
python -u Scripts/diag_band_edges.py --ct-dir 01_Mickey_sprockets --z 1850 2050 --step 5

echo
echo "############ 2. spool-centre tracks ############"
R="$PROJ/unwrapping/inr/results"
python -u Scripts/diag_centers.py --zbin 60 --caches \
    "$R/newscan_dense/b0/walk_anchors.npz" \
    "$R/newscan_dense/b1/walk_anchors.npz" \
    "$R/newscan_dense/b2/walk_anchors.npz" \
    "$R/newscan_dense/b3/walk_anchors.npz" \
    "$R/newscan_fix/V0_baseline/walk_anchors.npz" \
    "$R/newscan_dense/v0fix/walk_anchors.npz"
echo "=== DONE ==="
