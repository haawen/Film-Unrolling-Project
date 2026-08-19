#!/bin/bash
# Localise the SECOND artifact (video ~7 s, left ~15% of the picture area) and
# show the walks that region is built from, BEFORE spending a GPU day on a
# re-segmentation + dense re-walk there.
#
# Coordinate mapping used below (all of it verified against the 5 s artifact,
# which sat at cols 97500-103500 = cells 88-92 = frames 136-140):
#   video x  -> strip ROW: the frame is (1130 cols) x (2361 z) rot90'd, so the
#               horizontal axis is z. x=0 is z40, x=1504 is z2400.
#               The picture area sits at video x 295-1230 = z 503-1971.
#               "left 15% of the picture area" = video x 295-435 = z 503-723.
#   video t  -> strip COLUMN: 25 fps, 229 cells, --reverse, so cell = 228 - frame
#               and cols = cell * 1130.14. 7.0 s = frame 175 = cell 53
#               = cols 59.9k-61.0k.
#
# PART 1 cuts the strip at full resolution over frames 160-200 (cells 28-68), so
# the defect can be read in walk coordinates instead of in seconds -- both over
# the full z (is it really confined to low z?) and zoomed on z470-770.
#
# PART 2 draws the RAW walks that low-z band was rendered from. Those anchors are
# in newscan_dense/b0 (batch 0 = chunks z470-769, slice-frac 1.0, 300 anchors),
# so anchor index = z - 470. Six anchors spread over the band. Deliberately given
# a directory holding ONLY walk_anchors.npz, so inspect_walk_hires falls back to
# the RAW walked paths instead of preferring matched_walks.npz, whose "points"
# are post-resample strip columns rather than walk steps.
#
#   sbatch Scripts/slurm/slurm_art7s.sh
#
#SBATCH --job-name=art7s
#SBATCH --partition=hourly
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/art7s_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"

V="$PROJ/unwrapping/inr/results/newscan_dense/v0fix"
OUT="$PROJ/unwrapping/inr/results/artifact_7s"
mkdir -p "$OUT"

echo "=== PART 1: strip crops around the 7 s cells ==="
python -u Scripts/crop_strip_png.py --strip "$V/wholeroll.npy" --out-dir "$OUT" \
    --frames 160 200 --n-frames 229 --pitch 1130.14 --xscale 4 --tag fullz
python -u Scripts/crop_strip_png.py --strip "$V/wholeroll.npy" --out-dir "$OUT" \
    --frames 168 184 --n-frames 229 --pitch 1130.14 --z 400 780 --tag lowz
python -u Scripts/crop_strip_png.py --strip "$V/wholeroll.npy" --out-dir "$OUT" \
    --frames 168 184 --n-frames 229 --pitch 1130.14 --z 1100 1480 --tag midz

echo
echo "=== PART 1b: where does the strip step along z? ==="
python -u Scripts/diag_strip_artifact.py --a "$V/wholeroll.npy" --cblock 1130 --top 20

echo
echo "=== PART 2: raw walk overlays over z470-770 (b0 cache) ==="
B="$PROJ/unwrapping/inr/results/newscan_dense/b0"
RAW="$B/raw_only"; rm -rf "$RAW"; mkdir -p "$RAW"
ln -s "$B/walk_anchors.npz" "$RAW/walk_anchors.npz"
python -u -m unwrapping.inr.inspect_walk_hires \
    --cache-dir "$RAW" --ct-dir 01_Mickey_sprockets --full-only \
    --anchors 10,60,110,160,210,260 --out-dir "$OUT/walks_b0"
ls -la "$OUT" "$OUT/walks_b0"
echo "=== DONE ==="
