#!/bin/bash
# THE OLD (tw3) STABILISATION PIPELINE, RUN VERBATIM ON THE NEW FULL-WIDTH SCAN.
#
#   frame_match.py  --export-video   ->  film_stabilized_gtsync.mp4 (1:1 with GT)
#   stabilize_affine.py --mode affine ->  film_affine.mp4
#
# This is the chain that produced the extremely stable old-scan video
# (unwrapping/eval/results/frame_match_tw3/). The new-scan chain replaced
# frame_match with border_render + a whole-frame translation track, and its
# affine ENDS at roughly the deformation the old chain STARTED from
# (0.0095/0.0137/0.0091 vs the old input's 0.0082/0.0127/0.0127) -- i.e. one
# whole stage behind. frame_match is that stage: it searches, per GT frame, the
# exact strip position that frame sits at (coarse-to-fine, then a cell-quantized
# DP + per-frame GT lock), so every frame is CUT at its own correct place before
# any warp is fitted.
#
# WHAT THE NEW SCAN NEEDS -- FLAGS ONLY, NO CODE CHANGES:
#   --z-crop 0.203,0.80  the strip is 2361 rows of FULL film width; only
#                        z520-1930 (rows 480-1890 = fractions 0.203-0.800) is
#                        picture. The 0.05,0.95 default would feed perforation
#                        rows and margins into the matcher. Strip row = z - 40.
#   --pitch 1130         new-scan frame pitch (old scan: 905). Still refined to
#                        a fractional pitch internally.
# NO --rotate is needed: frame_match already does np.rot90 in window_small and
# window_full (that is make_film_video --rotate's orientation, which the new
# scan needs and the old scan's canonical command did NOT pass).
#
# GT IS THE RAW SCAN with the default 0.235,0.12,0.81,0.88 crop and leader trim
# -- exactly what the old chain used. NOT the perf-locked GT: that one is
# already cut and trimmed, and frame_match trims/crops unconditionally.
#
# FLIPLR IS THE ONE UNVERIFIED ASSUMPTION, so both are run and compared. The old
# scan needed the mirror ("the validated video orientation vs GT"); the new
# reconstruction may or may not. A wrong mirror shows up as a much lower mean
# match score, so the two runs settle it -- do not guess.
#
#   sbatch Scripts/slurm/slurm_newscan_framematch.sh
#
#SBATCH --job-name=nsfmatch
#SBATCH --partition=daily
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/nsfmatch_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export PYTHONPATH="$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

R="$PROJ/unwrapping/inr/results/newscan"
STRIP="$R/render/wholeroll.npy"
GT="$PROJ/data/h265_1080p.mp4"
[ -f "$STRIP" ] || { echo "missing strip"; exit 1; }
[ -f "$GT" ] || { echo "missing raw GT"; exit 1; }

COMMON="--strip $STRIP --gt $GT --z-crop 0.203,0.80 --pitch 1130 \
        --export-video --montage --video --fps 25"

echo "=== [1/3] frame_match, mirrored (old-scan convention) ==="
python -u -m unwrapping.eval.frame_match $COMMON \
    --out-dir "$R/frame_match" 2>&1 | tail -60

echo "=== [2/3] frame_match, NOT mirrored (control) ==="
python -u -m unwrapping.eval.frame_match $COMMON --no-fliplr \
    --out-dir "$R/frame_match_noflip" 2>&1 | tail -60

echo "=== mean match score decides the mirror ==="
for d in frame_match frame_match_noflip; do
    python -u - "$R/$d" <<'EOF'
import json, sys, os, glob
d = sys.argv[1]
for p in glob.glob(os.path.join(d, "*.json")):
    try:
        j = json.load(open(p))
    except Exception:
        continue
    if isinstance(j, dict):
        for k in ("grad_corr_mean", "score_mean", "mean_score"):
            if k in j:
                print(f"  {os.path.basename(d):20s} {k} = {j[k]:.4f}")
EOF
done

echo "=== [3/3] affine on the GT-synced cut (old recipe) ==="
for d in frame_match frame_match_noflip; do
    IN="$R/$d/film_stabilized_gtsync.mp4"
    [ -f "$IN" ] || { echo "  no gtsync video in $d"; continue; }
    python -u -m unwrapping.eval.stabilize_affine \
        --in "$IN" --gt "$GT" --mode affine \
        --out "$R/$d/film_affine.mp4"
done

ls -la "$R/frame_match" "$R/frame_match_noflip"
echo "=== nsfmatch DONE ==="
