#!/bin/bash
# Re-render tw2 and tw3 (film_safe's transverse-px=3) so their strips survive
# long enough to run THE validated stabilisation chain from
# slurm_newscan_framematch.sh (frame_match --no-fliplr --z-crop 0.203,0.80
# --pitch 1130 -> stabilize_affine --mode affine) on all three thickness
# variants (tw2, tw3, tw4). tw4's strip was already safeguarded to
# render_saved/wholeroll_tw4.npy before this job started (nothing else may
# touch results/newscan/render/ until this job finishes b/c it is a shared
# single-strip directory -- see slurm_newscan_render.sh's own header note).
#
# --no-fliplr is NOT the frame_match default -- it won the mirror test for
# this scan (grad-corr 0.602 vs 0.106 mirrored, job nsfmatch_8251015) and is
# hardcoded here rather than testing both again.
#
#SBATCH --job-name=nstwsweep
#SBATCH --partition=daily
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=360G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/nstwsweep_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export PYTHONPATH="$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

R="$PROJ/unwrapping/inr/results/newscan"
W="$R/walks"; OUT="$R/render"; SAVED="$R/render_saved"
GT="$PROJ/data/h265_1080p.mp4"
mkdir -p "$SAVED"

render_one () {
    local TAG=$1 TPX=$2
    echo "=== rendering $TAG (transverse-px=$TPX) ==="
    B1_FROM=$(python - "$W/b0/walk_anchors.npz" <<'PY'
import sys, numpy as np
print(int(np.load(sys.argv[1], allow_pickle=True)["z_anchor"].max()) + 1)
PY
)
    python -u Scripts/merge_walk_caches.py \
        --out "$OUT/walk_anchors.npz" --refit-centers \
        --caches "$W/b0/walk_anchors.npz" \
                 "$W/b1/walk_anchors.npz" \
                 "$W/b2/walk_anchors.npz" \
                 "$W/b3/walk_anchors.npz" \
                 "$W/tail_dense/walk_anchors.npz" \
        --z-range - "${B1_FROM}:" - :1589 1590:1949

    python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 \
        --ct-dir 01_Mickey_sprockets --seg-dir "$PROJ/newscan_low/pairs" \
        --out-dir "$OUT" --use-cached-anchors \
        --seam-deg 353.0 --ref-mode track \
        --min-sep 13 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
        --z-heal-px 7.5 --min-coverage 300 --no-gap-infill \
        --transverse-px "$TPX" --seam-exclude-deg 0.3 --seam-bridge

    cp "$OUT/wholeroll.npy" "$SAVED/wholeroll_${TAG}.npy"
    cp "$OUT/z_index.npy" "$SAVED/z_index_${TAG}.npy"
    echo "=== $TAG strip saved -> $SAVED/wholeroll_${TAG}.npy ==="
}

render_one tw2 2
render_one tw3 3

echo "=== all strips ready: $(ls -la "$SAVED") ==="

stabilize_one () {
    local TAG=$1
    local STRIP="$SAVED/wholeroll_${TAG}.npy"
    local D="$R/frame_match_${TAG}"
    echo "=== [$TAG] frame_match --no-fliplr ==="
    python -u -m unwrapping.eval.frame_match \
        --strip "$STRIP" --gt "$GT" --z-crop 0.203,0.80 --pitch 1130 \
        --no-fliplr --export-video --montage --video --fps 25 \
        --out-dir "$D" 2>&1 | tail -60
    echo "=== [$TAG] stabilize_affine ==="
    python -u -m unwrapping.eval.stabilize_affine \
        --in "$D/film_stabilized_gtsync.mp4" --gt "$GT" --mode affine \
        --out "$D/film_affine.mp4"
}

# tw4 uses the strip saved BEFORE this job started
stabilize_one tw4
stabilize_one tw2
stabilize_one tw3

ls -la "$R/frame_match_tw2" "$R/frame_match_tw3" "$R/frame_match_tw4"
echo "=== nstwsweep DONE ==="
