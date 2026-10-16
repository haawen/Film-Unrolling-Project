#!/bin/bash
# FULL-DENSITY low-band walk, z520-779, with the outer-gate fix.
#
# WHY DENSE: the quarter-density low_sparse cache renders flat bright blocks into
# z679-873 (measured: 3 frames at t 5.16-5.72s carry a near-flat bright region
# >55% of the column height; the b0-based render has 1, two pixels wide, at the
# frame edge). b0 is dense and clean there, but its find_spool_center samples were
# taken on perforation slices -- 46 anchors 94px off a robust line. This run gets
# both: dense anchors AND centres taken on picture material.
#
# PARAMETERS = sweep variant B (see slurm_lowfix.sh):
#   --radial-cov-outer 0.12  recovers the outermost wrap, which the strict 0.3
#                            coverage gate cuts because that wrap is a PARTIAL
#                            turn. Measured: 34 -> 35 windings KEPT (walked, not
#                            infilled) in both variants A and B.
#   --max-dr 1.5             the documented primary anti-jump lever, tightened
#                            from 2.5. B matched A's winding recovery, so this
#                            costs nothing on that axis.
#
# --inspect-anchors is passed deliberately: matched_walks.npz is only written
# under that flag, and without it the walk-quality and geometry diagnostics have
# nothing to read (that is why the sweep's quality tables came back empty).
#
# Segmentation reused from newscan_low/pairs -- CPU only, no GPU.
#
#   sbatch Scripts/slurm/slurm_lowdense.sh
#
#SBATCH --job-name=lowdense
#SBATCH --partition=daily
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/lowdense_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

PAIRS="$PROJ/newscan_low/pairs"
OUT="$PROJ/unwrapping/inr/results/newscan/walks/low_dense"
[ -d "$PAIRS" ] || { echo "missing $PAIRS"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT"

python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 \
    --ct-dir 01_Mickey_sprockets --seg-dir "$PAIRS" --out-dir "$OUT" \
    --slice-frac 1.0 --n-centers 6 --seam-deg 353.0 \
    --smooth-px 0.75 --snap-max 10 --step 2.5 --coast-max 50 \
    --film-min-thick 7.5 --jump-max 5 --snap-accept 7.5 \
    --recenter-search 7.5 --recenter-sigma 25 \
    --min-sep 13 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
    --z-heal-px 7.5 --min-coverage 20 \
    --transverse-px 3 --max-infill 3 --radial-robust \
    --seam-exclude-deg 0.3 \
    --radial-cov-outer 0.12 --max-dr 1.5 \
    --inspect-anchors --inspect-max 6

echo "=== walk quality vs z ==="
python -u Scripts/diag_walk_quality.py --npz "$OUT/matched_walks.npz" --zbin 40 || true
echo "=== centre track ==="
python -u Scripts/diag_centers.py --zbin 60 --caches "$OUT/walk_anchors.npz" || true

echo "=== full-slice overlays of the RAW walks ==="
RAW="$OUT/raw_only"; rm -rf "$RAW"; mkdir -p "$RAW"
ln -s "$OUT/walk_anchors.npz" "$RAW/walk_anchors.npz"
IDX=$(python - "$OUT/walk_anchors.npz" <<'PY'
import sys, numpy as np
z = np.asarray(np.load(sys.argv[1], allow_pickle=True)["z_anchor"], float)
print(",".join(str(int(np.argmin(np.abs(z - t))))
                for t in (525, 580, 640, 700, 740, 775)))
PY
)
python -u -m unwrapping.inr.inspect_walk_hires \
    --cache-dir "$RAW" --ct-dir 01_Mickey_sprockets --full-only \
    --anchors "$IDX" --out-dir "$OUT/hires"
ls -la "$OUT" "$OUT/hires"
echo "=== lowdense DONE ==="
