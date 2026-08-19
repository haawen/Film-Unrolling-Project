#!/bin/bash
# Low band z520-779: SPARSE walk + overlays, reusing the segmentation that
# slurm_lowband.sh already produced (newscan_low/pairs, 13 chunks, ~47 GB).
#
# Split out because slurm_lowband.sh aborted AFTER the segmentation succeeded:
# its post-seg sanity check read the CT chunk with key "image", but chunks built
# by newscan_seg_test.py use "volume" (and the softmax uses "exported_data" --
# "image" is the convention of the single-slice raw files in 01_Mickey_sprockets,
# not of the chunks). Under `set -e` that killed the job. The check is kept here
# with the right keys because the thing it guards against is real: nnU-Net/NIfTI
# silently swaps this scan's near-equal H,W, and a transposed seg walks a
# mirrored mask -- which has already cost one full render.
#
# Pure CPU, so plain merlin7 -- no GPU needed once the softmax exists.
# Rationale for z520 and for --slice-frac 0.25: see slurm_lowband.sh.
#
#   sbatch Scripts/slurm/slurm_lowband_walk.sh
#
#SBATCH --job-name=lowwalk
#SBATCH --partition=daily
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/lowwalk_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

PAIRS="$PROJ/newscan_low/pairs"
OUT="$PROJ/unwrapping/inr/results/newscan_low/sparse"
[ -d "$PAIRS" ] || { echo "missing $PAIRS -- rerun slurm_lowband.sh"; exit 1; }
ls "$PAIRS"/*_Probabilities.h5 | wc -l
rm -rf "$OUT"; mkdir -p "$OUT"

echo "=== orientation check (seg must match its CT chunk) ==="
python - "$PAIRS" <<'PY'
import glob, os, sys, h5py
bad = 0
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*_Probabilities.h5"))):
    ct = f.replace("_Probabilities", "")
    with h5py.File(f) as a, h5py.File(ct) as b:
        s = a["exported_data"].shape          # (Z, H, W, C)
        c = b["volume"].shape                 # (Z, H, W)
        ok = s[:3] == c
        bad += not ok
        if not ok or f.endswith("0520-0539_Probabilities.h5"):
            print(f"  {os.path.basename(f)}: seg {s} ct {c} {'OK' if ok else 'MISMATCH'}")
print(f"  {bad} mismatched chunk(s)")
raise SystemExit(1 if bad else 0)
PY

echo "=== sparse walk z520-779 (slice-frac 0.25) ==="
python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 \
    --ct-dir 01_Mickey_sprockets --seg-dir "$PAIRS" --out-dir "$OUT" \
    --slice-frac 0.25 --n-centers 3 --seam-deg 353.0 \
    --smooth-px 0.75 --snap-max 10 --step 2.5 --coast-max 50 \
    --film-min-thick 7.5 --jump-max 5 --snap-accept 7.5 --max-dr 2.5 \
    --recenter-search 7.5 --recenter-sigma 25 \
    --min-sep 13 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
    --z-heal-px 7.5 --min-coverage 20 \
    --transverse-px 3 --max-infill 3 --radial-robust \
    --seam-exclude-deg 0.3 \
    --inspect-anchors --inspect-max 6

echo "=== walk quality vs z ==="
python -u Scripts/diag_walk_quality.py --npz "$OUT/matched_walks.npz" --zbin 10 || true
echo "=== centre track ==="
python -u Scripts/diag_centers.py --zbin 20 --caches "$OUT/walk_anchors.npz" || true

echo "=== full-slice overlays of the RAW walks ==="
RAW="$OUT/raw_only"; rm -rf "$RAW"; mkdir -p "$RAW"
ln -s "$OUT/walk_anchors.npz" "$RAW/walk_anchors.npz"
IDX=$(python - "$OUT/walk_anchors.npz" <<'PY'
import sys, numpy as np
z = np.asarray(np.load(sys.argv[1], allow_pickle=True)["z_anchor"], float)
print(",".join(str(int(np.argmin(np.abs(z - t))))
                for t in (525, 560, 600, 640, 680, 720, 760)))
PY
)
echo "overlay anchors: $IDX"
python -u -m unwrapping.inr.inspect_walk_hires \
    --cache-dir "$RAW" --ct-dir 01_Mickey_sprockets --full-only \
    --anchors "$IDX" --out-dir "$OUT/hires"
ls -la "$OUT" "$OUT/hires"
echo "=== lowwalk DONE ==="
