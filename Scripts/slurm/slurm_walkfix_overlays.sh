#!/bin/bash
# Full-resolution overlays of the RAW WALKS for every walk-fix variant, at the
# same z, so the four can be compared like for like.
#
# Deliberately RAW: inspect_walk_hires prefers matched_walks.npz when it is
# present, but that file holds post-resample / post-z-heal / post-arc-equalize
# geometry (its "points" are strip columns, not walk steps) -- exactly the
# confusion that made an earlier "the geometry is clean" verdict overstated. Each
# variant is therefore given a directory containing ONLY walk_anchors.npz, so the
# loader falls back to the raw walked paths.
#
# z picked across the failing range: 1640 (user-reported onset), 1770, 1900, 1965.
# Anchor index: these runs used --slice-frac 0.25 over chunks starting at 1590,
# so anchors are ~every 4th slice; the script maps z -> nearest anchor itself.
#
#   sbatch Scripts/slurm/slurm_walkfix_overlays.sh
#
#SBATCH --job-name=wfoverlay
#SBATCH --partition=daily
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/wfoverlay_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
R="$PROJ/unwrapping/inr/results/newscan_fix"

for V in V0_baseline V1_conserv V2_strict V3_contonly; do
  SRC="$R/$V/walk_anchors.npz"
  [ -f "$SRC" ] || { echo "### $V: no cache, skipping"; continue; }
  RAW="$R/$V/raw_only"; OUT="$R/$V/hires"
  rm -rf "$RAW"; mkdir -p "$RAW" "$OUT"
  ln -s "$SRC" "$RAW/walk_anchors.npz"

  IDX=$(python - "$SRC" <<'PY'
import sys, numpy as np
z = np.asarray(np.load(sys.argv[1], allow_pickle=True)["z_anchor"], float)
print(",".join(str(int(np.argmin(np.abs(z - t)))) for t in (1640, 1770, 1900, 1965)))
PY
)
  echo "############### $V -> anchors $IDX ###############"
  python -u -m unwrapping.inr.inspect_walk_hires \
      --cache-dir "$RAW" --ct-dir 01_Mickey_sprockets --full-only \
      --anchors "$IDX" --out-dir "$OUT" || echo "  overlay failed for $V"
  ls "$OUT"
done
echo "=== DONE ==="
find "$R" -name "*_full.png" | sort
