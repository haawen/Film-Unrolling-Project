#!/bin/bash
# Low band z520-779: fix the two faults left in the sparse walk --
#   (1) the OUTERMOST WINDING is missed at some anchors,
#   (2) a few WINDING JUMPS.
#
# No re-segmentation: this reuses newscan_low/pairs, so it is CPU-only and each
# variant costs ~12 min. --n-centers 1 for the same reason -- the walk is
# centre-free after seeding, so centres do not affect a ranking run, and the 3
# centre calls were 26 of the previous run's 30 minutes.
#
# (1) MEASURED CAUSE. --radial-robust builds the seeding annulus from azimuthal
# coverage: a radius bin counts only if >--radial-cov (0.3) of 72 azimuth bins
# carry film. The outermost wrap is a PARTIAL turn -- the film's outer end stops
# part way round -- so it scores below that and the annulus is cut inside it;
# walk_anchor then gets rf.max()-5 and never reaches it. The z720 chunk printed
# "radial range: coverage>0.3 -> r 752-1720 (raw min/max 532-1728)" and the
# overlay shows an untraced arc outside the outermost coloured path.
# Lowering --radial-cov itself is NOT the fix: it exists to stop bore artefacts
# setting r0, where unbounded infill once turned 37 real windings into 71. So
# --radial-cov-outer relaxes the OUTER edge only, outward-contiguously, capped by
# --radial-outer-max-px (40px ~ 1.5 winding pitches).
#
# (2) The anti-jump levers are --max-dr (cap radius change per step) and
# --snap-accept (only snap to a confident target, else coast).
#
# VARIANTS -- one lever at a time so a win is attributable:
#   A  outer gate only            radial-cov-outer 0.12
#   B  A + tighter max-dr         + max-dr 1.5
#   C  B + conservative snapping  + snap-accept 5.0, coast-max 120
# The baseline is the existing walks/low_sparse run, already scored, so it is not
# repeated here.
#
#   sbatch Scripts/slurm/slurm_lowfix.sh
#
#SBATCH --job-name=lowfix
#SBATCH --partition=daily
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/lowfix_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

PAIRS="$PROJ/newscan_low/pairs"
W="$PROJ/unwrapping/inr/results/newscan/walks"
[ -d "$PAIRS" ] || { echo "missing $PAIRS"; exit 1; }

run () {                    # name  extra args...
  local NAME=$1; shift
  local OUT="$W/low_$NAME"
  rm -rf "$OUT"; mkdir -p "$OUT"
  echo
  echo "################## low_$NAME: $* ##################"
  python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 \
      --ct-dir 01_Mickey_sprockets --seg-dir "$PAIRS" --out-dir "$OUT" \
      --slice-frac 0.25 --n-centers 1 --seam-deg 353.0 \
      --smooth-px 0.75 --snap-max 10 --step 2.5 --coast-max 50 \
      --film-min-thick 7.5 --jump-max 5 --snap-accept 7.5 --max-dr 2.5 \
      --recenter-search 7.5 --recenter-sigma 25 \
      --min-sep 13 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
      --z-heal-px 7.5 --min-coverage 20 \
      --transverse-px 3 --max-infill 3 --radial-robust \
      --seam-exclude-deg 0.3 "$@" \
    || { echo "  low_$NAME FAILED"; return 0; }
  echo "--- low_$NAME quality ---"
  python -u Scripts/diag_walk_quality.py --npz "$OUT/matched_walks.npz" --zbin 20 || true
  echo "--- low_$NAME per-winding radius vs z (outermost present?) ---"
  python -u Scripts/diag_z_geometry.py --npz "$OUT/matched_walks.npz" --pitch 26 \
      --zbin 20 | tail -45 || true
}

# Later args win in argparse, so the overrides after "$@" expansion are fine.
run A --radial-cov-outer 0.12
run B --radial-cov-outer 0.12 --max-dr 1.5
run C --radial-cov-outer 0.12 --max-dr 1.5 --snap-accept 5.0 --coast-max 120

echo
echo "=== overlays for every variant at the same z ==="
for V in A B C; do
  OUT="$W/low_$V"
  [ -f "$OUT/walk_anchors.npz" ] || continue
  RAW="$OUT/raw_only"; rm -rf "$RAW"; mkdir -p "$RAW"
  ln -s "$OUT/walk_anchors.npz" "$RAW/walk_anchors.npz"
  IDX=$(python - "$OUT/walk_anchors.npz" <<'PY'
import sys, numpy as np
z = np.asarray(np.load(sys.argv[1], allow_pickle=True)["z_anchor"], float)
print(",".join(str(int(np.argmin(np.abs(z - t)))) for t in (525, 640, 720)))
PY
)
  python -u -m unwrapping.inr.inspect_walk_hires \
      --cache-dir "$RAW" --ct-dir 01_Mickey_sprockets --full-only \
      --anchors "$IDX" --out-dir "$OUT/hires" || echo "  overlays failed for $V"
done
echo
echo "=== summary: windings kept per variant ==="
grep -H "windings kept\|whole roll\|outer edge extended" \
    "$PROJ/logs/lowfix_${SLURM_JOB_ID}.out" || true
echo "=== lowfix DONE ==="
