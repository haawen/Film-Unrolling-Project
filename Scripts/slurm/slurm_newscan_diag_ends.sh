#!/bin/bash
# Diagnose the thin vertical lines at the START of the video (= the strip's OUTER
# end): dump the 2 innermost + 2 outermost kept windings' geometry (pre-heal and
# final) with the exact order-fix/restitch render flags (cache-fast, no render),
# then correlate the strip's line rows with those windings' jumped/clamped bins.
#   sbatch Scripts/slurm/slurm_newscan_diag_ends.sh
#SBATCH --job-name=nsends
#SBATCH --partition=daily
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/nsends_%j.out
set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
R="$PROJ/unwrapping/inr/results/newscan"
D="${D:-$R/diag/end_windings}"; mkdir -p "$D"
ln -sfn "$R/render/walk_anchors.npz" "$D/walk_anchors.npz"
FLAGS="${FLAGS:-}"
echo "=== [1] dump end-winding geometry $(date) FLAGS=[$FLAGS] ==="
[ "${SKIP_DUMP:-0}" = 1 ] && [ -f "$D/geo.npz" ] && echo "  SKIP_DUMP: reusing $D/geo.npz" || \
python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 \
    --ct-dir 01_Mickey_sprockets --seg-dir "$PROJ/newscan_low/pairs" \
    --out-dir "$D" --use-cached-anchors \
    --seam-deg 353.0 --ref-mode track \
    --min-sep 13 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
    --z-heal-px 7.5 --min-coverage 300 --no-gap-infill \
    --transverse-px 3 --seam-exclude-deg 0.3 --seam-bridge \
    --enforce-radial-order $FLAGS --dump-geometry "$D/geo.npz" 2>&1 \
    | grep -v -E "per-anchor winding counts" | tee "$D/dump_cmd.log"
echo "=== [2] line vs geometry $(date) ==="
# stage the strip locally: mmap page-faulting it on Lustre hung (rsfmatch 2x)
LOC=/scratch/$USER/nsends_${SLURM_JOB_ID:-$$}; mkdir -p "$LOC"; trap 'rm -rf "$LOC"' EXIT
STRIP="${STRIP:-$R/restitch/render/wholeroll.npy}"
cp "$STRIP" "$LOC/strip.npy"
python -u Scripts/diag_end_windings.py --geo "$D/geo.npz" \
    --strip "$LOC/strip.npy" \
    --zidx "$R/restitch/render/z_index.npy" --out-dir "$D" \
    --ct-dir "$PROJ/01_Mickey_sprockets"
echo "=== nsends DONE $(date) ==="
