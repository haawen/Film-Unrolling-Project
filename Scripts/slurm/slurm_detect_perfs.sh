#!/bin/bash
# PERFORATION DETECTION on the first full-width new-scan render.
#
# The whole point of the Mickey_merged_rigid_nobin reconstruction is that it
# keeps BOTH perf rows of the 16 mm double-perf stock. Those holes are ISO
# standard (1.83 mm across x 1.27 mm along, pitch 7.62 mm = one frame), so they
# are a metric ruler AND a GT-free per-frame fiducial: the first thing in this
# project that can measure the unrolling's residual geometric error without
# leaning on the GT scan.
#
# This is the measurement step only -- it writes perfs.npz plus two diagnostics
# (perf_diag.png = sway / pitch / hole-size traces, perf_crops.png = 12 holes
# spread over the reel so the fit can be eyeballed). Rectification is a separate
# job, gated on looking at these.
#
#   sbatch Scripts/slurm/slurm_detect_perfs.sh
#
#SBATCH --job-name=detect_perfs
#SBATCH --partition=hourly
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/detect_perfs_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

R="$PROJ/unwrapping/inr/results"
OUT="$R/newscan_perfs"
mkdir -p "$OUT"

# Pitch is left to auto-detect: 1130.06 was found independently by the render and
# recovers the old scan's 905 x 1.245 sampling ratio, so it is a free check that
# the detector is reading the same strip geometry.
python -u -m unwrapping.eval.detect_perfs \
    "$R/newscan_render1/wholeroll.npy" \
    --out-dir "$OUT"

echo
echo "=== outputs ==="
ls -la "$OUT"
