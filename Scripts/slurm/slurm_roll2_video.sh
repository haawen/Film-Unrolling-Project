#!/bin/bash
# Rough Roll2 film video from the winding-matched rough strip.
# 1. bridge the seam-wedge NaNs by per-row linear interpolation
# 2. render a couple of flag variants + frame montages for inspection
#SBATCH --job-name=roll2_vid
#SBATCH --output=logs/roll2_vid_%j.out
#SBATCH --error=logs/roll2_vid_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=hourly
#SBATCH --time=01:00:00

set -euo pipefail
PROJECT_DIR="$HOME/M_thesis"
RES="${PROJECT_DIR}/unwrapping/inr/results/roll2_rough"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "${PROJECT_DIR}"
mkdir -p logs "${RES}/video"

echo "=== [1/3] bridge seam-wedge NaNs  $(date) ==="
python -u - <<'PY'
import numpy as np
s = np.load("unwrapping/inr/results/roll2_rough/roll2_rough_strip.npy")
Z, W = s.shape
x = np.arange(W)
nn = np.isnan(s)
print(f"strip {s.shape}, NaN frac {nn.mean():.3f}")
for i in range(Z):
    m = nn[i]
    if m.any() and (~m).sum() > 10:
        s[i, m] = np.interp(x[m], x[~m], s[i, ~m])
s = np.nan_to_num(s, nan=float(np.nanmedian(s)))
np.save("unwrapping/inr/results/roll2_rough/roll2_strip_bridged.npy", s)
print("bridged ->", s.shape, "finite", np.isfinite(s).all())
PY

STRIP="${RES}/roll2_strip_bridged.npy"

echo "=== [2/3] auto-pitch render (rotate+invert+reverse, global lock) ==="
python -u -m unwrapping.inr.make_film_video "${STRIP}" \
    "${RES}/video/roll2_rough_global.mp4" 10 \
    --phase-lock global --rotate --invert --reverse --height 640

echo "=== [3/3] track-lock + no-reverse variant ==="
python -u -m unwrapping.inr.make_film_video "${STRIP}" \
    "${RES}/video/roll2_rough_track.mp4" 10 \
    --phase-lock track --rotate --invert --height 640

echo "=== done $(date) ==="
ls -la "${RES}/video/"
