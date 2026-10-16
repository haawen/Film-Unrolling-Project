#!/bin/bash
# Roll2 rough video v2: correct rotation (--rot-k 3) + dump individual frames.
# Reuses the already-bridged 8192-arc strip from the previous job.
#SBATCH --job-name=roll2_vid2
#SBATCH --output=logs/roll2_vid2_%j.out
#SBATCH --error=logs/roll2_vid2_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --partition=hourly
#SBATCH --time=01:00:00

set -euo pipefail
PROJECT_DIR="$HOME/M_thesis"
RES="${PROJECT_DIR}/unwrapping/inr/results/roll2_rough"
STRIP="${RES}/roll2_strip_bridged.npy"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "${PROJECT_DIR}"
mkdir -p logs "${RES}/video" "${RES}/frames"

echo "=== [1/3] video (global lock, reverse, rot-k 3)  $(date) ==="
python -u -m unwrapping.inr.make_film_video "${STRIP}" \
    "${RES}/video/roll2_rough_v2.mp4" 10 \
    --phase-lock global --rotate --rot-k 3 --invert --reverse --height 720

echo "=== [2/3] video (track lock, rot-k 3) ==="
python -u -m unwrapping.inr.make_film_video "${STRIP}" \
    "${RES}/video/roll2_rough_v2_track.mp4" 10 \
    --phase-lock track --rotate --rot-k 3 --invert --reverse --height 720

echo "=== [3/3] dump individual frames (correct orientation) ==="
python -u - <<'PY'
import numpy as np
from PIL import Image
s = np.load("unwrapping/inr/results/roll2_rough/roll2_strip_bridged.npy", mmap_mode="r")
Z, W = s.shape
pitch = 1590.854
col = np.asarray(s[:, ::40]).mean(0)
# global best phase (same logic as make_film_video)
from numpy.fft import rfft, irfft, rfftfreq
d = col - np.convolve(np.pad(col, 1000, "edge"), np.ones(2001)/2001, "valid")[:len(col)]
F = rfft(d); fr = rfftfreq(len(d)); f0 = 40.0/pitch
F[np.abs(fr - f0) > 0.4*f0] = 0
p = irfft(F, n=len(d))
# just cut on a uniform grid from a decent offset; rough is rough
n = int(W/pitch)
bounds = np.round(np.arange(n+1)*pitch).astype(int)
lo, hi = np.percentile(np.asarray(s[:, ::50], np.float32), [1, 99])
# pick 10 frames spread over the middle 80% (skip washed ends), save full-res
picks = np.linspace(int(n*0.12), int(n*0.88), 10).astype(int)
for k, i in enumerate(picks):
    fr_ = np.asarray(s[:, bounds[i]:bounds[i+1]], np.float32)
    fr_ = np.clip((fr_-lo)/(hi-lo+1e-6), 0, 1)
    fr_ = 1.0 - fr_                       # CT negative
    fr_ = np.rot90(fr_, 3)               # correct orientation
    q2, q98 = np.percentile(fr_, [2, 98])
    fr_ = np.clip((fr_-q2)/(q98-q2+1e-6), 0, 1)
    Image.fromarray((fr_*255).astype(np.uint8)).save(
        f"unwrapping/inr/results/roll2_rough/frames/frame_{k:02d}_cell{i}.png")
    print(f"  frame_{k:02d}_cell{i}.png  {fr_.shape}")
PY

echo "=== done $(date) ==="
ls -la "${RES}/video/" "${RES}/frames/"
