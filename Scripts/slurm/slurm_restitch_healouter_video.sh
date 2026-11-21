#!/bin/bash
# Steps [3]+[4] of slurm_restitch_healouter.sh, re-run after the video step hung
# (job 11352598: make_film_video's np.load(mmap_mode="r") on the 2.4 GB Lustre
# strip stalled at frame 50/226 — same cl_sync_io_wait hang as rsfmatch). The
# strip is staged to node-local scratch first. Two videos: auto pitch (locked
# 1138.9 on this strip) and pitch 1130.13 = the previous restitch video's lock.
#   sbatch Scripts/slurm/slurm_restitch_healouter_video.sh
#SBATCH --job-name=rshealv
#SBATCH --partition=daily
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/rshealv_%j.out
set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
R="$PROJ/unwrapping/inr/results/newscan"
RS="$R/restitch_healouter"; OUT="$RS/render"
mkdir -p "$RS/videos"
LOC=/scratch/$USER/rshealv_${SLURM_JOB_ID:-$$}; mkdir -p "$LOC"
trap 'rm -rf "$LOC"' EXIT
cp "$OUT/wholeroll_bandfix.npy" "$LOC/strip.npy"
echo "staged strip -> $LOC/strip.npy ($(du -h "$LOC/strip.npy" | cut -f1)) $(date)"

echo "=== [3a] video, auto pitch $(date) ==="
python -u -m unwrapping.inr.make_film_video \
    "$LOC/strip.npy" "$RS/videos/film_restitch_healouter_bandfix.mp4" \
    --phase-lock global --invert-phase --reverse --invert --rotate --height 720
echo "=== [3b] video, pitch 1130.13 $(date) ==="
python -u -m unwrapping.inr.make_film_video \
    "$LOC/strip.npy" "$RS/videos/film_restitch_healouter_bandfix_p1130.mp4" \
    --pitch 1130.13 --no-refine-pitch \
    --phase-lock global --invert-phase --reverse --invert --rotate --height 720

echo "=== [4] frame_match + per-frame PNGs $(date) ==="
STRIP="$OUT/wholeroll_bandfix.npy" O="$RS/frame_match_bandfix" \
    bash Scripts/slurm/slurm_restitch_framematch.sh
echo "=== rshealv DONE $(date) ==="
