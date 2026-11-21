#!/bin/bash
# Per-frame GT alignment of the RESTITCHED + BAND-CORRECTED strip (order-fix walk),
# the old tw3 chain verbatim (doc s12.1a: --no-fliplr, z-crop 0.203,0.80, pitch 1130)
# plus LOSSLESS per-frame PNG folders:
#   frame_match --png-dir  -> png/gt/, png/ours_locked/ (GT-locked cut, translation
#                             only = faithful), pairs/ (GT|ours|overlay|diff)
#   stabilize_affine --png-dir -> png/ours_affine/, png/pairs_affine/,
#                             png/affine_params.csv (per-frame transform)
#   stabilize_affine (mp4, old recipe) -> film_affine.mp4
#   sbatch Scripts/slurm/slurm_restitch_framematch.sh
#SBATCH --job-name=rsfmatch
#SBATCH --partition=daily
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/rsfmatch_%j.out
set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
R="$PROJ/unwrapping/inr/results/newscan"
STRIP=${STRIP:-$R/restitch/render/wholeroll_bandfix.npy}
O=${O:-$R/restitch/frame_match_bandfix}
GT="$PROJ/data/h265_1080p.mp4"
[ -f "$STRIP" ] && [ -f "$GT" ] || { echo "missing strip or GT"; exit 1; }
mkdir -p "$O"
# Stage the strip to node-local scratch: mmap page-faulting a 2.4 GB .npy on
# Lustre hung twice (cl_sync_io_wait at the same 1.34 GB) on two different nodes.
LOC=/scratch/$USER/rsfm_${SLURM_JOB_ID:-$$}; mkdir -p "$LOC"
trap 'rm -rf "$LOC"' EXIT
cp "$STRIP" "$LOC/strip.npy"; STRIP="$LOC/strip.npy"
echo "staged strip -> $STRIP ($(du -h "$STRIP" | cut -f1)) $(date)"

echo "=== [1/3] frame_match --no-fliplr + lossless PNGs $(date) ==="
python -u -m unwrapping.eval.frame_match --strip "$STRIP" --gt "$GT" \
    --z-crop 0.203,0.80 --pitch 1130 --export-video --montage --video --fps 25 \
    --no-fliplr --out-dir "$O" --png-dir "$O/png" 2>&1 | tail -70

echo "=== [2/3] stabilize_affine, old recipe on the gtsync mp4 $(date) ==="
python -u -m unwrapping.eval.stabilize_affine \
    --in "$O/film_stabilized_gtsync.mp4" --gt "$GT" --mode affine \
    --out "$O/film_affine.mp4"

echo "=== [3/3] stabilize_affine, LOSSLESS on the PNG frames $(date) ==="
python -u -m unwrapping.eval.stabilize_affine --png-dir "$O/png" \
    --in "$O/film_stabilized_gtsync.mp4" --gt "$GT" --mode affine \
    --out "$O/png/film_affine_from_png.mp4"

for s in gt ours_locked ours_affine pairs_affine; do echo "  $s: $(ls $O/png/$s | wc -l) frames"; done
du -sh "$O/png" "$O/pairs"
echo "=== rsfmatch DONE $(date) ==="
