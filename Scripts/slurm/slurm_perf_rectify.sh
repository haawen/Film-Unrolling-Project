#!/bin/bash
# PERF-LOCKED RECTIFICATION of the full-width new-scan strip.
#
# GT-FREE stabilisation: the two perforation rows and the frame lines define
# where every frame really is, so each output frame is resampled from the strip
# ONCE, already rectified. This is the first stabilisation in this project that
# is a measurement rather than a registration to the GT scan.
#
# Two variants are rendered so the fiducial choice is settled by looking rather
# than by argument:
#   along=frameline  the along-film rails and the shear come from the frame
#                    lines, which sit in the WALKED picture band
#   along=perf       they come from the perforations, which sit in the
#                    z-EXTRAPOLATED bands
# Measured per-frame scatter says frameline should win (7-18 px vs 14 px low /
# 44 px high row), but the render is what decides.
#
# Plus a no-op control (--smooth-frames 0 --along perf is NOT the control; the
# control is the existing film_newscan1_ph.mp4 from make_film_video).
#
#   sbatch Scripts/slurm/slurm_perf_rectify.sh
#
#SBATCH --job-name=perf_rect
#SBATCH --partition=hourly
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/perf_rect_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

R="$PROJ/unwrapping/inr/results"
STRIP="$R/newscan_render1/wholeroll.npy"
PERFS="$R/newscan_perfs/perfs.npz"

# --reverse (arc runs inner->outer) and --invert (the film is a NEGATIVE) carry
# over from the old scan. --rotate is NOT needed: perf_rectify already emits
# [across, along] and rotates so the picture is upright, and the crop is defined
# from the perf rows so the sideways-frame trap cannot happen.
for ALONG in frameline perf; do
  echo
  echo "=================== along=$ALONG ==================="
  python -u -m unwrapping.eval.perf_rectify "$STRIP" \
      --perfs "$PERFS" \
      --out-dir "$R/newscan_perflock_$ALONG" \
      --video "film_perflock_$ALONG.mp4" \
      --along "$ALONG" \
      --smooth-frames 1.0 \
      --reverse --invert --height 720
done

echo
ls -la "$R"/newscan_perflock_*
