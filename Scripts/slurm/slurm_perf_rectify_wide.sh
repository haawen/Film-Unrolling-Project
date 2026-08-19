#!/bin/bash
# Wide-crop perf-locked render, for the STILLNESS measurement.
#
# The GT fidelity metrics came out tied across the baseline and both perf-locked
# variants, with a 58-82 px residual registration shift -- i.e. the guessed crops
# are mismatched enough to swamp the effect being measured. So measure stillness
# directly instead of through GT.
#
# The measurement needs the picture APERTURE EDGE inside the frame: it is a
# physical boundary that lives in the WALKED picture band, whereas the
# correction comes from perforations in the EXTRAPOLATED bands, so its residual
# motion is an independent check that the perf-driven correction actually
# stabilised the picture. --crop-mm 13.5 leaves ~1.6 mm of margin on each side.
#
#   sbatch Scripts/slurm/slurm_perf_rectify_wide.sh
#
#SBATCH --job-name=perf_wide
#SBATCH --partition=hourly
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/perf_wide_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

R="$PROJ/unwrapping/inr/results"
STRIP="$R/newscan_render1/wholeroll.npy"
PERFS="$R/newscan_perfs/perfs.npz"

# The CONTROL: smooth-frames 0 and no correction is not available as a flag, so
# the control is the un-rectified baseline film_newscan1_ph.mp4, which already
# carries the full film width.
python -u -m unwrapping.eval.perf_rectify "$STRIP" \
    --perfs "$PERFS" \
    --out-dir "$R/newscan_perflock_wide" \
    --video "film_perflock_wide.mp4" \
    --along perf --smooth-frames 1.0 --crop-mm 13.5 \
    --save-frames \
    --reverse --invert --height 720

ls -la "$R/newscan_perflock_wide"
