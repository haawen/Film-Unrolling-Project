#!/bin/bash
# Synthetic benchmark - roll a known 2D video strip into a CT-realistic spiral
# (256 slices x 4096^2 px, 28 windings, outer emulsion, flat emulsion slab) and save
# the CT volume, exact segmentation and exact strip-coordinate ground truth; then
# render the ORACLE strip (CT sampled at the true mapping = the best any learned
# mapping can do on this volume; used by syn_02's evaluation).
#
#   sbatch Code/slurm/syn_01_generate.sh                                  # both presets
#   sbatch Code/slurm/syn_01_generate.sh hq_video_4k_outer_imperfect     # one preset
#
# hq_video_4k_outer: clean; hq_video_4k_outer_imperfect: noise 0.03, eccentricity 10,
# radial jitter 5. ~70 GB per preset (per-slice segmentation files dominate).
# Output: $WORK/synthetic/<preset>/ (HDF5 slices + volume, ground_truth.npz, oracle_strip.npz)
#
#SBATCH --job-name=syn_generate
#SBATCH --partition=daily
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --output=work/logs/%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/Code/slurm/_activate.sh"
for p in ${@:-hq_video_4k_outer hq_video_4k_outer_imperfect}; do
    echo "=== generating $p ==="
    python -u -m virtual_unrolling.synthetic.generate --preset "$p" \
        --video "$THESIS_ROOT/Data/5_synthetic_inputs/big_buck_bunny_1080p.mp4" \
        --output-dir "$WORK/synthetic/$p"
    python -u -m virtual_unrolling.learned_unwrapping.render_oracle_strip \
        --syn-dir "$WORK/synthetic/$p" --out "$WORK/synthetic/$p/oracle_strip.npz"
done
