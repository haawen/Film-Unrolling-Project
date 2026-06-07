#!/bin/bash
# =============================================================================
# Generate the HQ outer-emulsion 256-slice synthetic presets with the upgraded
# generator pipeline (continuous seam jitter; flat emulsion slab).
#
# Outer emulsion is the standard going forward (closest to real archival reels
# and to real Mickey). BBB 1080p video content, 4096 px, 28 windings, n_z=256.
#
# Usage:
#   sbatch Scripts/slurm/slurm_gen_hq_outer.sh                 # all 4 presets
#   sbatch Scripts/slurm/slurm_gen_hq_outer.sh hq_video_4k_outer_realistic
# =============================================================================
#SBATCH --job-name=gen_hq_outer
#SBATCH --partition=a100-daily
#SBATCH --cluster=gmerlin7
#SBATCH --account=merlin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=23:59:00
#SBATCH --output=logs/gen_hq_outer_%j.out
#SBATCH --error=logs/gen_hq_outer_%j.err

set -euo pipefail

PROJECT_DIR="/data/user/li_k1/M_thesis"
CONDA_ENV="nnunet"
VIDEO="${PROJECT_DIR}/unwrapping/synthetic/big_buck_bunny_1080p.mp4"

if [ -f "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"

if [ "$#" -ge 1 ]; then
  PRESETS=("$@")
else
  PRESETS=(hq_video_4k_outer hq_video_4k_outer_imperfect \
           hq_video_4k_outer_realistic hq_video_4k_outer_harsh)
fi

for p in "${PRESETS[@]}"; do
  echo ""
  echo "=== generating ${p} ($(date)) ==="
  srun -n1 --exclusive python -u -m unwrapping.synthetic.generate \
      --preset "${p}" --video "${VIDEO}"
done

echo "  done: $(date)"
