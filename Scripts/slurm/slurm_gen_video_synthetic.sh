#!/bin/bash
# =============================================================================
# Generate the missing video_4k synthetic presets (video_4k clean + imperfect)
# so we can train + evaluate on them and produce comparison images.
# =============================================================================
#SBATCH --job-name=gen_video
#SBATCH --partition=a100-hourly
#SBATCH --cluster=gmerlin7
#SBATCH --account=merlin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:55:00
#SBATCH --output=logs/gen_video_%j.out
#SBATCH --error=logs/gen_video_%j.err

set -euo pipefail

PROJECT_DIR="/data/user/li_k1/M_thesis"
CONDA_ENV="nnunet"
VIDEO="${PROJECT_DIR}/unwrapping/synthetic/sample_movie.mp4"

if [ -f "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh" ]; then
  source "/opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh"
fi
conda activate "${CONDA_ENV}"

cd "${PROJECT_DIR}"

# clean video_4k
echo ""
echo "=== generating video_4k (clean) ==="
srun -n1 --exclusive python -u -m unwrapping.synthetic.generate \
    --preset video_4k --video "${VIDEO}"

# imperfect video_4k
echo ""
echo "=== generating video_4k_imperfect (ecc=10, jit=5) ==="
srun -n1 --exclusive python -u -m unwrapping.synthetic.generate \
    --preset video_4k_imperfect --video "${VIDEO}"

echo "  done: $(date)"
