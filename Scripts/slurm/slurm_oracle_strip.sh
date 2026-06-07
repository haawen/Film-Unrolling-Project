#!/bin/bash
# Render oracle strips for HQ outer presets (clean + imperfect).
# Output: oracle_strip.npz alongside ground_truth.npz in each preset dir.
#
#   sbatch Scripts/slurm/slurm_oracle_strip.sh

#SBATCH --cluster=gmerlin7
#SBATCH --job-name=oracle
#SBATCH --output=logs/oracle_%j.out
#SBATCH --error=logs/oracle_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --partition=a100-hourly
#SBATCH --gpus=1

set -euo pipefail
PROJECT_DIR="$HOME/M_thesis"
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "${PROJECT_DIR}"
mkdir -p logs

for PRESET in hq_video_4k_outer hq_video_4k_outer_imperfect; do
  SYN="${PROJECT_DIR}/unwrapping/synthetic/results/${PRESET}"
  OUT="${SYN}/oracle_strip.npz"
  echo "=== ${PRESET} → ${OUT} ==="
  srun python -u -m unwrapping.inr.render_oracle_strip --syn-dir "${SYN}" --out "${OUT}"
done

echo "Done at $(date)"
