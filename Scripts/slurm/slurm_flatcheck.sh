#!/bin/bash
# =============================================================================
# Flat-slab confirmation: re-eval existing checkpoints (trained on the OLD
# depth-profile data) against the REGENERATED flat-slab volumes. Geometry is
# deterministic/identical, so the learned mapping still applies; only the CT
# intensities changed. Isolates the rendering-floor improvement on a fixed
# mapping (no retraining). Writes to *_flatcheck out-dirs (no clobber).
#   sbatch Scripts/slurm/slurm_flatcheck.sh
# =============================================================================
#SBATCH --cluster=gmerlin7
#SBATCH --job-name=flatcheck
#SBATCH --output=logs/flatcheck_%j.out
#SBATCH --error=logs/flatcheck_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=160G
#SBATCH --partition=a100-daily

set -euo pipefail
PROJECT_DIR="$HOME/M_thesis"
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "${PROJECT_DIR}"
mkdir -p logs
R="${PROJECT_DIR}/unwrapping/inr/results"
S="${PROJECT_DIR}/unwrapping/synthetic/results"

# label  ckpt  preset
ROWS=(
  "G1_grid_imperfect|${R}/arch_smoke_hq_video_4k_outer_imperfect/G1_grid/train/model_final.pt|hq_video_4k_outer_imperfect"
  "R1S_clean|${R}/R1_hq_full_hq_video_4k_outer/R1S/train/model_final.pt|hq_video_4k_outer"
  "R1n_clean|${R}/R1_hq_full_hq_video_4k_outer/R1n/train/model_final.pt|hq_video_4k_outer"
  "R1S_imperfect|${R}/R1_hq_full_hq_video_4k_outer_imperfect/R1S/train/model_final.pt|hq_video_4k_outer_imperfect"
  "R1n_imperfect|${R}/R1_hq_full_hq_video_4k_outer_imperfect/R1n/train/model_final.pt|hq_video_4k_outer_imperfect"
)

for row in "${ROWS[@]}"; do
  IFS='|' read -r LABEL CKPT PRESET <<< "$row"
  SYN="${S}/${PRESET}"
  OUT="${R}/flatcheck/${LABEL}"
  echo ""
  echo "=== ${LABEL}  ($(date)) ==="
  echo "  ckpt: ${CKPT}"
  if [ ! -f "${CKPT}" ]; then echo "  MISSING ckpt, skip"; continue; fi
  mkdir -p "${OUT}"
  srun python -u -m unwrapping.inr.surface_eval \
      --data-dir "${SYN}" --out-dir "${OUT}" \
      --attachment emulsion --centerline-erode 1 \
      --ckpt "${CKPT}" \
      --gt-npz "${SYN}/ground_truth.npz" \
      --oracle-npz "${SYN}/oracle_strip.npz"
done
echo ""
echo "=== SUMMARY (new flat-slab data) ==="
python - <<'PY'
import json, os
R=os.path.expanduser("~/M_thesis/unwrapping/inr/results/flatcheck")
for lab in ["R1S_clean","R1n_clean","G1_grid_imperfect","R1S_imperfect","R1n_imperfect"]:
    f=os.path.join(R,lab,"eval_metrics.json")
    if not os.path.exists(f): print(f"{lab:20s} (no metrics)"); continue
    d=json.load(open(f))
    print(f"{lab:20s} ssim={d.get('ssim')}  row_corr={d.get('row_corr_mean'):.4f}  "
          f"oracle_vs_gt={d.get('oracle_ssim_vs_gt')}  pred_vs_oracle={d.get('pred_ssim_vs_oracle')}")
PY
echo "done $(date)"
