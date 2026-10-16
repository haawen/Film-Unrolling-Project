#!/bin/bash
# Mid-roll smoke: 1-chunk segmentation + R1T5 fine-tune + strip preview on 02_Sample.
# Runs in parallel with the full-roll seg job (job 167998), in isolated dirs.
# Mid-roll target: slices 1000-1019 (z=1000..1019, 20 slices).

#SBATCH --cluster=gmerlin7
#SBATCH --job-name=smoke02
#SBATCH --output=logs/smoke02_%j.out
#SBATCH --error=logs/smoke02_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --partition=a100-hourly
#SBATCH --time=01:00:00

set -euo pipefail
PROJECT_DIR="$HOME/M_thesis"
CHUNK="volume_1000-1019.h5"

PAIRS_FULL="${PROJECT_DIR}/02_Sample_3d"
PAIRS_SMOKE="${PROJECT_DIR}/02_Sample_3d_smoke"
NNUNET_IN="${PROJECT_DIR}/nnUNet_data/nnUNet_raw/Dataset502_MickeyScroll3D/imagesTs_02Sample_smoke"
NNUNET_OUT="${PROJECT_DIR}/nnUNet_data/predictions_02Sample_smoke"

# R1T5 outputs
TRAIN_DIR="${PROJECT_DIR}/unwrapping/inr/results/02Sample_R1T5_smoke/train"
EVAL_DIR="${PROJECT_DIR}/unwrapping/inr/results/02Sample_R1T5_smoke/eval"
PREVIEW_DIR="${PROJECT_DIR}/02_Sample_seg_samples_smoke"

INIT_FROM="${PROJECT_DIR}/unwrapping/inr/results/R1_smoke_mickey_geometry/R1S/train/model_final.pt"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "${PROJECT_DIR}"
mkdir -p logs "${PAIRS_SMOKE}" "${NNUNET_IN}" "${NNUNET_OUT}" "${TRAIN_DIR}" "${EVAL_DIR}" "${PREVIEW_DIR}"

export nnUNet_raw="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_DIR}/nnUNet_data/nnUNet_results"

echo "============================================"
echo "  02_Sample mid-roll smoke (chunk ${CHUNK})"
echo "  init-from : ${INIT_FROM}"
echo "  start     : $(date)"
echo "============================================"

# Wait for chunk file to exist (job 167998 may still be in stage 2)
ATTEMPTS=0
while [ ! -f "${PAIRS_FULL}/${CHUNK}" ] && [ "${ATTEMPTS}" -lt 60 ]; do
  echo "[wait] ${CHUNK} not yet present, sleeping 30s..."
  sleep 30
  ATTEMPTS=$((ATTEMPTS + 1))
done
if [ ! -f "${PAIRS_FULL}/${CHUNK}" ]; then
  echo "ERROR: ${CHUNK} never appeared in ${PAIRS_FULL}" >&2; exit 2
fi

# --- 1. Stage the chunk in isolated smoke dir ---
echo ""
echo "--- [1/5] Staging chunk to smoke dir ---"
cp -n "${PAIRS_FULL}/${CHUNK}" "${PAIRS_SMOKE}/${CHUNK}"

# --- 2. HDF5 -> NIfTI ---
echo ""
echo "--- [2/5] HDF5 -> NIfTI ---"
python -u Scripts/hdf5_to_nifti_for_predict.py \
    --in-dir "${PAIRS_SMOKE}" --out-dir "${NNUNET_IN}" --prefix Sample3DSmoke

# --- 3. nnUNet 3D predict (single chunk) ---
echo ""
echo "--- [3/5] nnUNet 3D predict ---"
nnUNetv2_predict \
    -i "${NNUNET_IN}" -o "${NNUNET_OUT}" \
    -d 502 -c 3d_fullres -tr nnUNetTrainerProgress -f 0 \
    --save_probabilities --disable_tta

# --- 4. Softmax -> HDF5 Probabilities ---
echo ""
echo "--- [4/5] Softmax -> HDF5 Probabilities ---"
python -u Scripts/nnunet_softmax_to_h5_probs.py \
    --nnunet-out "${NNUNET_OUT}" --pairs-dir "${PAIRS_SMOKE}" --prefix Sample3DSmoke

# Render seg preview for these 20 slices
python -u Scripts/visualize_seg_samples.py \
    --pairs-dir "${PAIRS_SMOKE}" --out-dir "${PREVIEW_DIR}" --n-samples 5

# --- 5. R1T5 fine-tune + eval ---
echo ""
echo "--- [5/5] R1T5 fine-tune (synth prior + SS on real) ---"
srun python -u -m unwrapping.inr.surface_train \
    --data-dir  "${PAIRS_SMOKE}" \
    --out-dir   "${TRAIN_DIR}" \
    --init-from "${INIT_FROM}" \
    --attachment emulsion --centerline-erode 1 \
    --use-distance-attach \
    --supervised-steps 0 --steps 1500 --ramp-steps 500 \
    --batch-size 131072 --lr 1e-4 \
    --w-attach 1.0 --w-conformal 0.1 --w-res-reg 0.0 \
    --sigma 20.0 --hidden-dim 256 --n-layers-mlp 4 --model-type inr \
    --log-every 100 --ckpt-every 0

srun python -u -m unwrapping.inr.surface_eval \
    --data-dir  "${PAIRS_SMOKE}" \
    --out-dir   "${EVAL_DIR}" \
    --attachment emulsion --centerline-erode 1 \
    --ckpt      "${TRAIN_DIR}/model_final.pt"

# Strip score (unsupervised proxies — no GT for real data)
srun python -u -m unwrapping.inr.strip_score \
    "${EVAL_DIR}/strip.npz" || true

echo ""
echo "============================================"
echo "  02_Sample smoke done at $(date)"
echo "  Strip:    ${EVAL_DIR}/strip_full.png"
echo "  Seg PNGs: ${PREVIEW_DIR}/"
echo "============================================"
