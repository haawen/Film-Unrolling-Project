#!/bin/bash
# Full segmentation pipeline for 02_Sample (new roll).
#
# Stages (each is a no-op if its output already exists):
#   1. unzip 02_Sample.zip → 02_Sample_raw/*.tiff
#   2. chunk TIFFs → 02_Sample_3d/volume_NNNN-MMMM.h5
#   3. convert HDF5 → NIfTI for nnU-Net inference
#   4. nnUNetv2_predict (3D fullres, Dataset502, fold 0, save_probabilities)
#   5. package softmax → 02_Sample_3d/volume_NNNN-MMMM_Probabilities.h5
#   6. render N preview PNGs in 02_Sample_seg_samples/
#
# Run as:   sbatch Scripts/slurm/slurm_segment_sample.sh

#SBATCH --cluster=gmerlin7
#SBATCH --job-name=seg02
#SBATCH --output=logs/seg02_%j.out
#SBATCH --error=logs/seg02_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=180G
#SBATCH --partition=a100-daily
#SBATCH --time=08:00:00

set -euo pipefail
PROJECT_DIR="$HOME/M_thesis"
ZIP="${PROJECT_DIR}/02_Sample.zip"
TIFF_DIR="${PROJECT_DIR}/02_Sample_raw"
PAIRS_DIR="${PROJECT_DIR}/02_Sample_3d"
NNUNET_IN="${PROJECT_DIR}/nnUNet_data/nnUNet_raw/Dataset502_MickeyScroll3D/imagesTs_02Sample"
NNUNET_OUT="${PROJECT_DIR}/nnUNet_data/predictions_02Sample"
PREVIEW_DIR="${PROJECT_DIR}/02_Sample_seg_samples"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "${PROJECT_DIR}"
mkdir -p logs "${TIFF_DIR}" "${PAIRS_DIR}" "${NNUNET_IN}" "${NNUNET_OUT}" "${PREVIEW_DIR}"

export nnUNet_raw="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_DIR}/nnUNet_data/nnUNet_results"

echo "=============================================="
echo "  02_Sample segmentation pipeline"
echo "  start: $(date)"
echo "=============================================="

# ── Stage 1: unzip (skip if TIFFs present) ────────────────────────────────
if [ "$(ls -1 "${TIFF_DIR}" 2>/dev/null | wc -l)" -lt 1900 ]; then
  echo ""
  echo "--- [1/6] Unzipping ${ZIP} ---"
  unzip -q -n "${ZIP}" -d "${TIFF_DIR}"
  echo "Unzipped $(ls -1 "${TIFF_DIR}" | wc -l) files"
else
  echo "[1/6] TIFFs already present in ${TIFF_DIR}, skipping unzip"
fi

# ── Stage 2: chunk TIFFs → HDF5 ───────────────────────────────────────────
echo ""
echo "--- [2/6] Chunking TIFFs → HDF5 volumes ---"
python -u Scripts/tiff_to_hdf5_chunks.py \
    --tiff-dir "${TIFF_DIR}" \
    --out-dir  "${PAIRS_DIR}" \
    --chunk-z  20

# ── Stage 3: convert HDF5 → NIfTI ─────────────────────────────────────────
echo ""
echo "--- [3/6] HDF5 → NIfTI for nnU-Net ---"
python -u Scripts/hdf5_to_nifti_for_predict.py \
    --in-dir  "${PAIRS_DIR}" \
    --out-dir "${NNUNET_IN}" \
    --prefix  Sample3D

# ── Stage 4: nnU-Net 3D inference ─────────────────────────────────────────
echo ""
echo "--- [4/6] nnU-Net 3D fullres inference (Dataset502, fold 0) ---"
nnUNetv2_predict \
    -i "${NNUNET_IN}" \
    -o "${NNUNET_OUT}" \
    -d 502 \
    -c 3d_fullres \
    -tr nnUNetTrainerProgress \
    -f 0 \
    --save_probabilities \
    --disable_tta

# ── Stage 5: package softmax → HDF5 Probabilities ─────────────────────────
echo ""
echo "--- [5/6] Softmax → HDF5 Probabilities ---"
python -u Scripts/nnunet_softmax_to_h5_probs.py \
    --nnunet-out "${NNUNET_OUT}" \
    --pairs-dir  "${PAIRS_DIR}" \
    --prefix     Sample3D

# ── Stage 6: preview PNGs ─────────────────────────────────────────────────
echo ""
echo "--- [6/6] Rendering preview PNGs ---"
python -u Scripts/visualize_seg_samples.py \
    --pairs-dir "${PAIRS_DIR}" \
    --out-dir   "${PREVIEW_DIR}" \
    --n-samples 8

echo ""
echo "=============================================="
echo "  done at $(date)"
echo "  PNG previews: ${PREVIEW_DIR}"
echo "=============================================="
