#!/bin/bash
# Segment the FULL continuous Mickey scroll (01_Mickey_hdf, 1184 slices) with the
# trained nnU-Net 3D model (Dataset502, 3d_fullres, fold 0) so every slice can be
# walked densely (fallback if the 500 gapped-chunk slices don't suffice).
#
# Stages:
#   1. chunk 01_Mickey_hdf (per-slice) -> 01_Mickey_full_3d/volume_*.h5 (contiguous 20)
#   2. HDF5 -> NIfTI
#   3. nnUNetv2_predict (3d_fullres, Dataset502, fold 0, save_probabilities)
#   4. softmax -> 01_Mickey_full_3d/volume_*_Probabilities.h5  (walk-ready)
#
# SMOKE TEST:  sbatch Scripts/slurm/slurm_segment_fullscroll.sh 0856 0915   # ~3 chunks
# FULL RUN:    sbatch Scripts/slurm/slurm_segment_fullscroll.sh             # all 1184
#
#SBATCH --clusters=gmerlin7
#SBATCH --job-name=seg_fullscroll
#SBATCH --output=/data/user/li_k1/M_thesis/logs/segfull_%j.out
#SBATCH --error=/data/user/li_k1/M_thesis/logs/segfull_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --partition=a100-daily
#SBATCH --time=12:00:00

set -euo pipefail
PROJECT_DIR="/data/user/li_k1/M_thesis"
PAIRS_DIR="${PROJECT_DIR}/01_Mickey_full_3d"
NNUNET_IN="${PROJECT_DIR}/nnUNet_data/nnUNet_raw/Dataset502_MickeyScroll3D/imagesTs_MickeyFull"
NNUNET_OUT="${PROJECT_DIR}/nnUNet_data/predictions_MickeyFull"
Z_START="${1:-}"
Z_END="${2:-}"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "${PROJECT_DIR}"
mkdir -p logs "${PAIRS_DIR}" "${NNUNET_IN}" "${NNUNET_OUT}"
export nnUNet_raw="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_DIR}/nnUNet_data/nnUNet_results"

ZARGS=""
if [ -n "${Z_START}" ]; then ZARGS="--z-start ${Z_START} --z-end ${Z_END}"; fi
echo "=== full-scroll seg  start $(date)  z-range: ${Z_START:-all}-${Z_END:-all} ==="

echo "--- [1/4] chunk 01_Mickey_hdf -> volumes ---"
python -u Scripts/mickey_hdf_to_chunks.py --in-dir 01_Mickey_hdf \
    --out-dir "${PAIRS_DIR}" --chunk-z 20 ${ZARGS}

echo "--- [2/4] HDF5 -> NIfTI ---"
python -u Scripts/hdf5_to_nifti_for_predict.py \
    --in-dir "${PAIRS_DIR}" --out-dir "${NNUNET_IN}" --prefix MickeyFull

echo "--- [3/4] nnU-Net 3D fullres inference (Dataset502, fold 0) ---"
nnUNetv2_predict -i "${NNUNET_IN}" -o "${NNUNET_OUT}" \
    -d 502 -c 3d_fullres -tr nnUNetTrainerProgress -f 0 \
    --save_probabilities --disable_tta

echo "--- [4/4] softmax -> HDF5 Probabilities ---"
python -u Scripts/nnunet_softmax_to_h5_probs.py \
    --nnunet-out "${NNUNET_OUT}" --pairs-dir "${PAIRS_DIR}" --prefix MickeyFull

echo "=== done $(date).  walk-ready seg in ${PAIRS_DIR} ==="
