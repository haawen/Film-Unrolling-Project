#!/bin/bash
# TEN evenly-spaced Roll2 anchors: segment + walk + CT|SEG|WALK panels.
#
# Roll2 has only 4 segmented slices (z 200/700/1200/1700), two of which walk
# poorly, so the 4-anchor z-interpolation smears the rough strip. This segments
# 10 evenly-spaced 20-slice chunks with the SAME model the 4 originals used
# (Dataset502 nnU-Net 3D fold 0, zero-shot -- Roll2 has no labels), then walks
# the mid slice of each and draws the three panels.
#
# NOTE the whole 20-slice softmax is kept per chunk (not just the one slice we
# draw): those chunks are exactly the anchors a real 10-anchor render would use,
# and dense walks inside them are then free.
#
#   sbatch Scripts/slurm/slurm_roll2_ten.sh            # seg + walk
#   sbatch Scripts/slurm/slurm_roll2_ten.sh walkonly   # re-walk existing seg
#
#SBATCH --clusters=gmerlin7
#SBATCH --job-name=roll2_ten
#SBATCH --output=/data/user/li_k1/M_thesis/logs/roll2_ten_%j.out
#SBATCH --error=/data/user/li_k1/M_thesis/logs/roll2_ten_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --partition=a100-daily
#SBATCH --time=06:00:00

set -euo pipefail
MODE="${1:-full}"
PROJECT_DIR="/data/user/li_k1/M_thesis"
SRC_CHUNKS="${PROJECT_DIR}/02_Sample_3d"
STAGE="${PROJECT_DIR}/02_Sample_ten"
NN_IN="${PROJECT_DIR}/nnUNet_data/nnUNet_raw/Dataset502_MickeyScroll3D/imagesTs_Roll2Ten"
NN_OUT="${PROJECT_DIR}/nnUNet_data/predictions_Roll2Ten"
OUT="${PROJECT_DIR}/unwrapping/inr/results/roll2_ten"

# evenly spaced chunk starts across Roll2's good film band
Z0S="200 360 540 700 880 1060 1220 1400 1580 1740"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "${PROJECT_DIR}"
mkdir -p logs "${STAGE}" "${NN_IN}" "${NN_OUT}" "${OUT}"
export nnUNet_raw="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_DIR}/nnUNet_data/nnUNet_results"

echo "=== roll2 ten anchors (${MODE})  $(date) ==="

if [ "${MODE}" != "walkonly" ]; then
  echo "--- [1/4] stage the 10 chunks ---"
  for z0 in ${Z0S}; do
    src=$(ls ${SRC_CHUNKS}/volume_$(printf %04d ${z0})-*.h5 2>/dev/null \
          | grep -v Probabilities | head -1)
    if [ -z "${src}" ]; then echo "  MISSING chunk z0=${z0}"; exit 1; fi
    ln -sf "${src}" "${STAGE}/$(basename ${src})"
    echo "  staged $(basename ${src})"
  done

  echo "--- [2/4] HDF5 -> NIfTI ---"
  python -u Scripts/hdf5_to_nifti_for_predict.py \
      --in-dir "${STAGE}" --out-dir "${NN_IN}" --prefix Roll2Ten

  echo "--- [3/4] nnU-Net 3D fullres (Dataset502 fold 0, zero-shot on Roll2) ---"
  nnUNetv2_predict -i "${NN_IN}" -o "${NN_OUT}" \
      -d 502 -c 3d_fullres -tr nnUNetTrainerProgress -f 0 \
      --save_probabilities --disable_tta

  echo "--- [3b] softmax -> Probabilities.h5 ---"
  python -u Scripts/nnunet_softmax_to_h5_probs.py \
      --nnunet-out "${NN_OUT}" --pairs-dir "${STAGE}" --prefix Roll2Ten
fi

echo "--- [3c] free the transient softmax/NIfTI (disk is tight) ---"
rm -f "${NN_OUT}"/*.npz "${NN_OUT}"/*.nii.gz "${NN_IN}"/*.nii.gz || true
df -h /data/user/li_k1 | tail -1

echo "--- [4/4] walk + CT|SEG|WALK panels ---"
# chunk-dir and probs-dir are the SAME staging dir: the CT chunk and its
# Probabilities.h5 sit side by side, so the panel script reads both at the same
# local index and they are z-aligned by construction.
python -u -m unwrapping.inr.roll2_ten_slices \
    --chunk-dir "${STAGE}" --probs-dir "${STAGE}" \
    --chunk-z0s "$(echo ${Z0S} | tr ' ' ',')" --idx 10 \
    --out-dir "${OUT}"

echo "=== done $(date) ==="
ls -la "${OUT}"
