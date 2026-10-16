#!/bin/bash
# TEN evenly-spaced Roll2 anchors via the 2D net: segment (GPU) then walk (CPU).
#
# Roll2 has only 4 segmented slices (z 200/700/1200/1700) ~500 apart, and a
# tracked winding drifts ~1.4 windings over that gap -- which is why the rough
# strip smeared. Ten anchors ~170 apart should hold each track on its own layer.
#
# Uses nnU-Net 2D Dataset503 (0.9717 Dice on Mickey vs the 3D's 0.9728) instead
# of the 3D net the original 4 anchors used: a 2D net needs no z-context, so a
# single slice costs seconds instead of a whole 20-slice 3064^2 volume.
# Dataset503's training TIFs are RAW CT as float32 (no normalisation) -- match
# that exactly or the net sees out-of-distribution intensities.
#
#   sbatch Scripts/slurm/slurm_roll2_ten2d.sh seg    # GPU, ~5 min  (gmerlin7)
#   sbatch Scripts/slurm/slurm_roll2_ten2d.sh walk   # CPU, ~70 min (merlin7)
#
#SBATCH --job-name=roll2_ten2d
#SBATCH --output=/data/user/li_k1/M_thesis/logs/roll2_ten2d_%j.out
#SBATCH --error=/data/user/li_k1/M_thesis/logs/roll2_ten2d_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail
MODE="${1:-seg}"
PROJECT_DIR="/data/user/li_k1/M_thesis"
SRC_CHUNKS="${PROJECT_DIR}/02_Sample_3d"
IMG_IN="${PROJECT_DIR}/02_Sample_ten2d/images"
LBL_OUT="${PROJECT_DIR}/02_Sample_ten2d/labels"
OUT="${PROJECT_DIR}/unwrapping/inr/results/roll2_ten2d"

Z0S="200 360 540 700 880 1060 1220 1400 1580 1740"
IDX=10                                   # mid slice of each 20-slice chunk

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "${PROJECT_DIR}"
mkdir -p logs "${IMG_IN}" "${LBL_OUT}" "${OUT}"
export nnUNet_raw="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_DIR}/nnUNet_data/nnUNet_results"

echo "=== roll2 ten anchors, 2D net (${MODE})  $(date) ==="

if [ "${MODE}" = "seg" ]; then
  echo "--- [1/2] extract the 10 CT slices as float32 TIFs ---"
  python -u - <<PY
import glob, os
import h5py, numpy as np, tifffile
src = "${SRC_CHUNKS}"; out = "${IMG_IN}"; idx = ${IDX}
for z0 in [${Z0S// /, }]:
    cand = [f for f in glob.glob(os.path.join(src, "volume_%04d-*.h5" % z0))
            if "Probabilities" not in f]
    with h5py.File(cand[0], "r") as f:
        key = "volume" if "volume" in f else list(f.keys())[0]
        sl = f[key][idx].astype(np.float32)      # RAW CT, no normalisation
    z = z0 + idx
    tifffile.imwrite(os.path.join(out, "Roll2Ten_%04d_0000.tif" % z), sl)
    print("  z=%d  %s  min=%.0f max=%.0f" % (z, sl.shape, sl.min(), sl.max()))
PY

  echo "--- [2/2] nnU-Net 2D (Dataset503 fold 0, zero-shot on Roll2) ---"
  nnUNetv2_predict -i "${IMG_IN}" -o "${LBL_OUT}" \
      -d 503 -c 2d -tr nnUNetTrainerProgress -f 0 --disable_tta
  ls -la "${LBL_OUT}"
  echo "=== seg done $(date) — now: sbatch Scripts/slurm/slurm_roll2_ten2d.sh walk ==="
  exit 0
fi

echo "--- walk + CT|SEG|WALK panels ---"
python -u -m unwrapping.inr.roll2_ten_slices \
    --chunk-dir "${SRC_CHUNKS}" --label-dir "${LBL_OUT}" --prefix Roll2Ten \
    --chunk-z0s "$(echo ${Z0S} | tr ' ' ',')" --idx ${IDX} \
    --out-dir "${OUT}"

echo "=== done $(date) ==="
ls -la "${OUT}"
