#!/bin/bash
# Held-out inference + visual check on the trained Dataset504 model.
# Predicts 7 z-locations not in {200, 700, 1200, 1700} and renders
# CT/pred/overlay 3-panel PNGs for visual judgement.

#SBATCH --cluster=gmerlin7
#SBATCH --job-name=ds504_check
#SBATCH --output=logs/ds504_check_%j.out
#SBATCH --error=logs/ds504_check_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --partition=a100-hourly
#SBATCH --time=00:30:00

set -euo pipefail
PROJECT_DIR="$HOME/M_thesis"
DATASET_ID=504
RAW_TIFFS="${PROJECT_DIR}/02_Sample_raw"
HELDOUT_DIR="${PROJECT_DIR}/nnUNet_data/imagesTs_504_heldout"
HELDOUT_PRED="${PROJECT_DIR}/nnUNet_data/predictions_504_heldout"
HELDOUT_VIS="${PROJECT_DIR}/02_Sample_seg_check_504"

export nnUNet_raw="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_DIR}/nnUNet_data/nnUNet_results"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "${PROJECT_DIR}"
mkdir -p logs "${HELDOUT_DIR}" "${HELDOUT_PRED}" "${HELDOUT_VIS}"

# Stage held-out symlinks (all-Python, no bash array tricks)
python -u - <<'PY'
import os
src = os.path.expanduser("~/M_thesis/02_Sample_raw")
dst = os.path.expanduser("~/M_thesis/nnUNet_data/imagesTs_504_heldout")
for z in [50, 100, 450, 500, 950, 1450, 1850]:
    name = f"{z:04d}"
    target = os.path.join(dst, f"Sample_{name}_0000.tif")
    if not os.path.exists(target):
        os.symlink(os.path.join(src, f"{name}.tiff"), target)
print(f"staged {len(os.listdir(dst))} symlinks → {dst}")
PY

nnUNetv2_predict -i "${HELDOUT_DIR}" -o "${HELDOUT_PRED}" \
    -d ${DATASET_ID} -c 2d -tr nnUNetTrainer_250epochs -f all \
    --disable_tta

# Render CT / Pred / Overlay 3-panel PNGs
python -u - <<'PY'
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

raw = os.path.expanduser("~/M_thesis/02_Sample_raw")
pred_dir = os.path.expanduser("~/M_thesis/nnUNet_data/predictions_504_heldout")
out_dir = os.path.expanduser("~/M_thesis/02_Sample_seg_check_504")

for f in sorted(os.listdir(pred_dir)):
    if not f.endswith(".tif"): continue
    z = int(f.replace("Sample_", "").replace(".tif", ""))
    ct = np.array(Image.open(os.path.join(raw, f"{z:04d}.tiff")))
    pred = np.array(Image.open(os.path.join(pred_dir, f)))
    fracs = [(pred == c).mean() for c in [0, 1, 2]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(ct, cmap="gray"); axes[0].set_title(f"CT z={z}"); axes[0].axis("off")
    cmap = np.zeros((*pred.shape, 3), dtype=np.uint8)
    cmap[pred == 1] = [0, 150, 200]
    cmap[pred == 2] = [255, 0, 150]
    axes[1].imshow(cmap)
    axes[1].set_title(f"Pred  cyan=base ({fracs[1]:.3f})  magenta=emul ({fracs[2]:.3f})")
    axes[1].axis("off")
    ct_n = ((ct - ct.min()) / max(1, ct.max() - ct.min()) * 255).astype(np.uint8)
    overlay = (0.55 * np.stack([ct_n] * 3, axis=-1) + 0.45 * cmap).astype(np.uint8)
    axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"check_z{z:04d}.png"), dpi=110)
    plt.close()
    print(f"  wrote check_z{z:04d}.png  (emul {fracs[2]*100:.2f}%)")
PY

echo "Done at $(date)"
