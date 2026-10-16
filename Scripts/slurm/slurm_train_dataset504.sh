#!/bin/bash
# Train ONLY: Dataset504 (Roll2 2D) from 4 labelled slices, 250 epochs.
# Then run a held-out inference on a few non-training z-locations to
# visually verify generalization. No full-roll prediction, no unwrapping —
# the downstream pipeline is wired up only after the seg is judged good.

#SBATCH --cluster=gmerlin7
#SBATCH --job-name=ds504_train
#SBATCH --output=logs/ds504_train_%j.out
#SBATCH --error=logs/ds504_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=180G
#SBATCH --partition=a100-daily
#SBATCH --time=04:00:00

set -euo pipefail
PROJECT_DIR="$HOME/M_thesis"

DATASET_ID=504
DATASET_NAME="Dataset504_Roll2_2D"
RAW_TIFFS="${PROJECT_DIR}/02_Sample_raw"
LABEL_PROBS="${PROJECT_DIR}/data/roll2_probs"
NN_RAW="${PROJECT_DIR}/nnUNet_data/nnUNet_raw/${DATASET_NAME}"
NN_PRE="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
NN_RES="${PROJECT_DIR}/nnUNet_data/nnUNet_results"

# Held-out verification: a few z-locations NOT in {200, 700, 1200, 1700}
HELDOUT_DIR="${PROJECT_DIR}/nnUNet_data/imagesTs_504_heldout"
HELDOUT_PRED="${PROJECT_DIR}/nnUNet_data/predictions_504_heldout"
HELDOUT_VIS="${PROJECT_DIR}/02_Sample_seg_check_504"
HELDOUT_Z=(50 450 950 1450 1850 100 500)   # 7 spots spread across the roll

export nnUNet_raw="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${NN_PRE}"
export nnUNet_results="${NN_RES}"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "${PROJECT_DIR}"
mkdir -p logs "${NN_RAW}/imagesTr" "${NN_RAW}/labelsTr" \
              "${HELDOUT_DIR}" "${HELDOUT_PRED}" "${HELDOUT_VIS}"

echo "==============================================="
echo "  Dataset504 train-only pipeline"
echo "  start: $(date)"
echo "==============================================="

# --- 1. Build Dataset504 (idempotent) ---
echo ""
echo "--- [1/4] Build Dataset504 ---"
python -u - <<PY
import os, json
import numpy as np
import h5py
from PIL import Image

raw_dir = "${RAW_TIFFS}"
prob_dir = "${LABEL_PROBS}"
img_out = "${NN_RAW}/imagesTr"
lbl_out = "${NN_RAW}/labelsTr"

for z in [200, 700, 1200, 1700]:
    src_tiff = os.path.join(raw_dir, f"{z:04d}.tiff")
    img = np.array(Image.open(src_tiff))
    Image.fromarray(img).save(os.path.join(img_out, f"Sample_{z:04d}_0000.tif"))
    with h5py.File(os.path.join(prob_dir, f"{z:04d}_Probabilities.h5"), "r") as h:
        probs = h["exported_data"][...]
    labels = np.argmax(probs, axis=-1).astype(np.uint8)
    Image.fromarray(labels).save(os.path.join(lbl_out, f"Sample_{z:04d}.tif"))
    print(f"  z={z:04d}: img+label written, class fractions = "
          f"{[(labels==c).mean() for c in [0,1,2]]}")

dataset_json = {
    "channel_names": {"0": "CT"},
    "labels": {"background": 0, "film_base": 1, "emulsion": 2},
    "numTraining": 4,
    "file_ending": ".tif",
    "overwrite_image_reader_writer": "NaturalImage2DIO",
}
with open(os.path.join("${NN_RAW}", "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=2)
print("  dataset.json written")
PY

# --- 2. Plan + preprocess (idempotent) ---
echo ""
echo "--- [2/4] plan + preprocess ---"
nnUNetv2_plan_and_preprocess -d ${DATASET_ID} --verify_dataset_integrity

# --- 3. Train 2D with -f all (no CV split since we only have 4 samples) ---
echo ""
echo "--- [3/4] train 2D fold=all (built-in nnUNetTrainer_250epochs) ---"
nnUNetv2_train ${DATASET_ID} 2d all -tr nnUNetTrainer_250epochs --npz

# --- 4. Held-out inference + visual check ---
echo ""
echo "--- [4/4] Held-out inference on a few z-locations ---"
python -u - <<PY
import os
src = "${RAW_TIFFS}"
dst = "${HELDOUT_DIR}"
for z in ${HELDOUT_Z[@]@K}:
    z_int = int(z)
    name = f"{z_int:04d}"
    target = os.path.join(dst, f"Sample_{name}_0000.tif")
    if not os.path.exists(target):
        os.symlink(os.path.join(src, f"{name}.tiff"), target)
print(f"  staged {len(os.listdir(dst))} held-out symlinks → {dst}")
PY

nnUNetv2_predict -i "${HELDOUT_DIR}" -o "${HELDOUT_PRED}" \
    -d ${DATASET_ID} -c 2d -tr nnUNetTrainer_250epochs -f all \
    --disable_tta

# Build CT+label overlays for visual inspection
python -u - <<PY
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

raw = "${RAW_TIFFS}"
pred_dir = "${HELDOUT_PRED}"
out_dir = "${HELDOUT_VIS}"

for f in sorted(os.listdir(pred_dir)):
    if not f.endswith(".tif"): continue
    z_str = f.replace("Sample_", "").replace(".tif", "")
    z = int(z_str)
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

echo ""
echo "==============================================="
echo "  Dataset504 train + verify done at $(date)"
echo "  visual checks: ${HELDOUT_VIS}/"
echo "==============================================="
