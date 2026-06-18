#!/bin/bash
# End-to-end: build Dataset504 (Roll2 2D) from 4 labelled slices →
# plan+preprocess → train 250 epochs → predict on all 1936 slices →
# stack into 97-chunk Probabilities.h5 matching 02_Sample_3d/.

#SBATCH --cluster=gmerlin7
#SBATCH --job-name=ds504
#SBATCH --output=logs/ds504_%j.out
#SBATCH --error=logs/ds504_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=180G
#SBATCH --partition=a100-daily
#SBATCH --time=08:00:00

set -euo pipefail
PROJECT_DIR="$HOME/M_thesis"

DATASET_ID=504
DATASET_NAME="Dataset504_Roll2_2D"
RAW_TIFFS="${PROJECT_DIR}/02_Sample_raw"
LABEL_PROBS="${PROJECT_DIR}/data/roll2_probs"
PAIRS="${PROJECT_DIR}/02_Sample_3d"
N_EPOCHS=250

NN_RAW="${PROJECT_DIR}/nnUNet_data/nnUNet_raw/${DATASET_NAME}"
NN_PRE="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
NN_RES="${PROJECT_DIR}/nnUNet_data/nnUNet_results"
PRED_RAW="${PROJECT_DIR}/nnUNet_data/imagesTs_02Sample_full_2d"
PRED_OUT="${PROJECT_DIR}/nnUNet_data/predictions_02Sample_full_2d"

export nnUNet_raw="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${NN_PRE}"
export nnUNet_results="${NN_RES}"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "${PROJECT_DIR}"
mkdir -p logs "${NN_RAW}/imagesTr" "${NN_RAW}/labelsTr" "${PRED_RAW}" "${PRED_OUT}"

echo "==============================================="
echo "  Dataset504 (Roll2 2D) end-to-end pipeline"
echo "  start: $(date)"
echo "==============================================="

# --- 1. Build Dataset504 from raw TIFFs + Ilastik probabilities ---
echo ""
echo "--- [1/6] Build Dataset504 ---"
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
    # nnU-Net 2D expects single-channel uint16 image with _0000 suffix.
    Image.fromarray(img).save(os.path.join(img_out, f"Sample_{z:04d}_0000.tif"))

    with h5py.File(os.path.join(prob_dir, f"{z:04d}_Probabilities.h5"), "r") as h:
        probs = h["exported_data"][...]    # (H, W, 3) float32
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

# --- 2. Plan + preprocess ---
echo ""
echo "--- [2/6] plan + preprocess ---"
nnUNetv2_plan_and_preprocess -d ${DATASET_ID} --verify_dataset_integrity

# --- 3. Train 2D fold 0 ---
echo ""
echo "--- [3/6] train 2D fold 0 (${N_EPOCHS} epochs) ---"
# Subclass the default trainer to override num_epochs.
python -u - <<PY
import os
# Mirror nnUNetTrainer's exact __init__ signature so its locals()-based
# my_init_kwargs capture works. Star-args break that capture.
trainer_src = '''
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerShort(nnUNetTrainer):
    def __init__(self,
                 plans: dict,
                 configuration: str,
                 fold: int,
                 dataset_json: dict,
                 unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.num_epochs = ${N_EPOCHS}
        self.save_every = 50
'''
import nnunetv2.training.nnUNetTrainer as pkg
target = os.path.join(os.path.dirname(pkg.__file__), "variants", "nnUNetTrainerShort.py")
os.makedirs(os.path.dirname(target), exist_ok=True)
with open(target, "w") as f:
    f.write(trainer_src)
print(f"  trainer written to {target}")
PY
nnUNetv2_train ${DATASET_ID} 2d 0 -tr nnUNetTrainerShort --npz

# --- 4. Predict on all 1936 raw TIFFs ---
echo ""
echo "--- [4/6] stage 1936 raw TIFFs for prediction ---"
python -u - <<PY
import os, shutil
src = "${RAW_TIFFS}"
dst = "${PRED_RAW}"
os.makedirs(dst, exist_ok=True)
for f in sorted(os.listdir(src)):
    if not f.endswith(".tiff"):
        continue
    name = os.path.splitext(f)[0]  # e.g. "0200"
    target = os.path.join(dst, f"Sample_{name}_0000.tif")
    if not os.path.exists(target):
        os.symlink(os.path.join(src, f), target)
print(f"  staged {len(os.listdir(dst))} symlinks → {dst}")
PY

echo ""
echo "--- [5/6] nnUNet predict on 1936 slices ---"
nnUNetv2_predict -i "${PRED_RAW}" -o "${PRED_OUT}" \
    -d ${DATASET_ID} -c 2d -tr nnUNetTrainerShort -f 0 \
    --save_probabilities --disable_tta

# --- 6. Stack per-slice softmax → per-chunk Probabilities.h5 ---
echo ""
echo "--- [6/6] Stack 1936 per-slice softmax → 97 chunk Probabilities.h5 ---"
python -u - <<PY
import os, glob, re
import numpy as np
import h5py

pred = "${PRED_OUT}"
pairs = "${PAIRS}"

# Collect per-slice .npz softmax files (nnU-Net 2D names them Sample_<NNNN>.npz)
files = sorted(glob.glob(os.path.join(pred, "Sample_*.npz")))
print(f"  found {len(files)} per-slice softmax outputs")

# Group by chunk based on existing volume_*.h5 files in PAIRS
chunk_files = sorted(glob.glob(os.path.join(pairs, "volume_*-*.h5")))
print(f"  found {len(chunk_files)} existing CT chunks in {pairs}")

for cf in chunk_files:
    name = os.path.basename(cf)
    m = re.match(r"volume_(\d+)-(\d+)\.h5", name)
    if not m: continue
    z0, z1 = int(m.group(1)), int(m.group(2))
    out_path = os.path.join(pairs, name.replace(".h5", "_Probabilities.h5"))
    if os.path.exists(out_path) and os.path.getsize(out_path) > 10 * 1024 * 1024:
        print(f"  skip {name} (probs exist)"); continue

    # The TIFF naming starts at 0001.tiff = z=0 in chunker order? Actually the
    # chunker uses sorted file order. Filenames are 0001.tiff..1936.tiff so
    # filename-1 == zero-based index. Chunk z0=1000 means filenames 1001..1020.
    # But chunk volume_1000-1019.h5 contains the slices in chunker zero-index
    # 1000..1019 → filenames 1001.tiff..1020.tiff (assuming 1-indexed input).
    # ACTUAL: the chunker reads sorted .tiff files and labels chunks by index.
    # Check tiff_to_hdf5_chunks.py to confirm; here we assume filename i+1
    # maps to chunker index i.
    stack = []
    for i in range(z0, z1 + 1):
        # Match: chunker index i ↔ filename (i+1). e.g. index 1000 ↔ 1001.tiff
        # ↔ Sample_1001.npz
        nf = os.path.join(pred, f"Sample_{i+1:04d}.npz")
        if not os.path.exists(nf):
            raise SystemExit(f"missing softmax: {nf}")
        d = np.load(nf)
        # nnU-Net 2D saves as (C, 1, H, W) or (C, H, W). Squeeze + transpose.
        p = d["probabilities"]
        if p.ndim == 4 and p.shape[1] == 1:
            p = p[:, 0]   # (C, H, W)
        if p.shape[0] in (2, 3, 4):
            p = np.transpose(p, (1, 2, 0))   # (H, W, C)
        stack.append(p.astype(np.float32))
    arr = np.stack(stack, axis=0)        # (Z, H, W, C)
    Z, H, W, C = arr.shape
    chunks = (min(8, Z), min(512, H), min(512, W), C)
    with h5py.File(out_path, "w") as h:
        h.create_dataset("exported_data", data=np.ascontiguousarray(arr),
                         compression="lzf", chunks=chunks)
    print(f"  wrote {name.replace('.h5','_Probabilities.h5')} shape={arr.shape}")
PY

# Render seg previews for visual verification
python -u Scripts/visualize_seg_samples.py \
    --pairs-dir "${PAIRS}" --out-dir "${PROJECT_DIR}/02_Sample_seg_samples_504" --n-samples 8 || true

echo ""
echo "==============================================="
echo "  Dataset504 pipeline done at $(date)"
echo "==============================================="
