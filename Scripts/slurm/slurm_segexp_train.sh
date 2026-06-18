#!/bin/bash
# =============================================================================
# Seg-sensitivity experiment, stage 1: transfer-learn nnU-Net 3D on synthetic.
#
# Builds a chunked 3D nnU-Net dataset from the CLEAN HQ preset's GT masks,
# transfers Dataset502's plans (so the architecture matches the real model),
# and fine-tunes from the Dataset502 3d_fullres fold-0 checkpoint. The result
# is a seg model in the synthetic intensity domain whose generalization error
# on the (unseen) imperfect preset gives realistic, model-quality masks.
#
#   sbatch Scripts/slurm/slurm_segexp_train.sh
# =============================================================================
#SBATCH --cluster=gmerlin7
#SBATCH --job-name=segexp_tr
#SBATCH --output=logs/segexp_tr_%j.out
#SBATCH --error=logs/segexp_tr_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --partition=a100-daily
#SBATCH --time=23:59:00

set -euo pipefail
PROJECT_DIR="$HOME/M_thesis"
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "${PROJECT_DIR}"
mkdir -p logs

export nnUNet_raw="${PROJECT_DIR}/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_DIR}/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="${PROJECT_DIR}/nnUNet_data/nnUNet_results"

DID=510
DNAME="Dataset510_SynthClean"
RAW="${nnUNet_raw}/${DNAME}"
CLEAN="${PROJECT_DIR}/unwrapping/synthetic/results/hq_video_4k_outer"
PRETRAINED="${nnUNet_results}/Dataset502_MickeyScroll3D/nnUNetTrainerProgress__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
EPOCHS_TR="nnUNetTrainer_100epochs"

echo "=== [1/5] build chunked dataset from CLEAN ($(date)) ==="
if [ ! -f "${RAW}/dataset.json" ]; then
  python -u Scripts/build_synth_nnunet_dataset.py \
      --syn-dir "${CLEAN}" --mode train --chunk 24 \
      --prefix Synth3D --raw-dir "${RAW}"
else
  echo "  ${RAW}/dataset.json exists, skip build"
fi

echo "=== [2/5] extract fingerprint ($(date)) ==="
nnUNetv2_extract_fingerprint -d ${DID} --verify_dataset_integrity

echo "=== [3/5] move Dataset502 plans -> ${DID} (architecture match) ($(date)) ==="
nnUNetv2_move_plans_between_datasets -s 502 -t ${DID} -sp nnUNetPlans -tp nnUNetPlans

echo "=== [4/5] preprocess 3d_fullres ($(date)) ==="
nnUNetv2_preprocess -d ${DID} -plans_name nnUNetPlans -c 3d_fullres

echo "=== [5/5] transfer-learn (fold 0, ${EPOCHS_TR}, pretrained from 502) ($(date)) ==="
nnUNetv2_train ${DID} 3d_fullres 0 -tr ${EPOCHS_TR} \
    -pretrained_weights "${PRETRAINED}" --npz

echo "=== done $(date) ==="
echo "checkpoint: ${nnUNet_results}/${DNAME}/${EPOCHS_TR}__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
