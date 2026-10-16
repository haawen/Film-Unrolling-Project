#!/bin/bash
# QUICK TEST: does the existing nnU-Net (Dataset502 3d_fullres fold0, trained on the
# OLD scan) transfer to the NEW scan? Two 20-slice PICTURE-BAND chunks, no sprockets:
#   native  -- full new-scan resolution   (scale + contrast shift)
#   scaled  -- downsampled by 1.2508      (contrast shift only)
# Outputs overlay PNGs + thickness stats to compare against the old scan's
# film 18.0 / emulsion 4.0 / spacing 21.0 px (x1.2508 for native).
#
#   sbatch Scripts/slurm/slurm_newscan_segtest.sh
#
#SBATCH --clusters=gmerlin7
#SBATCH --job-name=newscan_segtest
#SBATCH --output=/data/user/li_k1/M_thesis/logs/newscan_segtest_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --partition=a100-hourly
#SBATCH --time=01:00:00

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export nnUNet_raw="$PROJ/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="$PROJ/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="$PROJ/nnUNet_data/nnUNet_results"

BASE="$PROJ/newscan_segtest"
rm -rf "$BASE"; mkdir -p "$BASE"

# picture-band slices (survey: picture band ~z470-1975); z500-519 is inside it and
# inside the uploaded range, and carries real movie content (not margin).
for V in native scaled; do
  if [ "$V" = native ]; then SC=1.0; else SC=1.2508; fi
  echo "=== [$V] build chunk (scale $SC) ==="
  python -u Scripts/newscan_seg_test.py chunks \
      --in-dir 01_Mickey_sprockets --out-dir "$BASE/$V/vol" \
      --z0 500 --z1 519 --scale "$SC"

  echo "=== [$V] hdf5 -> nifti ==="
  python -u Scripts/hdf5_to_nifti_for_predict.py \
      --in-dir "$BASE/$V/vol" --out-dir "$BASE/$V/nii" --prefix NewScan

  echo "=== [$V] predict ==="
  nnUNetv2_predict -i "$BASE/$V/nii" -o "$BASE/$V/pred" \
      -d 502 -c 3d_fullres -tr nnUNetTrainerProgress -f 0 --disable_tta

  echo "=== [$V] check ==="
  python -u Scripts/newscan_seg_test.py check \
      --pred "$BASE/$V/pred/NewScan_0500.nii.gz" \
      --vol "$BASE/$V/vol/volume_0500-0519.h5" \
      --out-dir "$BASE/checks" --tag "$V"
done

echo "=== DONE. overlays in $BASE/checks ==="
ls -la "$BASE/checks"
