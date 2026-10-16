#!/usr/bin/env bash
# Segment and walk one of five 15-chunk batches from the full-width scan.
# Run from the repository/handoff root after exporting nnUNet_* as in README.md.
# Usage: bash Scripts/reproduce_fullwidth_batch.sh 0   # then 1, 2, 3, 4
set -euo pipefail

batch="${1:?usage: bash Scripts/reproduce_fullwidth_batch.sh <0..4>}"
[[ "$batch" =~ ^[0-4]$ ]] || { echo "batch must be 0..4" >&2; exit 2; }

project_root="${PROJECT_ROOT:-$(pwd)}"
project_root="$(cd "$project_root" && pwd)"
ct_dir="${CT_DIR:-$project_root/work/01_Mickey_sprockets}"
run_dir="${RUN_DIR:-$project_root/work/fullwidth}"
export nnUNet_raw="${nnUNet_raw:-$project_root/Segmentation/nnUNet_data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$project_root/Segmentation/nnUNet_data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$project_root/Segmentation/nnUNet_data/nnUNet_results}"

cd "$project_root"
[[ -f "$ct_dir/Mickey_merged_0470.h5" ]] || {
  echo "Missing per-slice CT data in $ct_dir; export the TIFF zip first." >&2; exit 1;
}
batch_dir="$run_dir/b$batch"
pairs="$batch_dir/pairs"
nii="$batch_dir/nii"
pred="$batch_dir/pred"
walk="$batch_dir/walk"
mkdir -p "$pairs" "$nii" "$pred" "$walk"

for ((i=batch*15; i<(batch+1)*15; i++)); do
  z0=$((470 + 20*i))
  z1=$((z0 + 19))
  python Scripts/newscan_seg_test.py chunks \
    --in-dir "$ct_dir" --out-dir "$pairs" --z0 "$z0" --z1 "$z1" --scale 1.0
done

python Scripts/hdf5_to_nifti_for_predict.py \
  --in-dir "$pairs" --out-dir "$nii" --prefix NewScan
nnUNetv2_predict -i "$nii" -o "$pred" \
  -d 502 -c 3d_fullres -tr nnUNetTrainerProgress -f 0 \
  --save_probabilities --disable_tta
python Scripts/nnunet_softmax_to_h5_probs.py \
  --nnunet-out "$pred" --pairs-dir "$pairs" --prefix NewScan

# The inspection flag writes walk_anchors.npz and diagnostic overlays, then
# skips this batch's unnecessary whole-roll render. Final rendering happens
# after all five anchor caches have been merged.
python -m unwrapping.inr.unroll_walk_wholeroll_v13 \
  --ct-dir "$ct_dir" --seg-dir "$pairs" --out-dir "$walk" \
  --slice-frac 1.0 --n-centers 6 --seam-deg 353.0 \
  --smooth-px 0.75 --snap-max 10 --step 2.5 --coast-max 50 \
  --film-min-thick 7.5 --jump-max 5 --snap-accept 7.5 --max-dr 2.5 \
  --recenter-search 7.5 --recenter-sigma 25 \
  --min-sep 7.5 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
  --z-heal-px 7.5 --min-coverage 100 --transverse-px 3 \
  --max-infill 3 --radial-robust --seam-exclude-deg 0.3 \
  --inspect-anchors --inspect-max 3

[[ -s "$walk/walk_anchors.npz" ]] || {
  echo "Walk cache was not written; retaining all intermediates." >&2; exit 1;
}

# The predicted softmax files are very large. Reclaim them only after the
# cache exists. Every target must resolve under this batch's run directory.
if [[ "${CLEAN_TEMP:-1}" == 1 ]]; then
  python - "$batch_dir" "$pairs" "$nii" "$pred" <<'PY'
from pathlib import Path
import shutil
import sys
root = Path(sys.argv[1]).resolve()
for value in sys.argv[2:]:
    target = Path(value).resolve()
    if target == root or not target.is_relative_to(root):
        raise SystemExit(f"Unsafe cleanup target: {target}")
    shutil.rmtree(target)
    print(f"Removed temporary batch data: {target}")
PY
fi
echo "Batch $batch complete: $walk/walk_anchors.npz"
