# Source this file before running anything:   source Code/env.sh
#
# Sets the deliverable root, makes the `virtual_unrolling` package importable and
# points nnU-Net / MONAI at the shipped data and models. Every script also has
# these locations as defaults, so this file only needs editing if you move parts
# of the deliverable elsewhere.

# Root of the deliverable (the folder containing README.md, Data/, Code/, Results/)
export THESIS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$THESIS_ROOT/Code${PYTHONPATH:+:$PYTHONPATH}"

# Scratch space for everything that is (re)generated: nnU-Net preprocessing,
# retrained models, new walks/renders, evaluation outputs.
export WORK="${WORK:-$THESIS_ROOT/work}"

# nnU-Net v2
export nnUNet_raw="$THESIS_ROOT/../Segmentation/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$WORK/nnUNet_preprocessed}"
# Shipped (trained) models. To RETRAIN instead, point this at a writable folder:
#   export nnUNet_results=$WORK/nnUNet_results
export nnUNet_results="${nnUNet_results:-$THESIS_ROOT/Data/2_trained_models/nnUNet_results}"

# MONAI baselines (UNet3D / SwinUNETR)
export MONAI_RESULTS="${MONAI_RESULTS:-$THESIS_ROOT/Data/2_trained_models/monai_results}"
export FAIR_DIR="${FAIR_DIR:-$WORK/fair_comparison}"

# Frequently used data locations (used in the README commands)
export D_SCAN="$THESIS_ROOT/Data/3_full_width_ct_scan"
export GT_VIDEO="$THESIS_ROOT/Data/4_optical_ground_truth/optical_scan_h265_1080p.mp4"
export GT_PERFLOCK="$THESIS_ROOT/Data/4_optical_ground_truth/optical_scan_perflocked_picture.mp4"

mkdir -p "$WORK"
echo "THESIS_ROOT=$THESIS_ROOT"
