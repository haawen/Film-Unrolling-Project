#!/bin/bash
# LOW BAND z520-779: re-segment, sparse walk, and draw overlays for inspection.
#
# WHY THIS RANGE, NOT b0's z470-769. b0 was launched at Z_START=470 on CLAUDE.md's
# recorded picture band (z470-1975). That is wrong at the low end, and
# Scripts/diag_band_edges.py now measures where the film actually becomes clean
# (fixed Otsu from a reference picture slice; "gaps/ring" counts air runs > 1.5
# deg around each radius ring, so it counts perforation holes rather than their
# area):
#
#     z420-460  cover 0.555-0.582   gaps 12.5-12.8   full perforation
#     z470      cover 0.669         gaps 11.13       <- b0 started HERE
#     z500      cover 0.831         gaps 6.08
#     z510      cover 0.828         gaps 7.51        transition, still ragged
#     z535      cover 0.784         gaps 6.05
#     z600+     cover ~0.757        gaps ~5.0        clean picture band
#     z1935+                        gaps 6.1->12.9   upper perforation
#
# So the walk was fed ~50 slices of perforation. Two things followed: the walks
# there are junk (the z470 and z480 overlays show dashed bands, radial jogs and
# zigzag scribbles), and -- the expensive one -- b0's find_spool_center samples
# were taken on that material. b0's centre track spans 92 px with 46 anchors at
# z470-515 sitting 94 px off a robust line (cx 1812 vs 1904); b1/b2/b3, all on
# clean picture, are sub-pixel (max residual 0.57/0.28/0.35 px).
#
# Starting at z520 rather than z500: gaps/ring peaks at 7.5 around z510 and is
# back to 6.05 by z535. z520 is the first chunk boundary past the peak. Slices
# below it still render -- v13's renderer clamps below the first anchor -- they
# just reuse z520's geometry instead of geometry walked on perforations.
#
# SPARSE ON PURPOSE (--slice-frac 0.25, 70 anchors). This run exists to be looked
# at before committing to the dense one, per the standing fit-check rule. The
# segmentation is KEPT on disk so the dense re-walk reuses it -- that is the
# expensive half (~42 GB, ~13 chunks) and it does not need doing twice.
#
# --n-centers 3 only: the walk is centre-free after seeding, so centres barely
# affect a ranking run. The dense follow-up gets the full share.
#
#   sbatch Scripts/slurm/slurm_lowband.sh
#
#SBATCH --clusters=gmerlin7
#SBATCH --job-name=lowband
#SBATCH --output=/data/user/li_k1/M_thesis/logs/lowband_%j.out
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=180G
#SBATCH --partition=a100-daily
#SBATCH --time=08:00:00

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export nnUNet_raw="$PROJ/nnUNet_data/nnUNet_raw"
export nnUNet_preprocessed="$PROJ/nnUNet_data/nnUNet_preprocessed"
export nnUNet_results="$PROJ/nnUNet_data/nnUNet_results"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

BASE="$PROJ/newscan_low"; PAIRS="$BASE/pairs"
OUT="$PROJ/unwrapping/inr/results/newscan_low/sparse"
rm -rf "$BASE"; mkdir -p "$PAIRS" "$BASE/nii" "$OUT"

echo "=== build chunks z520-779 ==="
for Z0 in 520 540 560 580 600 620 640 660 680 700 720 740 760; do
  Z1=$((Z0+19))
  python -u Scripts/newscan_seg_test.py chunks \
      --in-dir 01_Mickey_sprockets --out-dir "$PAIRS" --z0 "$Z0" --z1 "$Z1" --scale 1.0
done
df -h /data | tail -1

echo "=== nifti + predict ==="
python -u Scripts/hdf5_to_nifti_for_predict.py \
    --in-dir "$PAIRS" --out-dir "$BASE/nii" --prefix NewScan
nnUNetv2_predict -i "$BASE/nii" -o "$BASE/pred" \
    -d 502 -c 3d_fullres -tr nnUNetTrainerProgress -f 0 \
    --save_probabilities --disable_tta
python -u Scripts/nnunet_softmax_to_h5_probs.py \
    --nnunet-out "$BASE/pred" --pairs-dir "$PAIRS" --prefix NewScan
rm -rf "$BASE/pred" "$BASE/nii"
df -h /data | tail -1

# Sanity check that has cost a full render before: nnU-Net/NIfTI silently swaps
# the near-equal H,W of this scan, and a transposed seg walks a mirrored mask.
python - "$PAIRS" <<'PY'
import glob, os, sys, h5py
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*_Probabilities.h5")))[:2]:
    ct = f.replace("_Probabilities", "")
    with h5py.File(f) as a, h5py.File(ct) as b:
        ks = list(a.keys())
        print(f"  {os.path.basename(f)}: seg {a[ks[0]].shape}  ct {b['image'].shape}")
PY

echo "=== sparse walk z520-779 (slice-frac 0.25) ==="
python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 \
    --ct-dir 01_Mickey_sprockets --seg-dir "$PAIRS" --out-dir "$OUT" \
    --slice-frac 0.25 --n-centers 3 --seam-deg 353.0 \
    --smooth-px 0.75 --snap-max 10 --step 2.5 --coast-max 50 \
    --film-min-thick 7.5 --jump-max 5 --snap-accept 7.5 --max-dr 2.5 \
    --recenter-search 7.5 --recenter-sigma 25 \
    --min-sep 13 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
    --z-heal-px 7.5 --min-coverage 20 \
    --transverse-px 3 --max-infill 3 --radial-robust \
    --seam-exclude-deg 0.3 \
    --inspect-anchors --inspect-max 6

echo "=== walk quality vs z ==="
python -u Scripts/diag_walk_quality.py --npz "$OUT/matched_walks.npz" --zbin 10 || true
echo "=== centre track ==="
python -u Scripts/diag_centers.py --zbin 20 --caches "$OUT/walk_anchors.npz" || true

echo "=== full-slice overlays of the RAW walks ==="
# Raw, not matched: inspect_walk_hires prefers matched_walks.npz when present,
# but that file holds post-resample strip columns, not walk steps.
RAW="$OUT/raw_only"; rm -rf "$RAW"; mkdir -p "$RAW"
ln -s "$OUT/walk_anchors.npz" "$RAW/walk_anchors.npz"
IDX=$(python - "$OUT/walk_anchors.npz" <<'PY'
import sys, numpy as np
z = np.asarray(np.load(sys.argv[1], allow_pickle=True)["z_anchor"], float)
print(",".join(str(int(np.argmin(np.abs(z - t))))
                for t in (525, 560, 600, 640, 680, 720, 760)))
PY
)
echo "overlay anchors: $IDX"
python -u -m unwrapping.inr.inspect_walk_hires \
    --cache-dir "$RAW" --ct-dir 01_Mickey_sprockets --full-only \
    --anchors "$IDX" --out-dir "$OUT/hires"
ls -la "$OUT" "$OUT/hires"
echo "=== KEEPING $PAIRS for the dense re-walk ==="
echo "=== lowband DONE ==="
