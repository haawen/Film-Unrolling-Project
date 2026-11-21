#!/bin/bash
# Restitch render + --heal-ends outer: the thin vertical lines at the START of the
# video were the OUTERMOST kept winding's unhealed jumps / whole-turn clamps
# (z_heal_windings never healed the end windings; Scripts/diag_end_windings.py,
# fit check diag/end_windings_healends approved 2026-11-16: outer heal good,
# inner tongue left untouched). Everything else = slurm_restitch_render.sh.
#   [1] v13 render (restitched CT, order-fix cache, tw3 flags + --heal-ends outer)
#   [2] band_correct round-1 defaults + 16-bit TIFF
#   [3] video
#   [4] frame_match + lossless per-frame PNG folders (slurm_restitch_framematch.sh)
#   sbatch Scripts/slurm/slurm_restitch_healouter.sh
#SBATCH --job-name=rsheal
#SBATCH --partition=daily
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/rsheal_%j.out
set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
R="$PROJ/unwrapping/inr/results/newscan"
RS="$R/restitch_healouter"; OUT="$RS/render"
mkdir -p "$OUT" "$RS/videos"
ln -sfn "$R/render/walk_anchors.npz" "$OUT/walk_anchors.npz"

echo "=== [1] render restitched CT, --heal-ends outer $(date) ==="
python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 \
    --ct-dir 01_Mickey_restitch/h5 --seg-dir "$PROJ/newscan_low/pairs" \
    --out-dir "$OUT" --use-cached-anchors \
    --ct-remap "$R/diag/restitch/restitch_map.npz" --remap-z 40,2400 \
    --seam-deg 353.0 --ref-mode track \
    --min-sep 13 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
    --z-heal-px 7.5 --min-coverage 300 --no-gap-infill \
    --transverse-px 3 --seam-exclude-deg 0.3 --seam-bridge \
    --enforce-radial-order --heal-ends outer 2>&1 \
    | grep -v -E "per-anchor winding counts" | tee "$OUT/render_cmd.log"
grep -q "heal-ends: kept w33" "$OUT/render_cmd.log" || { echo "heal-ends did not run"; exit 1; }

echo "=== [2] band_correct + TIFF $(date) ==="
python -u -m unwrapping.inr.band_correct "$OUT/wholeroll.npy" "$OUT/wholeroll_bandfix.npy" \
    --z-index "$OUT/z_index.npy" --save-band "$OUT/band_field.npz"
python -u Scripts/export_strip_tiff.py "$OUT/wholeroll_bandfix.npy" \
    "$OUT/wholeroll_restitch_healouter_tw3_bandfix_16bit.tif" \
    --note "RESTITCHED tomogram (01_Mickey_restitch); order-fix tw3 walk via --ct-remap + --heal-ends outer (outermost winding healed: removes the thin vertical lines at the video start); rows in the ORIGINAL new-scan z frame; band_correct round-1 defaults"

echo "=== [3] video $(date) ==="
python -u -m unwrapping.inr.make_film_video \
    "$OUT/wholeroll_bandfix.npy" "$RS/videos/film_restitch_healouter_bandfix.mp4" \
    --phase-lock global --invert-phase --reverse --invert --rotate --height 720

echo "=== [4] frame_match + per-frame PNGs $(date) ==="
STRIP="$OUT/wholeroll_bandfix.npy" O="$RS/frame_match_bandfix" \
    bash Scripts/slurm/slurm_restitch_framematch.sh
echo "=== rsheal DONE $(date) ==="
