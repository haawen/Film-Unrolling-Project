#!/bin/bash
# Render Federica's RESTITCHED tomogram (01_Mickey_restitch/h5) with the CURRENT BEST
# geometry, unchanged: the merged order-fix walk cache in render/ (b0+b1+b2+b3+
# tail_dense) and the exact film_safe/orderfix_tw3 flags. No re-walk, no re-seg.
# Rows stay in the ORIGINAL z frame (--ct-remap / --remap-z 40,2400 = the original's
# 2361 rows), each read from the restitched slice the registration maps it to, with
# the path shifted by t(z) (restitch_map.npz; fit check approved 2026-11-12).
# Then the border-lock chain with the order-fix strip's border + phase reused, so
# the A/B against stabilized_orderfix_tw3 differs in the CT only, and comparisons.
#   sbatch Scripts/slurm/slurm_restitch_render.sh
#SBATCH --job-name=rsrender
#SBATCH --partition=daily
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/rsrender_%j.out
set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
R="$PROJ/unwrapping/inr/results/newscan"
RS="$R/restitch"; OUT="$RS/render"
mkdir -p "$OUT" "$RS/videos" "$RS/compare"
grep -q "radial-order:" "$R/render/render_cmd.log" || { echo "render/ is not the order-fix render"; exit 1; }
ln -sfn "$R/render/walk_anchors.npz" "$OUT/walk_anchors.npz"

echo "=== [1] render restitched CT $(date) ==="
python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 \
    --ct-dir 01_Mickey_restitch/h5 --seg-dir "$PROJ/newscan_low/pairs" \
    --out-dir "$OUT" --use-cached-anchors \
    --ct-remap "$R/diag/restitch/restitch_map.npz" --remap-z 40,2400 \
    --seam-deg 353.0 --ref-mode track \
    --min-sep 13 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
    --z-heal-px 7.5 --min-coverage 300 --no-gap-infill \
    --transverse-px 3 --seam-exclude-deg 0.3 --seam-bridge \
    --enforce-radial-order 2>&1 | tee "$OUT/render_cmd.log"
python - "$OUT/wholeroll.npy" "$R/render/wholeroll.npy" <<'PY'
import sys, numpy as np
a, b = np.load(sys.argv[1], mmap_mode="r"), np.load(sys.argv[2], mmap_mode="r")
print("restitch strip", a.shape, " orderfix strip", b.shape)
assert a.shape == b.shape, "shape differs from the order-fix strip -> geometry not identical"
PY

python -u -m unwrapping.inr.make_film_video \
    "$OUT/wholeroll.npy" "$RS/videos/film_restitch_tw3.mp4" \
    --phase-lock global --invert-phase --reverse --invert --rotate --height 720

echo "=== [2] border-lock chain $(date) ==="
STRIP="$OUT/wholeroll.npy" SFX=_restitch \
PHASE_STRIP="$R/render/wholeroll.npy" BORDER_FROM="$R/border_orderfix_tw3/border.npz" \
    bash Scripts/slurm/slurm_newscan_orderfix_borderlock.sh

echo "=== [3] comparisons $(date) ==="
GT="$PROJ/unwrapping/eval/results/stabilization/gt/gt_perflock_picture.mp4"
OLD="$R/stabilized_orderfix_tw3/FLAGSHIP_borderlock_affine.mp4"
NEW="$R/stabilized_orderfix_restitch/FLAGSHIP_borderlock_affine.mp4"
python -u -m unwrapping.eval.compare_stack --videos "$GT" "$NEW" \
    --labels GT_perflock restitch_affine --height 560 \
    --out "$RS/compare/gt_vs_restitch_affine.mp4"
python -u -m unwrapping.eval.compare_stack --videos "$GT" "$OLD" "$NEW" \
    --labels GT_perflock old_stitch restitch --height 560 \
    --out "$RS/compare/gt_vs_oldstitch_vs_restitch.mp4"
python -u -m unwrapping.eval.border_stillness \
    --videos "$OLD" "$NEW" "$GT" --labels old_stitch restitch GT_perflock \
    --margins 0.15 0.15 0.15 --size 720 720 --out-dir "$RS/compare"
ls -la "$OUT" "$RS/videos" "$RS/compare"
echo "=== rsrender DONE $(date) ==="
