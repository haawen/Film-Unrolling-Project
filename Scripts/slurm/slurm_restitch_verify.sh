#!/bin/bash
# Prove the restitch render read the RESTITCHED CT values (Scripts/verify_restitch_read.py):
# v13 --dump-rows on the same 6 rows, once on the restitched volume (with --ct-remap,
# exactly the rsrender flags) and once on the original volume, then an independent
# re-sampling check + location of the rows inside both delivered strips + h5-vs-RA-TIF.
#   sbatch Scripts/slurm/slurm_restitch_verify.sh
#SBATCH --job-name=rsverify
#SBATCH --partition=daily
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/rsverify_%j.out
set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
R="$PROJ/unwrapping/inr/results/newscan"; V="$R/restitch/verify"
ROWS=600,980,1300,1700,1900,2200
FLAGS="--seg-dir $PROJ/newscan_low/pairs --use-cached-anchors --seam-deg 353.0 --ref-mode track \
  --min-sep 13 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
  --z-heal-px 7.5 --min-coverage 300 --no-gap-infill \
  --transverse-px 3 --seam-exclude-deg 0.3 --seam-bridge --enforce-radial-order --dump-rows $ROWS"
for d in dump_new dump_old; do mkdir -p "$V/$d"; ln -sfn "$R/render/walk_anchors.npz" "$V/$d/walk_anchors.npz"; done

echo "=== [1] dump rows: RESTITCHED volume $(date) ==="
python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 --ct-dir 01_Mickey_restitch/h5 \
    --out-dir "$V/dump_new" --ct-remap "$R/diag/restitch/restitch_map.npz" --remap-z 40,2400 \
    $FLAGS 2>&1 | grep -E "DUMP|ct-remap|Error|Traceback|strip|radial-order:"
echo "=== [2] dump rows: ORIGINAL volume $(date) ==="
python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 --ct-dir 01_Mickey_sprockets \
    --out-dir "$V/dump_old" $FLAGS 2>&1 | grep -E "DUMP|Error|Traceback|strip|radial-order:"
echo "=== [3] verify $(date) ==="
python -u Scripts/verify_restitch_read.py \
    --dump-new "$V/dump_new/dump_rows.npz" --dump-old "$V/dump_old/dump_rows.npz" \
    --strip-new "$R/restitch/render/wholeroll.npy" --strip-old "$R/render/wholeroll.npy" \
    --new-dir 01_Mickey_restitch/h5 --old-dir 01_Mickey_sprockets --tif-dir "$V/tif"
echo "=== rsverify DONE $(date) ==="
