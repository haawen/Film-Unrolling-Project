#!/bin/bash
# Band-correction variants on the order-fix tw3 strip, each rendered with the SAME
# framing as the uncorrected strip (border_render --phase-strip + the uncorrected
# border.npz), so the m02 videos differ in brightness only. Strips are deleted
# after rendering (2.4 GB each); band fields and videos are kept.
#   sbatch Scripts/slurm/slurm_newscan_bandsweep.sh
#SBATCH --job-name=nsbandsw
#SBATCH --partition=daily
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/nsbandsw_%j.out
set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
R="$PROJ/unwrapping/inr/results/newscan"; SRC="$R/render/wholeroll.npy"
O="$R/diag/bandsweep"; mkdir -p "$O"
grep -q "radial-order:" "$R/render/render_cmd.log" || { echo "render/ is not the order-fix strip"; exit 1; }
run() {   # tag, band_correct args...
    tag=$1; shift
    echo "=== $tag: $* $(date) ==="
    python -u -m unwrapping.inr.band_correct "$SRC" "$O/strip_$tag.npy" \
        --z-index "$R/render/z_index.npy" --z-lo 515 --z-hi 1945 \
        --save-band "$O/band_$tag.npz" "$@"
    python -u -m unwrapping.eval.border_render "$O/strip_$tag.npy" \
        --border "$R/border_orderfix_tw3/border.npz" --out-dir "$O" --along phaselock \
        --smooth-frames 0.8 --reverse --invert --height 720 \
        --along-shift 0.156 --across-shift 0.0 --across-scale 1.0 \
        --phase-strip "$SRC" --margin 0.02 --video "m02_$tag.mp4" | grep -i "pitch\|rendered"
    rm -f "$O/strip_$tag.npy"
}
run mean150   --agg mean   --sz-keep 150
run mean300   --agg mean   --sz-keep 300
run med150    --agg median --med-win 6000 --sz-keep 150
run med300    --agg median --med-win 6000 --sz-keep 300
ls -la "$O"; echo "=== DONE $(date) ==="
