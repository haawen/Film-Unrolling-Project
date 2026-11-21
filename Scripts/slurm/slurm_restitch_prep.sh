#!/bin/bash
# Restitched tomogram (Federica, 2026-11; RA /das/work/p23/p23576/Mickey_merged_norm/tif,
# pushed to 01_Mickey_restitch/tif): convert -> register to the original new-scan
# volume -> fit check of the reused walk geometry. NO render here (fit check first).
#   sbatch Scripts/slurm/slurm_restitch_prep.sh
#SBATCH --job-name=restitch
#SBATCH --partition=daily
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/restitch_%j.out
set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
D="$PROJ/01_Mickey_restitch"; R="$PROJ/unwrapping/inr/results/newscan"; OUT="$R/diag/restitch"
mkdir -p "$D/h5" "$OUT"

echo "=== [1/3] convert TIF -> HDF5 (in place) $(date) ==="
python -u Scripts/restitch_tif_to_h5.py "$D/tif" "$D/h5"
echo "h5 slices: $(ls $D/h5 | wc -l)"

echo "=== [2/3] register restitched -> original $(date) ==="
python -u Scripts/register_restitch.py --old 01_Mickey_sprockets --new "$D/h5" \
    --out "$OUT/restitch_map.npz"

echo "=== [3/3] fit check of the reused geometry $(date) ==="
python -u Scripts/check_restitch_fit.py --matched "$R/diag/order_fix3/matched_walks.npz" \
    --remap "$OUT/restitch_map.npz" --old 01_Mickey_sprockets --new "$D/h5" \
    --out-dir "$OUT/fit"
ls -la "$OUT" "$OUT/fit" | head -60; echo "=== DONE $(date) ==="
