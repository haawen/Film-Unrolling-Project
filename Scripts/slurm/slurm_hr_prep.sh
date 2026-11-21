#!/bin/bash
# HIGH-RES Mickey scan (Federica, 2026-11; RA /das/work/p23/p23576/Mickey_Kei_High_Res/tif,
# 2610 x 4205x4263 uint8, bin4 of 1.625 um = 6.5 um voxels; the roll was physically
# RE-ROLLED after the Uni Zurich optical scan => nothing from the previous scans is
# reusable). Relayed RA -> local -> Merlin7 into 01_Mickey_hr/tif.
#   [1] z survey (film area / components / radial extent vs z) on the TIFs
#   [2] raw-CT geometry probe (centre, film radius, layer count, pitch) at 7 z
#   [3] TIF -> HDF5 in place (Mickey_HR_{z:04d}.h5, key image; read-back verified)
#   sbatch Scripts/slurm/slurm_hr_prep.sh
#SBATCH --job-name=hrprep
#SBATCH --partition=daily
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/hrprep_%j.out
set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"; export PYTHONPATH="$PROJ"; export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
D="$PROJ/01_Mickey_hr"; OUT="$PROJ/unwrapping/inr/results/hr/diag"
mkdir -p "$D/h5" "$OUT/zsurvey" "$OUT/geometry"
echo "tifs: $(ls $D/tif | wc -l)"

echo "=== [1/3] z survey $(date) ==="
python -u Scripts/diag_zip_zsurvey.py --dir "$D/tif" --out-dir "$OUT/zsurvey" \
    --step 10 --sub 2 --montage 32

echo "=== [2/3] geometry probe $(date) ==="
python -u Scripts/diag_hr_geometry.py --dir "$D/tif" \
    --z 300,700,1000,1300,1600,1900,2300 --out-dir "$OUT/geometry"

echo "=== [3/3] TIF -> HDF5 in place $(date) ==="
python -u Scripts/restitch_tif_to_h5.py "$D/tif" "$D/h5" 'Mickey_HR_bin4_(\d+)\.tif' Mickey_HR
echo "h5 slices: $(ls $D/h5 | wc -l); tifs left: $(ls $D/tif | wc -l)"
du -sh "$D/h5"
echo "=== hrprep DONE $(date) ==="
