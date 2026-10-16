#!/bin/bash
# FULL-DENSITY re-walk of z1590-1969 with V0_baseline's walk parameters.
#
# WHY: the delivered v0fix render is a merge of b0-b3 (z470-1589, slice-frac 1.0,
# ~1120 contiguous anchors) with newscan_fix/V0_baseline (z1590-1969). But
# V0_baseline was a RANKING run -- slurm_newscan_walkfix.sh deliberately used
# --slice-frac 0.25 to compare four parameter variants cheaply -- so the tail
# carries only 95 anchors for 380 slices. Every z row still renders (v13
# interpolates between anchors), but 3 rows in 4 up there have interpolated
# rather than measured geometry, while everything below z1589 is measured. This
# job removes that inconsistency.
#
# WALK PARAMETERS ARE V0_baseline'S, UNCHANGED. They are also identical to the
# b0-b3 batch's walk parameters (compare slurm_newscan_dense_batch.sh: same
# smooth-px/snap-max/step/coast-max/film-min-thick/jump-max/snap-accept/max-dr/
# recenter-*), which matters: whatever made V0_baseline render clean where the
# b-batch tail tore, it was NOT a walk-threshold difference. The remaining
# candidates are anchor density, the spool centres, and the segmentation. Keeping
# the walk settings fixed here keeps that question answerable.
#
# --min-sep / --dedup-px / --track-merge-px / --min-coverage are NOT passed:
# those act at track/render time, and the merged render re-runs that stage from
# the raw walked paths with its own flags. Only the walk knobs matter here.
#
# --n-centers 8: MATCHES THE DENSITY THE REST OF THE MERGE WAS BUILT AT. The
# standing policy is ~30 find_spool_center calls over the full 2361-slice stack,
# but slurm_newscan_dense_batch.sh actually spends 6 per 300-slice batch = ~47 per
# stack, so 380 slices' matching share is 7.6. Sampling the tail thinner than its
# neighbours would put a slower-varying centre track on one end of the merge and
# a faster one on the other, and the merge's --refit-centers fits ONE line across
# both. Each call costs ~623 s on these 3738x3769 slices, so this is ~83 min of
# the budget. (V0_baseline used --n-centers 1, fine for a ranking run, far too
# thin for a deliverable.)
#
# Centre samples are only trustworthy on PICTURE slices: find_spool_center on a
# perforation slice is what produced b0's cx 1812 outlier against ~1904 for its
# neighbours. z1590-1969 is entirely inside the picture band (the high perf row
# starts ~1990), so every sample here is on good material.
#
# Segmentation is REUSED from newscan_fix/pairs (68 GB, 19 chunks covering
# exactly z1590-1969, written by slurm_segfix.sh). Do not delete it before this
# job finishes. Pure CPU, so plain merlin7 -- no --clusters=gmerlin7 here.
#
#   sbatch Scripts/slurm/slurm_v0dense.sh
#
#SBATCH --job-name=v0dense
#SBATCH --partition=daily
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --output=/data/user/li_k1/M_thesis/logs/v0dense_%j.out

set -euo pipefail
PROJ=/data/user/li_k1/M_thesis
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd "$PROJ"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

PAIRS="$PROJ/newscan_fix/pairs"
OUT="$PROJ/unwrapping/inr/results/newscan_fix/V0_dense"
[ -d "$PAIRS" ] || { echo "missing $PAIRS -- rerun slurm_segfix.sh"; exit 1; }
ls "$PAIRS"/*_Probabilities.h5 | wc -l
rm -rf "$OUT"; mkdir -p "$OUT"

python -u -m unwrapping.inr.unroll_walk_wholeroll_v13 \
    --ct-dir 01_Mickey_sprockets --seg-dir "$PAIRS" --out-dir "$OUT" \
    --slice-frac 1.0 --n-centers 8 --seam-deg 353.0 \
    --smooth-px 0.75 --snap-max 10 --step 2.5 --coast-max 50 \
    --film-min-thick 7.5 --jump-max 5 --snap-accept 7.5 --max-dr 2.5 \
    --recenter-search 7.5 --recenter-sigma 25 \
    --min-sep 13 --dedup-px 12.5 --track-merge-px 15 --match-tol-px 15 \
    --z-heal-px 7.5 --min-coverage 20 \
    --transverse-px 3 --max-infill 3 --radial-robust \
    --seam-exclude-deg 0.3 \
    --inspect-anchors --inspect-max 6

echo "=== walk quality vs z ==="
python -u Scripts/diag_walk_quality.py --npz "$OUT/matched_walks.npz" --zbin 40 || true
ls -la "$OUT"
echo "=== v0dense DONE ==="
