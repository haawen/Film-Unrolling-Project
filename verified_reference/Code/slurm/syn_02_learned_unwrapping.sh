#!/bin/bash
# Learned unwrapping on the synthetic benchmark. An implicit neural field
# f(u,z) -> (x,y) = analytical spiral + learned residual is fitted per volume; then
# the strip is rendered and compared with the exact GT strip and the oracle strip.
#
#   sbatch Code/slurm/syn_02_learned_unwrapping.sh R1S hq_video_4k_outer_imperfect   # supervised on the GT u-map
#   sbatch Code/slurm/syn_02_learned_unwrapping.sh R1n hq_video_4k_outer_imperfect   # self-supervised
#   sbatch Code/slurm/syn_02_learned_unwrapping.sh G1  hq_video_4k_outer_imperfect   # R1S with a multi-resolution grid encoder
#   CKPT=Results/3_synthetic_learned_unwrapping/<run>/train/model_final.pt \
#       sbatch Code/slurm/syn_02_learned_unwrapping.sh R1S <preset>                  # evaluate a shipped checkpoint only
#
# R1S / R1n: Fourier features sigma=20 + MLP 256x4, attachment (distance transform
# to the emulsion) + conformal loss (w=0.1, ramped over 2000 steps), lr 1e-4, 10k steps.
# G1: grid encoder (8 levels), encoder lr x100, 2000 steps, ramp 500.
# Metrics: row correlation and SSIM of the rendered strip vs the GT strip, SSIM vs oracle.
# Output: $WORK/learned/<variant>_<preset>/{train,eval}
#
#SBATCH --clusters=gmerlin7
#SBATCH --partition=a100-daily
#SBATCH --gres=gpu:1
#SBATCH --job-name=learned_unwrap
#SBATCH --time=23:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=160G
#SBATCH --output=work/logs/%x_%j.out
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/Code/slurm/_activate.sh"
VAR=${1:?usage: syn_02_learned_unwrapping.sh R1S|R1n|G1 <preset>}
PRESET=${2:?preset}
SYN="$WORK/synthetic/$PRESET"
OUT="$WORK/learned/${VAR}_${PRESET}"; mkdir -p "$OUT"
test -f "$SYN/oracle_strip.npz" || { echo "generate $PRESET first (syn_01_generate.sh)"; exit 1; }

COMMON="--data-dir $SYN --attachment emulsion --centerline-erode 1 --gt-npz $SYN/ground_truth.npz \
        --batch-size 131072 --lr 1e-4 \
        --w-attach 1.0 --w-conformal 0.1 --w-res-reg 0.0 --log-every 100"
case "$VAR" in
  R1S) ARCH="--model-type inr --sigma 20.0 --hidden-dim 256 --n-layers-mlp 4 \
             --steps 10000 --ramp-steps 2000 --supervised-steps 10000 --ckpt-every 2000" ;;
  R1n) ARCH="--model-type inr --sigma 20.0 --hidden-dim 256 --n-layers-mlp 4 \
             --steps 10000 --ramp-steps 2000 --supervised-steps 0 --ckpt-every 2000" ;;
  G1)  ARCH="--model-type grid --grid-n-levels 8 --grid-features 2 --grid-base-u 64 --grid-base-z 16 \
             --grid-finest-u 8192 --grid-finest-z 256 --grid-hidden 64 --grid-mlp-layers 2 \
             --encoder-lr-mult 100.0 --steps 2000 --ramp-steps 500 --supervised-steps 2000 --ckpt-every 1000" ;;
  *) echo "unknown variant $VAR"; exit 1 ;;
esac
CK=${CKPT:-$OUT/train/model_final.pt}
[ -n "${CKPT:-}" ] || python -u -m virtual_unrolling.learned_unwrapping.surface_train $COMMON $ARCH --out-dir "$OUT/train"
python -u -m virtual_unrolling.learned_unwrapping.surface_eval --data-dir "$SYN" --out-dir "$OUT/eval" \
    --attachment emulsion --centerline-erode 1 --ckpt "$CK" \
    --gt-npz "$SYN/ground_truth.npz" --oracle-npz "$SYN/oracle_strip.npz"
echo "=== done: $OUT/eval ==="
