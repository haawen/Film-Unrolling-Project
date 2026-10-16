#!/bin/bash
#SBATCH --clusters=gmerlin7
#SBATCH --partition=a100-daily
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=udvd_s
#SBATCH --output=/data/user/li_k1/M_thesis/video_restore/udvd_s_%j.out
set -e
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd /data/user/li_k1/M_thesis/video_restore/udvd

FRAMEDIR=/data/user/li_k1/M_thesis/video_restore/frames_deshaded
OUTDIR=/data/user/li_k1/M_thesis/video_restore/udvd_s_exp

# 1) export frames from the de-shaded stabilized video
python export_frames.py ../deshaded_stabilized.mp4 $FRAMEDIR

# 2) train UDVD-S self-supervised (blind-spot, real noise, no clean target)
python single_train.py --dataset RealVideo --data-path $FRAMEDIR \
  --model blind-video-net-4 --blind-noise \
  --n-frames 5 --image-size 128 --stride 96 --batch-size 8 \
  --lr 5e-5 --num-epochs 12 --valid-interval 2 \
  --output-dir $OUTDIR --no-visual

# 3) infer with the trained (blind) checkpoint
CKPT=$(ls -t $OUTDIR/blind-video-net-4/*/checkpoints/*.pt 2>/dev/null | head -1)
echo "CKPT=$CKPT"
python udvd_infer.py --input ../deshaded_stabilized.mp4 --outdir ../udvd_s_out \
  --model "$CKPT" --blind --outname udvd_s_trained
echo "JOB DONE"
ls -la ../udvd_s_out
