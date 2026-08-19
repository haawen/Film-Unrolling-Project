#!/bin/bash
#SBATCH --clusters=gmerlin7
#SBATCH --partition=a100-hourly
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --job-name=udvd_si
#SBATCH --output=/data/user/li_k1/M_thesis/video_restore/udvd_si_%j.out
set -e
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd /data/user/li_k1/M_thesis/video_restore/udvd
CKPT=$(ls -t /data/user/li_k1/M_thesis/video_restore/udvd_s_exp/blind-video-net-4/*/checkpoints/checkpoint_best.pt | head -1)
echo "CKPT=$CKPT"
python udvd_infer.py --input ../deshaded_stabilized.mp4 --outdir ../udvd_s_out \
  --model "$CKPT" --blind --outname udvd_s_trained
echo "JOB DONE"
ls -la ../udvd_s_out
