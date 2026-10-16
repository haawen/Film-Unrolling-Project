#!/bin/bash
#SBATCH --clusters=gmerlin7
#SBATCH --partition=a100-hourly
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --job-name=udvd_ns
#SBATCH --output=/data/user/li_k1/M_thesis/video_restore/udvd_ns_%j.out
set -e
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd /data/user/li_k1/M_thesis/video_restore/udvd
# UDVD zero-shot (pretrained blind_video_net) sigma 25 on the new-scan flagship video
python udvd_infer.py --input ../newscan_flagship.mp4 --outdir ../udvd_newscan_out --sigmas 5,10
echo "JOB DONE"
ls -la ../udvd_newscan_out
