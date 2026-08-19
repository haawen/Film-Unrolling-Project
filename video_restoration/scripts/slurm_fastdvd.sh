#!/bin/bash
#SBATCH --clusters=gmerlin7
#SBATCH --partition=a100-hourly
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --job-name=fastdvd
#SBATCH --output=/data/user/li_k1/M_thesis/video_restore/fastdvd_%j.out
set -e
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd /data/user/li_k1/M_thesis/video_restore/fastdvdnet
python infer_real.py --input ../deshaded_stabilized.mp4 --outdir ../fastdvd_out --sigmas 12,20,30
echo "JOB DONE"
ls -la ../fastdvd_out
