#!/bin/bash
#SBATCH --clusters=gmerlin7
#SBATCH --partition=a100-hourly
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --job-name=udvd_zs
#SBATCH --output=/data/user/li_k1/M_thesis/video_restore/udvd_zs_%j.out
set -e
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd /data/user/li_k1/M_thesis/video_restore/udvd
python udvd_infer.py --input ../deshaded_stabilized.mp4 --outdir ../udvd_zs_out --sigmas 15,25,40
echo "JOB DONE"
ls -la ../udvd_zs_out
