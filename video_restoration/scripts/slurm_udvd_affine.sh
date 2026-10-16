#!/bin/bash
#SBATCH --clusters=gmerlin7
#SBATCH --partition=a100-hourly
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --job-name=udvd_af
#SBATCH --output=/data/user/li_k1/M_thesis/video_restore/udvd_af_%j.out
set -e
source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate nnunet
cd /data/user/li_k1/M_thesis/video_restore/udvd
python udvd_infer.py --input ../newscan_affine.mp4 --outdir ../udvd_affine_out --sigmas 10
echo "JOB DONE"
ls -la ../udvd_affine_out
