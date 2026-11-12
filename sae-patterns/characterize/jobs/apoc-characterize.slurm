#!/bin/bash
#SBATCH -J char57
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/char57-%A.out
set -euo pipefail
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
apptainer exec -B /data/scratch/apx186:/data/scratch/apx186 \
  --env PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser "$SIF" \
  python3 /data/scratch/apx186/marin32b/characterize_nominees.py
