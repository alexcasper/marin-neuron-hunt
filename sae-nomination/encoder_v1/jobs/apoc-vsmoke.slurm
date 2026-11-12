#!/bin/bash
#SBATCH -J vsmoke
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:10:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/vsmoke-%A.out
set -euo pipefail
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
apptainer exec -B /data/scratch/apx186:/data/scratch/apx186 \
  --env PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser "$SIF" \
  python3 /data/scratch/apx186/marin32b/verify_smoke.py
