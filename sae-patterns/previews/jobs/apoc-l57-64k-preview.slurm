#!/bin/bash
#SBATCH -J l57-64k-prev
#SBATCH -p gpu
#SBATCH -A pilot_gpu
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH -c 4
#SBATCH --mem=45G
#SBATCH -t 01:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/l57prev64k-%j.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/l57prev64k-%j.err
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
export PYTHONPATH=$SCR
apptainer exec --nv -B /data/scratch/apx186:/data/scratch/apx186 \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$PYTHONPATH" "$SIF" \
  python3 -u $SCR/preview64k.py
