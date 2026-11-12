#!/bin/bash
#SBATCH -J l57-preview
#SBATCH -p gpu
#SBATCH -A pilot_gpu
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH -c 8
#SBATCH --mem=45G
#SBATCH -t 02:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/l57preview-%j.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/l57preview-%j.err
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
export PYTHONPATH=/data/scratch/apx186/marin32b
apptainer exec --nv -B /data/scratch/apx186:/data/scratch/apx186 \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$PYTHONPATH" "$SIF" \
  python3 -u /data/scratch/apx186/marin32b/feature_preview_L57.py
