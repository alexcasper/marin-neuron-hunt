#!/bin/bash
#SBATCH -J nomprobe2
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -a 0-2
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/nomprobe2-%A_%a.out
set -euo pipefail
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
apptainer exec -B /data/scratch/apx186:/data/scratch/apx186 \
  --env PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser \
  --env NOM_LAYER=$SLURM_ARRAY_TASK_ID "$SIF" \
  python3 /data/scratch/apx186/marin32b/nominate_probe.py
