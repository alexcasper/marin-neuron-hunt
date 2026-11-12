#!/bin/bash
#SBATCH -J interp-prev
#SBATCH -p gpu
#SBATCH -A pilot_gpu
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH -a 0-19%8
#SBATCH -c 4
#SBATCH --mem=45G
#SBATCH -t 01:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/interpprev-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/interpprev-%A_%a.err
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
export PYTHONPATH=$SCR
LAYERS="0 1 2 3 4 5 6 7 8 9 10 11 37 38 39 40 41 56 57 58"
LL=$(echo $LAYERS | cut -d" " -f $((SLURM_ARRAY_TASK_ID + 1)))
echo "task $SLURM_ARRAY_TASK_ID -> layer $LL on $(hostname -f)"
apptainer exec --nv -B /data/scratch/apx186:/data/scratch/apx186 \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$PYTHONPATH" \
  --env PREVIEW_LAYER="$LL" "$SIF" \
  python3 -u $SCR/interp_preview.py
