#!/bin/bash
#SBATCH -J interp12k
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -a 0-7
#SBATCH -c 8
#SBATCH --mem=45G
#SBATCH -t 06:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/interp12k-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/interp12k-%A_%a.err
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
export PYTHONPATH=$SCR
export HF_HUB_ENABLE_HF_TRANSFER=0
LAYERS="12 18 26 33 42 50 55 63"
LL=$(echo $LAYERS | cut -d" " -f $((SLURM_ARRAY_TASK_ID + 1)))
echo "task $SLURM_ARRAY_TASK_ID -> layer $LL on $(hostname -f) [12k topup interp]"
export PREVIEW_LAYER=$LL
apptainer exec -B /data/scratch/apx186:/data/scratch/apx186 \
  -B /data/home/apx186/.cache/huggingface:/data/home/apx186/.cache/huggingface \
  $SIF env PREVIEW_LAYER=$LL PREVIEW_DEV=cpu python3 $SCR/interp_preview_12k.py
echo "TASK $SLURM_ARRAY_TASK_ID DONE layer=$LL"
