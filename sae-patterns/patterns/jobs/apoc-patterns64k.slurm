#!/bin/bash
#SBATCH -J pat64k
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -a 0-2
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 08:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/pat64k-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/pat64k-%A_%a.err
# Per-latent pattern pass, sae-64k layers 0-2, CPU-only (sweep owns GPU).
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
export PYTHONPATH=$SCR
LAYERS="0 1 2"
LL=$(echo $LAYERS | cut -d" " -f $((SLURM_ARRAY_TASK_ID + 1)))
echo "task $SLURM_ARRAY_TASK_ID -> layer $LL on $(hostname -f) [64k pattern pass]"
apptainer exec -B /data/scratch/apx186:/data/scratch/apx186 \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$PYTHONPATH" "$SIF" \
  env PAT_LAYER=$LL python3 $SCR/patterns_pass.py
echo "TASK $SLURM_ARRAY_TASK_ID DONE layer=$LL"
