#!/bin/bash
#SBATCH -J nom64k
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -a 0-63
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/nom64k-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/nom64k-%A_%a.err
# Decode-side neuron nomination, sae-64k, all 64 layers, CPU-only.
# Prototype first: submit with --array=56 (L57) to smoke, then full array.
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
export PYTHONPATH=$SCR
LL=$((SLURM_ARRAY_TASK_ID))
echo "task $SLURM_ARRAY_TASK_ID -> layer $LL on $(hostname -f) [64k nominate pass]"
apptainer exec -B /data/scratch/apx186:/data/scratch/apx186 \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$PYTHONPATH" "$SIF" \
  env NOM_LAYER=$LL python3 $SCR/nominate_pass.py
echo "TASK $SLURM_ARRAY_TASK_ID DONE layer=$LL"
