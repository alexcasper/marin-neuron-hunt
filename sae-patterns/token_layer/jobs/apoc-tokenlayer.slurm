#!/bin/bash
#SBATCH -J toklayer
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -a 0-63
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 03:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/toklayer-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/toklayer-%A_%a.err
# Atlas-direct token-class x layer aggregation. Mapping LL=$SLURM_ARRAY_TASK_ID
# (verified direct, no offset -- same convention as nominate).
# Node storage: accumulators + intermediate npy live in $TMPDIR (/local); only
# the small class_summary parquet is copied back to network scratch.
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
export PYTHONPATH=$SCR
LL=$((SLURM_ARRAY_TASK_ID))
echo "task $SLURM_ARRAY_TASK_ID -> layer $LL on $(hostname -f) [token-class x layer, TMPDIR=$TMPDIR]"
apptainer exec -B /data/scratch/apx186:/data/scratch/apx186 -B "$TMPDIR:/node_tmp" \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$PYTHONPATH" "$SIF" \
  env TCL_LAYER=$LL TMPDIR=/node_tmp python3 $SCR/token_class_x_layer.py
echo "TASK $SLURM_ARRAY_TASK_ID DONE layer=$LL"
