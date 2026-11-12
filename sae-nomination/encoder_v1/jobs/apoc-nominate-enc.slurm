#!/bin/bash
#SBATCH -J nenc64k
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -a 0-63
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/nenc64k-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/nenc64k-%A_%a.err
# Encoder-side neuron nomination, sae-64k, CPU-only.
# Mode via env: NOM_MODE=weights (cheap, any layer) | attribution (data pass,
# 3 structured layers first: 0, 2, 57).
# Mapping is LL=$SLURM_ARRAY_TASK_ID (verified direct, no offset).
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
export PYTHONPATH=$SCR
LL=$((SLURM_ARRAY_TASK_ID))
echo "task $SLURM_ARRAY_TASK_ID -> layer $LL mode=${NOM_MODE:-weights} on $(hostname -f) [enc nominate]"
apptainer exec -B /data/scratch/apx186:/data/scratch/apx186 \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$PYTHONPATH" "$SIF" \
  env NOM_LAYER=$LL NOM_MODE=${NOM_MODE:-weights} python3 $SCR/nominate_enc_pass.py
echo "TASK $SLURM_ARRAY_TASK_ID DONE layer=$LL"
