#!/bin/bash
#SBATCH -J nenc64k
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -a 1,3-56,58-63
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/nencall-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/nencall-%A_%a.err
# Encoder-attribution nomination, sae-64k, remaining 61 layers
# (0,2 in flight as 35612529; 57 done as 35608993). LL=$SLURM_ARRAY_TASK_ID
# direct, no offset — same convention as all marin32b arrays.
# Storage: attribution work is in-RAM (accumulators ~2GB); only the small
# attribution parquet/json mirror to $SCR — no heavy intermediates, nothing
# to wire to TMPDIR.
# Submit with: --export=NOM_MODE=attribution
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
export PYTHONPATH=$SCR
LL=$((SLURM_ARRAY_TASK_ID))
echo "task $SLURM_ARRAY_TASK_ID -> layer $LL mode=${NOM_MODE:-weights} on $(hostname -f) [enc nominate all61]"
apptainer exec -B /data/scratch/apx186:/data/scratch/apx186 \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$PYTHONPATH" "$SIF" \
  env NOM_LAYER=$LL NOM_MODE=${NOM_MODE:-weights} python3 $SCR/nominate_enc_pass.py
echo "TASK $SLURM_ARRAY_TASK_ID DONE layer=$LL"
