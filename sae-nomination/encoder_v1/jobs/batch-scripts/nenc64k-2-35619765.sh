#!/bin/bash
#SBATCH -J nenc64k-2
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -a 1,3-56,58-63%15
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/nencall2-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/nencall2-%A_%a.err
# Encoder-attribution nomination, remaining 61 layers — RESUBMIT after wave 1
# (35614630) hit the 4h walltime cap on all 61 tasks (CPU contention from 2-3
# tasks/node on 8 threads each; solo L57 reference = 2h08).
# Changes vs apoc-nominate-enc-all61.slurm:
#   %15 array throttle (max 15 concurrent => <=2 tasks/node => ~2x contention,
#       worst case ~4.5h, 12h walltime = ~2.6x headroom)
#   --time=12:00:00 (compute MaxTime is 10d, no cap issue)
# NOM_TOKENS unchanged at 200000 (methodology held per decision; option (b)
# deliberately NOT taken). LL=$SLURM_ARRAY_TASK_ID direct, no offset.
# Layers 0,2 in flight separately (35612529_0 rerun needed - also timed out);
# 57 done (35608993).
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
export PYTHONPATH=$SCR
LL=$((SLURM_ARRAY_TASK_ID))
echo "task $SLURM_ARRAY_TASK_ID -> layer $LL mode=${NOM_MODE:-weights} on $(hostname -f) [enc nominate all61 resubmit]"
apptainer exec -B /data/scratch/apx186:/data/scratch/apx186 \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$PYTHONPATH" "$SIF" \
  env NOM_LAYER=$LL NOM_MODE=${NOM_MODE:-weights} python3 $SCR/nominate_enc_pass.py
echo "TASK $SLURM_ARRAY_TASK_ID DONE layer=$LL"
