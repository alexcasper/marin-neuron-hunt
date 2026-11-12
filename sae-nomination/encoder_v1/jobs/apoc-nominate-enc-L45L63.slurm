#!/bin/bash
#SBATCH -J nenc64k-2-L45L63
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -a 45,63%2            # run two tasks concurrently (one per node)
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00           # generous walltime (avoid 4h cap)
#SBATCH -o /data/scratch/apx186/marin32b/logs/nenc-L45L63-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/nenc-L45L63-%A_%a.err
# Encoder‑attribution nomination for layers 45 and 63 (previously timed‑out).
# Uses same environment as apoc‑nominate‑enc‑all61b.slurm.
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
export PYTHONPATH=$SCR
LL=$((SLURM_ARRAY_TASK_ID))
echo "task $SLURM_ARRAY_TASK_ID -> layer $LL on $(hostname -f)"
apptainer exec -B /data/scratch/apx186:/data/scratch/apx186 \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$PYTHONPATH" "$SIF" \
  env NOM_LAYER=$LL NOM_MODE=${NOM_MODE:-weights} python3 $SCR/nominate_enc_pass.py

echo "TASK $SLURM_ARRAY_TASK_ID DONE layer=$LL"
