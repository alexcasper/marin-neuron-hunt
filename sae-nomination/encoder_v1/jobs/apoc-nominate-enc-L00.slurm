#!/bin/bash
#SBATCH -J nenc64k-L00
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/nencL00-%A.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/nencL00-%A.err
# L00 rerun (35612529_0 timed out at 4h in wave contention). Same settings as
# apoc-nominate-enc-all61b: 200k tokens, 12h walltime, 64G, 8 cores.
# Submitted --dependency=afterany:35619765_1 so it starts when the staggered
# wave's first task completes and a slot frees (per alex directive).
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
export PYTHONPATH=$SCR
echo "L00 attribution rerun on $(hostname -f)"
apptainer exec -B /data/scratch/apx186:/data/scratch/apx186 \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$PYTHONPATH" "$SIF" \
  env NOM_LAYER=0 NOM_MODE=attribution python3 $SCR/nominate_enc_pass.py
echo "TASK L00 DONE"
