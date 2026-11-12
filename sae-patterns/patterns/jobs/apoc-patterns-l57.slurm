#!/bin/bash
#SBATCH -J pat57
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 08:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/pat57-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/pat57-%A_%a.err
# L57 pattern pass, RAW-ACTS convention (scale_vec=None, matches 64k training).
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
export PYTHONPATH=$SCR
echo "L57 pattern pass (raw-acts) on $(hostname -f)"
apptainer exec -B /data/scratch/apx186:/data/scratch/apx186 \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$PYTHONPATH" "$SIF" \
  env PAT_LAYER=57 PAT_RAW=1 python3 $SCR/patterns_pass.py
echo "L57 PAT DONE"
