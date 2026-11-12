#!/bin/bash
#SBATCH -J pat64k-all
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -a 0-63
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 08:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/pat64kall-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/pat64kall-%A_%a.err
# Per-latent pattern pass, sae-64k ALL 64 layers, CPU-only.
# Restage of apoc-patterns64k.slurm (which hardcoded layers 0-2 via its
# LAYERS var). Mapping LL=$SLURM_ARRAY_TASK_ID — direct, no offset, same
# convention as nominate/tokenlayer (dirs are layer=00..63).
# PAT_RAW=1: raw-acts convention, matches 64k sweep training (alex 2026-11-08).
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
export PYTHONPATH=$SCR
LL=$((SLURM_ARRAY_TASK_ID))
echo "task $SLURM_ARRAY_TASK_ID -> layer $LL on $(hostname -f) [64k pattern pass ALL64 PAT_RAW=1]"
apptainer exec -B /data/scratch/apx186:/data/scratch/apx186 \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$PYTHONPATH" "$SIF" \
  env PAT_LAYER=$LL PAT_RAW=1 python3 $SCR/patterns_pass.py
echo "TASK $SLURM_ARRAY_TASK_ID DONE layer=$LL"
