#!/bin/bash
#SBATCH -J census-u64
#SBATCH -p gpu
#SBATCH -A pilot_gpu
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH -a 0-63%8
#SBATCH -c 4
#SBATCH --mem=45G
#SBATCH -t 01:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/censusu64-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/censusu64-%A_%a.err
# Uniform-sampling census redo, sae-64k all layers. Convention: writes
# census_uniform.parquet + stats_uniform.json (fvu_atlas_uniform); renames
# census.parquet -> census_prefix.parquet; never overwrites existing uniform.
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
LL=$((SLURM_ARRAY_TASK_ID))
echo "task $SLURM_ARRAY_TASK_ID -> layer $LL on $(hostname -f)"
apptainer exec --nv -B /data/scratch/apx186:/data/scratch/apx186 \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$SCR" "$SIF" \
  python3 -u "$SCR/sae_stats_uniform.py" "$LL"
echo "TASK DONE layer=$LL"
