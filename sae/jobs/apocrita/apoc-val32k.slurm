#!/bin/bash
#SBATCH -J val32k
#SBATCH -p gpu
#SBATCH -A pilot_gpu
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH -c 4
#SBATCH --mem=45G
#SBATCH -t 01:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/val32k-%j.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/val32k-%j.err
# Honest-table pre-job: 32k step12000 ckpts of L45/L63 measured on the same
# 30k trailing holdout the 64k probe uses (census-sample fvu is train-biased).
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
export PYTHONPATH=$SCR
for L in 45 63; do
  CK=$SCR/sae-32k/layer=$L/sae_L${L}_step12000
  [ -d "$CK" ] || { echo "MISSING $CK"; continue; }
  apptainer exec --nv -B /data/scratch/apx186:/data/scratch/apx186 \
    --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$PYTHONPATH" "$SIF" \
    python3 -u "$SCR/val32k.py" "$L" "$CK"
done
echo "VAL32K DONE"
