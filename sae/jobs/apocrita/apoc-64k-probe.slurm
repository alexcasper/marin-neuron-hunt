#!/bin/bash
#SBATCH -J marin-64k-probe
#SBATCH -p gpu
#SBATCH -A pilot_gpu
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH -a 0-2%3
#SBATCH -c 4
#SBATCH --mem=45G
#SBATCH -t 08:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/64k-probe-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/64k-probe-%A_%a.err
# env_apoc adaptation of ops_sh apoc-64k-probe.sh: fresh-init TopK 64k (k=32)
# on L57 / L45 / L63 to plateau-detected convergence (6k window, 2% thresh),
# horizon 50k. Val holdout (30k trailing tokens) every 3k; held-out fvu is the
# metric. 64k/bf16 measured 27.4 GiB — fits A100-40. Task order = probe-layers.
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
RUNS=$SCR/64k-probe
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
export PYTHONPATH=$SCR
mkdir -p "$RUNS"

STAGED=$(mktemp -d /tmp/marin_64k_${SLURM_ARRAY_TASK_ID}_XXXXXX)
trap 'rm -rf "$STAGED"' EXIT

ORDER=$SCR/64k-probe-layers.txt
LINE=$((SLURM_ARRAY_TASK_ID + 1))
LL=$(sed -n "${LINE}p" "$ORDER")
[ -n "$LL" ] || { echo "no layer for task $SLURM_ARRAY_TASK_ID"; exit 1; }
OUT=$RUNS/layer=$LL
mkdir -p "$OUT"
echo "node $(hostname -f) 64k probe layer $LL task $SLURM_ARRAY_TASK_ID"

cp "$SCR/atlas-resorted/layer$LL-tokensorted.parquet" "$STAGED/"
cp "$SCR/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet" "$STAGED/"
df -h /tmp | tail -1

apptainer exec --nv -B /data/scratch/apx186:/data/scratch/apx186 \
  -B "$STAGED:$STAGED" \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$PYTHONPATH" "$SIF" \
  python3 -u "$SCR/train_sae.py" \
  --layer-parquet "$STAGED/layer$LL-tokensorted.parquet" \
  --scale-parquet "$STAGED/neuron_scale.parquet" \
  --out-dir "$OUT" \
  --layer-label "L${LL}w64" \
  --n-latents 65536 --k 32 \
  --batch-tokens 1024 --steps 50000 \
  --lr 3e-4 --seed 20261027 \
  --val-rows 1920000 --val-tokens 30000 --val-every 3000 \
  --plateau-window 6000 --plateau-thresh 0.02 \
  --save-every 3000 --keep-grid 9000 \
  --log-every 200

# census on the standard 200,704-token protocol (comparable to sweep tables)
apptainer exec --nv -B /data/scratch/apx186:/data/scratch/apx186 \
  -B "$STAGED:$STAGED" \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$PYTHONPATH" "$SIF" \
  python3 -u "$SCR/sae_stats.py" "$OUT/sae_L${LL}w64_final" \
  "$STAGED/layer$LL-tokensorted.parquet" \
  200704 "$OUT/census" "$STAGED/neuron_scale.parquet" \
  > "$OUT/stats.json" \
  || echo "STATS FAILED layer $LL (training artifacts kept)"

( cd "$OUT" && find . -type f -exec sha256sum {} \; > "$OUT/sha256.txt" )
echo "64K PROBE DONE layer $LL"
