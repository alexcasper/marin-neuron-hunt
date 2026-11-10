#!/bin/bash
#SBATCH -J marin-64k-sweep
#SBATCH -p gpu
#SBATCH -A pilot_gpu
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH -a 0-63%8
#SBATCH -c 4
#SBATCH --mem=45G
#SBATCH -t 08:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/64k-sweep-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/64k-sweep-%A_%a.err
# env_apoc adaptation of ops_sh apoc-64k-sweep.sh: FULL 64-layer 64k sweep.
# Fresh init, TopK 64k k=32 bf16, plateau detector (6k window / 2%), horizon
# 50k, val holdout 30k trailing every 3k. CHECKPOINT DISCIPLINE: keep-latest-
# only via keep-grid 999999999; finals land sae-64k/layer=NN/ (~6.8G/layer).
# Census targets the val-suffixed final via glob (probe-bug fix).
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
RUNS=$SCR/sae-64k
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
export PYTHONPATH=$SCR
mkdir -p "$RUNS"

STAGED=$(mktemp -d /tmp/marin_64ksw_${SLURM_ARRAY_TASK_ID}_XXXXXX)
trap 'rm -rf "$STAGED"' EXIT

LL=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
OUT=$RUNS/layer=$LL
mkdir -p "$OUT"
echo "node $(hostname -f) 64k sweep layer $LL task $SLURM_ARRAY_TASK_ID"

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
  --layer "${LL}" \
  --n-latents 65536 --k 32 \
  --batch-tokens 1024 --steps 50000 \
  --lr 3e-4 --seed 20261027 \
  --val-rows 1920000 --val-tokens 30000 --val-every 3000 \
  --plateau-window 6000 --plateau-thresh 0.02 \
  --save-every 3000 --keep-grid 999999999 \
  --log-every 200

CK=$(ls -d "$OUT"/sae_L${LL}w64_final_s* "$OUT"/sae_L${LL}w64_plateau_* 2>/dev/null | sort | tail -1)
[ -n "$CK" ] || { echo "no final ckpt found for $LL"; exit 1; }
apptainer exec --nv -B /data/scratch/apx186:/data/scratch/apx186 \
  -B "$STAGED:$STAGED" \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$PYTHONPATH" "$SIF" \
  python3 -u "$SCR/sae_stats.py" "$CK" "$STAGED/layer$LL-tokensorted.parquet" \
  200704 "$OUT/census" "$STAGED/neuron_scale.parquet" \
  > "$OUT/stats.json" 2> "$OUT/stats.err" \
  && echo "CENSUS OK $LL ($CK)" || echo "STATS FAILED layer $LL (kept; see stats.err)"

( cd "$OUT" && find . -type f -exec sha256sum {} \; > "$OUT/sha256.txt" )
echo "64K SWEEP DONE layer $LL"
