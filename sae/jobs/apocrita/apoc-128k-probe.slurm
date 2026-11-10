#!/bin/bash
#SBATCH -J marin-128k-probe
#SBATCH -p gpu
#SBATCH -A pilot_gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH -a 0-2%3
#SBATCH -c 4
#SBATCH --mem=60G
#SBATCH -t 10-00:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/128k-probe-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/128k-probe-%A_%a.err
# env_apoc adaptation of ops_sh apoc-128k-probe.sh: fresh-init TopK 128k
# (k=32) on L57/L45/L63, same val/plateau protocol as the 64k probe
# (val 30k holdout every 3k, plateau 6k-window/2%, horizon 50k, 3k grid,
# keep-grid 9000). A100-80 (access + CUDA verified by smoke 35422919).
# FIX vs 64k probe: census targets the val-suffixed final via glob.
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
RUNS=$SCR/128k-probe
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
export PYTHONPATH=$SCR
mkdir -p "$RUNS"

STAGED=$(mktemp -d /tmp/marin_128k_${SLURM_ARRAY_TASK_ID}_XXXXXX)
trap 'rm -rf "$STAGED"' EXIT

ORDER=$SCR/128k-probe-layers.txt
LINE=$((SLURM_ARRAY_TASK_ID + 1))
LL=$(sed -n "${LINE}p" "$ORDER")
[ -n "$LL" ] || { echo "no layer for task $SLURM_ARRAY_TASK_ID"; exit 1; }
OUT=$RUNS/layer=$LL
mkdir -p "$OUT"
echo "node $(hostname -f) 128k probe layer $LL task $SLURM_ARRAY_TASK_ID"

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
  --layer-label "L${LL}w128" \
  --n-latents 131072 --k 32 \
  --batch-tokens 1024 --steps 50000 \
  --lr 3e-4 --seed 20261027 \
  --val-rows 1920000 --val-tokens 30000 --val-every 3000 \
  --plateau-window 6000 --plateau-thresh 0.02 \
  --save-every 3000 --keep-grid 9000 \
  --log-every 200

# census on the plateau/val-suffixed final (glob — the 64k probe bug fix)
CK=$(ls -d "$OUT"/sae_L${LL}w128_final_s* "$OUT"/sae_L${LL}w128_plateau_* 2>/dev/null | sort | tail -1)
[ -n "$CK" ] || { echo "no final ckpt found for $LL"; exit 1; }
apptainer exec --nv -B /data/scratch/apx186:/data/scratch/apx186 \
  -B "$STAGED:$STAGED" \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$PYTHONPATH" "$SIF" \
  python3 -u "$SCR/sae_stats.py" "$CK" "$STAGED/layer$LL-tokensorted.parquet" \
  200704 "$OUT/census" "$STAGED/neuron_scale.parquet" \
  > "$OUT/stats.json" 2> "$OUT/stats.err" \
  && echo "CENSUS OK $LL ($CK)" || echo "STATS FAILED layer $LL (kept; see stats.err)"

( cd "$OUT" && find . -type f -exec sha256sum {} \; > "$OUT/sha256.txt" )
echo "128K PROBE DONE layer $LL"
