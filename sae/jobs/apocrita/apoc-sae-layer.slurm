#!/bin/bash
#SBATCH -J marin-sae
#SBATCH -p gpu
#SBATCH -A pilot_gpu
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH -a 0-63%8
#SBATCH -c 4
#SBATCH --mem=45G
#SBATCH -t 04:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/sae-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/sae-%A_%a.err
# env_apoc adaptation of ops_sh apoc-sae-layer.sh (refresh 2026-10-28):
# outputs -> SCRATCH sae-32k/layer=NN (durable promotion only if keeper);
# adds sae_stats.py census per layer. SMOKE PASSED: fvu 0.969->0.888, 1224 tok/s.
set -euo pipefail
ATLAS=/data/scratch/apx186/marin32b/atlas
RESORT=/data/scratch/apx186/marin32b/atlas-resorted
RUNS=/data/scratch/apx186/marin32b/sae-32k
TMPD=/data/scratch/apx186/marin32b/tmp
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
LOCAL="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
mkdir -p "$RUNS" "$TMPD"

LL=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
LDIR="$RUNS/layer=$LL"
mkdir -p "$LDIR"
echo "node $(hostname -f) layer $LL task $SLURM_ARRAY_TASK_ID local=$LOCAL"

STEPS=${STEPS:-6000}
if [ "${SMOKE:-0}" = "1" ]; then STEPS=200; fi

# ---- stage node-local slice (scratch resort copy -> node-local) ----
mkdir -p "$LOCAL/marin$LL"
cp "$RESORT/layer$LL-tokensorted.parquet" "$LOCAL/marin$LL/" \
  || { cp "/data/scratch/apx186/marin32b/layer$LL-tokensorted.parquet" "$LOCAL/marin$LL/"; }
cp "$ATLAS/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet" "$LOCAL/marin$LL/"
ls -la "$LOCAL/marin$LL"
df -h "$LOCAL" | tail -1

# ---- train (in SIF; eai-sparsify preinstalled in qwen35 userbase; bf16 params) ----
apptainer exec --nv -B /data/scratch/apx186:/data/scratch/apx186 \
  -B "$LOCAL/marin$LL:$LOCAL/marin$LL" \
  --env PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser "$SIF" \
  python3 -u /data/scratch/apx186/marin32b/train_sae.py \
  --layer-parquet "$LOCAL/marin$LL/layer$LL-tokensorted.parquet" \
  --scale-parquet "$LOCAL/marin$LL/neuron_scale.parquet" \
  --out-dir "$LDIR" \
  --layer-label "L$LL" \
  --n-latents 32768 --k 32 \
  --batch-tokens 1024 --steps "$STEPS" \
  --log-every 50 --save-every 1000

# ---- per-layer stats + census (portable, same SIF/userbase) ----
apptainer exec --nv -B /data/scratch/apx186:/data/scratch/apx186 \
  -B "$LOCAL/marin$LL:$LOCAL/marin$LL" \
  --env PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser "$SIF" \
  python3 -u /data/scratch/apx186/marin32b/sae_stats.py \
  "$LDIR/sae_L${LL}_final" "$LOCAL/marin$LL/layer$LL-tokensorted.parquet" \
  200000 "$LDIR/census" "$LOCAL/marin$LL/neuron_scale.parquet" \
  || echo "STATS FAILED layer $LL (training artifacts kept)"

# ---- manifest outputs ----
( cd "$LDIR" && find . -type f -exec sha256sum {} \; > "$LDIR/sha256.txt" )
rm -rf "$LOCAL/marin$LL"
echo "TASK DONE layer $LL"
