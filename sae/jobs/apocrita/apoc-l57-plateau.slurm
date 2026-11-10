#!/bin/bash
#SBATCH -J marin-l57-plateau
#SBATCH -p gpu
#SBATCH -A pilot_gpu
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH -c 4
#SBATCH --mem=45G
#SBATCH -t 08:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/l57-plateau-%j.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/l57-plateau-%j.err
# env_apoc adaptation of ops_sh apoc-l57-plateau.sh: L57 resume from 6k final,
# horizon 200k absolute, val holdout trailing 30k tokens, VAL every 3k (both
# train+val fvu -> val_log.jsonl), ckpts every 3k with 9k keep-grid + latest,
# plateau = <2% rel val-fvu improvement over 9k window -> save + stop.
# CHAIN on TIME LIMIT: resubmit with INIT_FROM=<latest kept ckpt> and
# START_STEP=<its step> (read here from env).
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
LDIR=$SCR/sae-32k/layer=57
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
mkdir -p "$LDIR"

# node-local staging (8h run; scratch copy is the source)
STAGED=$(mktemp -d /tmp/marin_l57_XXXXXX)
trap 'rm -rf "$STAGED"' EXIT
cp "$SCR/atlas-resorted/layer57-tokensorted.parquet" "$STAGED/"
cp "$SCR/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet" "$STAGED/"
echo "=== L57 plateau run start $(date -Is) node $(hostname -f) INIT_FROM=${INIT_FROM:-default} START_STEP=${START_STEP:-6000}"

apptainer exec --nv -B /data/scratch/apx186:/data/scratch/apx186 \
  -B "$STAGED:$STAGED" \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" "$SIF" \
  python3 -u "$SCR/train_sae.py" \
  --layer-parquet "$STAGED/layer57-tokensorted.parquet" \
  --scale-parquet "$STAGED/neuron_scale.parquet" \
  --out-dir "$LDIR" \
  --layer-label "L57" \
  --n-latents 32768 --k 32 \
  --batch-tokens 1024 \
  --steps 200000 --start-step "${START_STEP:-6000}" \
  --lr 1e-4 \
  --init-from "${INIT_FROM:-$LDIR/sae_L57_final}" \
  --val-rows 1920000 --val-tokens 30000 --val-every 3000 \
  --plateau-window 9000 --plateau-thresh 0.02 \
  --save-every 3000 --keep-grid 9000 \
  --log-every 200

echo "=== L57 plateau run end $(date -Is) ==="
tail -5 "$LDIR/val_log.jsonl" 2>/dev/null || true
