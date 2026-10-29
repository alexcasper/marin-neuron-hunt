#!/bin/bash
#SBATCH -J marin-l57-plateau
#SBATCH -p gpu
#SBATCH --gres gpu:nvidia_a100-pcie-40gb:1
#SBATCH -t 08:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/l57-plateau-%j.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/l57-plateau-%j.err
# L57 PLATEAU RUN: resume from the 6k final, horizon 200k absolute steps.
# Val holdout (30k tokens, trailing) evaluated every 3k steps; plateau =
# <2% relative val-fvu improvement across a 9k-step window -> save + stop.
# Checkpoints every 3k; retention keeps the 9k grid (every 3rd) + latest.
# NOTE: gpu partition caps at 8h. If the job hits TIME LIMIT before plateau,
# resubmit with --init-from <latest kept ckpt> --start-step <its step> (env_apoc chains).
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
LDIR=$SCR/sae-32k/layer=57
mkdir -p "$LDIR"

source "${SAE_VENV:-$SCR/sae-venv/bin/activate}"
cd "$SCR"

# fresh logs per submission
LOGPFX=$SCR/logs/l57-plateau-run
echo "=== L57 plateau run start $(date -Is) ==="

python train_sae.py \
  --layer-parquet "$SCR/atlas-resorted/layer57-tokensorted.parquet" \
  --scale-parquet "$SCR/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet" \
  --out-dir "$LDIR" \
  --layer-label "L57" \
  --n-latents 32768 --k 32 \
  --batch-tokens 1024 \
  --steps 200000 --start-step ${START_STEP:-6000} \
  --lr 1e-4 \
  --init-from "${INIT_FROM:-$LDIR/sae_L57_final}" \
  --val-rows 1920000 --val-tokens 30000 --val-every 3000 \
  --plateau-window 9000 --plateau-thresh 0.02 \
  --save-every 3000 --keep-grid 9000 \
  --log-every 200

echo "=== L57 plateau run end $(date -Is) ==="
tail -5 "$LDIR/val_log.jsonl" 2>/dev/null || true
