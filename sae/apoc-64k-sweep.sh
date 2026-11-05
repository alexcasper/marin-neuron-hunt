#!/bin/bash
#SBATCH -J marin-64k-sweep
#SBATCH -p gpu
#SBATCH --gres gpu:nvidia_a100-pcie-40gb:1
#SBATCH -a 0-63%8
#SBATCH -t 08:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/64k-sweep-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/64k-sweep-%A_%a.err
# FULL 64-LAYER 64k SWEEP (alex GO): fresh init, TopK 64k k=32 bf16, plateau
# detector (6k window / 2%), horizon 50k, val holdout (30k trailing tokens)
# evaluated every 3k steps. CHECKPOINT DISCIPLINE: 3k saves are transient —
# keep-latest-only via keep-grid 999999999 (all older step-ckpts culled on each
# new save); final lands as sae_LNNw64_final. Layers 00-63.
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
RUNS=$SCR/sae-64k
TMPD=$SCR/tmp
mkdir -p "$RUNS" "$TMPD"

LL=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
OUT=$RUNS/layer=$LL
mkdir -p "$OUT"
echo "node $(hostname -f) 64k sweep layer $LL task $SLURM_ARRAY_TASK_ID"

cp "$SCR/atlas-resorted/layer$LL-tokensorted.parquet" "$SLURM_TMPDIR/"
cp "$SCR/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet" "$SLURM_TMPDIR/"

source "${SAE_VENV:-$SCR/sae-venv/bin/activate}"
cd "$SCR"

python train_sae.py \
  --layer-parquet "$SLURM_TMPDIR/layer$LL-tokensorted.parquet" \
  --scale-parquet "$SLURM_TMPDIR/neuron_scale.parquet" \
  --out-dir "$OUT" \
  --layer-label "L${LL}w64" \
  --n-latents 65536 --k 32 \
  --batch-tokens 1024 --steps 50000 \
  --lr 3e-4 --seed 20261027 \
  --val-rows 1920000 --val-tokens 30000 --val-every 3000 \
  --plateau-window 6000 --plateau-thresh 0.02 \
  --save-every 3000 --keep-grid 999999999 \
  --log-every 200

python sae_stats.py "$OUT/sae_L${LL}w64_final" "$SLURM_TMPDIR/layer$LL-tokensorted.parquet" \
  200704 "$OUT/census" "$SLURM_TMPDIR/neuron_scale.parquet" \
  | tee "$OUT/stats.json" \
  || echo "STATS FAILED layer $LL (training artifacts kept)"

( cd "$OUT" && find . -type f -exec sha256sum {} \; > "$OUT/sha256.txt" )
echo "64K SWEEP DONE layer $LL"
