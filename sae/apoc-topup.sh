#!/bin/bash
#SBATCH -J marin-topup
#SBATCH -p gpu
#SBATCH --gres gpu:nvidia_a100-pcie-40gb:1
#SBATCH -a 0-43%8
#SBATCH -t 02:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/topup-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/topup-%A_%a.err
# Topup: resume each layer's 6k final, +6k steps at lowered LR (1e-4), to
# 12k ABSOLUTE. Retention: checkpoints ONLY at step 9000 and 12000 (final).
# Task order = topup-order.txt (worst fvu first).
set -euo pipefail
DUR=/data/SPCS-IOT/marin-32b
SCR=/data/scratch/apx186/marin32b
RUNS=$SCR/sae-32k
TMPD=$SCR/tmp
mkdir -p "$RUNS" "$TMPD"

ORDER=$SCR/topup-order.txt
LINE=$((SLURM_ARRAY_TASK_ID + 1))
LL=$(sed -n "${LINE}p" "$ORDER")
[ -n "$LL" ] || { echo "no layer for task $SLURM_ARRAY_TASK_ID"; exit 1; }
LDIR="$RUNS/layer=$LL"
INIT="$LDIR/sae_L${LL}_final"
[ -d "$INIT" ] || { echo "missing $INIT"; exit 1; }
echo "node $(hostname -f) layer $LL task $SLURM_ARRAY_TASK_ID (topup 6000->12000, lr 1e-4)"

# ---- stage node-local slice ----
cp "$SCR/atlas-resorted/layer$LL-tokensorted.parquet" "$SLURM_TMPDIR/"
cp "$SCR/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet" "$SLURM_TMPDIR/"

# ---- env ----
source "${SAE_VENV:-$SCR/sae-venv/bin/activate}"
cd "$SCR"

# ---- train: absolute step timeline 6000 -> 12000, save at 9000 + 12000 ----
python train_sae.py \
  --layer-parquet "$SLURM_TMPDIR/layer$LL-tokensorted.parquet" \
  --scale-parquet "$SLURM_TMPDIR/neuron_scale.parquet" \
  --out-dir "$LDIR" \
  --layer-label "L$LL" \
  --n-latents 32768 --k 32 \
  --batch-tokens 1024 --steps 12000 --start-step 6000 \
  --lr 1e-4 --save-at 9000,12000 \
  --init-from "$INIT" \
  --log-every 50

# ---- stats + census on the 12k final ----
python sae_stats.py "$LDIR/sae_L${LL}_step12000" "$SLURM_TMPDIR/layer$LL-tokensorted.parquet" \
  200000 "$LDIR/census_12k" "$SLURM_TMPDIR/neuron_scale.parquet" \
  | tee "$LDIR/stats_12k.json" \
  || echo "STATS FAILED layer $LL (training artifacts kept)"

# ---- refresh manifest ----
( cd "$LDIR" && find . -type f -exec sha256sum {} \; > "$LDIR/sha256.txt" )
echo "TOPUP DONE layer $LL"
