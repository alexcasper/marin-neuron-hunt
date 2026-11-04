#!/bin/bash
#SBATCH -J marin-128k-probe
#SBATCH -p gpu
#SBATCH --gres gpu:nvidia_a100_80gb_pcie:1
#SBATCH -a 0-2%3
#SBATCH -t 10-00:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/128k-probe-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/128k-probe-%A_%a.err
# 128k WIDTH PROBE (conditional on 64k probe completion): fresh-init TopK 128k
# (k=32) on the same three probe layers (L57/L45/L63), same val/plateau protocol.
# Horizon 50k absolute steps; plateau 6k-window/2% thresh; ckpt 3k grid,
# retention keep-grid 9000. Task order = 128k-probe-layers.txt (57,45,63).
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
RUNS=$SCR/128k-probe
TMPD=$SCR/tmp
mkdir -p "$RUNS" "$TMPD"

ORDER=$SCR/128k-probe-layers.txt
LINE=$((SLURM_ARRAY_TASK_ID + 1))
LL=$(sed -n "${LINE}p" "$ORDER")
[ -n "$LL" ] || { echo "no layer for task $SLURM_ARRAY_TASK_ID"; exit 1; }
OUT=$RUNS/layer=$LL
mkdir -p "$OUT"
echo "node $(hostname -f) 128k probe layer $LL task $SLURM_ARRAY_TASK_ID"

cp "$SCR/atlas-resorted/layer$LL-tokensorted.parquet" "$SLURM_TMPDIR/"
cp "$SCR/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet" "$SLURM_TMPDIR/"

source "${SAE_VENV:-$SCR/sae-venv/bin/activate}"
cd "$SCR"

# 128k/bf16: params+moments ~29G + activations — A100-80. Batch 1024.
python train_sae.py \
  --layer-parquet "$SLURM_TMPDIR/layer$LL-tokensorted.parquet" \
  --scale-parquet "$SLURM_TMPDIR/neuron_scale.parquet" \
  --out-dir "$OUT" \
  --layer-label "L${LL}w128" \
  --n-latents 131072 --k 32 \
  --batch-tokens 1024 --steps 50000 \
  --lr 3e-4 --seed 20261027 \
  --val-rows 1920000 --val-tokens 30000 --val-every 3000 \
  --plateau-window 6000 --plateau-thresh 0.02 \
  --save-every 3000 --keep-grid 9000 \
  --log-every 200

python sae_stats.py "$OUT/sae_L${LL}w128_final" "$SLURM_TMPDIR/layer$LL-tokensorted.parquet" \
  200704 "$OUT/census" "$SLURM_TMPDIR/neuron_scale.parquet" \
  | tee "$OUT/stats.json" \
  || echo "STATS FAILED layer $LL (training artifacts kept)"

( cd "$OUT" && find . -type f -exec sha256sum {} \; > "$OUT/sha256.txt" )
echo "128K PROBE DONE layer $LL"
