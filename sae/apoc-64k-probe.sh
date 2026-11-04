#!/bin/bash
#SBATCH -J marin-64k-probe
#SBATCH -p gpu
#SBATCH --gres gpu:nvidia_a100-pcie-40gb:1
#SBATCH -a 0-2%3
#SBATCH -t 08:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/64k-probe-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/64k-probe-%A_%a.err
# 64k WIDTH PROBE: fresh-init TopK 64k (k=32) on L57 / L45 / L63, run to
# convergence via the plateau detector (6k window, 2% thresh), horizon 50k.
# Val holdout (30k trailing tokens) every 3k steps — held-out fvu is the metric.
# Checkpoints every 3k; retention keeps 9k grid + latest.
# Task order = 64k-probe-layers.txt: 57 (best-case), 45 (mid-band 0.95-class), 63 (collapsed).
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
RUNS=$SCR/64k-probe
TMPD=$SCR/tmp
mkdir -p "$RUNS" "$TMPD"

ORDER=$SCR/64k-probe-layers.txt
LINE=$((SLURM_ARRAY_TASK_ID + 1))
LL=$(sed -n "${LINE}p" "$ORDER")
[ -n "$LL" ] || { echo "no layer for task $SLURM_ARRAY_TASK_ID"; exit 1; }
OUT=$RUNS/layer=$LL
mkdir -p "$OUT"
echo "node $(hostname -f) 64k probe layer $LL task $SLURM_ARRAY_TASK_ID"

cp "$SCR/atlas-resorted/layer$LL-tokensorted.parquet" "$SLURM_TMPDIR/"
cp "$SCR/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet" "$SLURM_TMPDIR/"

source "${SAE_VENV:-$SCR/sae-venv/bin/activate}"
cd "$SCR"

# 64k/bf16 measured at 27.4 GiB torch alloc (fits A100-40); batch 1024 unchanged
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
  --save-every 3000 --keep-grid 9000 \
  --log-every 200

# census on the standard 200,704-token protocol (comparable to sweep tables)
python sae_stats.py "$OUT/sae_L${LL}w64_final" "$SLURM_TMPDIR/layer$LL-tokensorted.parquet" \
  200704 "$OUT/census" "$SLURM_TMPDIR/neuron_scale.parquet" \
  | tee "$OUT/stats.json" \
  || echo "STATS FAILED layer $LL (training artifacts kept)"

( cd "$OUT" && find . -type f -exec sha256sum {} \; > "$OUT/sha256.txt" )
echo "64K PROBE DONE layer $LL"
