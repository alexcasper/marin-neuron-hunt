#!/bin/bash
#SBATCH -J marin-l63-retrain
#SBATCH -p gpu
#SBATCH --gres gpu:nvidia_a100-pcie-40gb:1
#SBATCH -a 0-2%3
#SBATCH -t 02:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/l63-retrain-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/l63-retrain-%A_%a.err
# L63 COLLAPSE DIAGNOSTIC: 3 fresh-init 6k variants (sweep-standard budget).
#   task 0 (variant-a): fresh init, seed 20261027        — tests seed-luck
#   task 1 (variant-b): fresh init, seed 20271027        — seed-luck replicate
#   task 2 (variant-c): fresh init, --l2-norm-input      — targeted at STEP-0 locus
#     (L63 top-64 captures only 12.5% of raw act^2; input is systematically
#      altered data — L2 norm removes residual scale heterogeneity)
# Original collapsed reference: 12k census fvu 1.0365 (6k census 1.1839).
# Sweep normal band: 0.87-1.04. Success gate: any variant < 0.95 becomes the
# shipped L63 (ops_sh verifies + pushes, truthful naming).
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
RUNS=$SCR/sae-32k
TMPD=$SCR/tmp
mkdir -p "$RUNS" "$TMPD"

case $SLURM_ARRAY_TASK_ID in
  0) VARIANT=variant-a; EXTRA="";                      SEED=20261027 ;;
  1) VARIANT=variant-b; EXTRA="";                      SEED=20271027 ;;
  2) VARIANT=variant-c; EXTRA="--l2-norm-input";       SEED=20261027 ;;
  *) echo "bad task id"; exit 1 ;;
esac
OUT=$RUNS/layer=63/$VARIANT
mkdir -p "$OUT"

cp "$SCR/atlas-resorted/layer63-tokensorted.parquet" "$SLURM_TMPDIR/"
cp "$SCR/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet" "$SLURM_TMPDIR/"

source "${SAE_VENV:-$SCR/sae-venv/bin/activate}"
cd "$SCR"

python train_sae.py \
  --layer-parquet "$SLURM_TMPDIR/layer63-tokensorted.parquet" \
  --scale-parquet "$SLURM_TMPDIR/neuron_scale.parquet" \
  --out-dir "$OUT" \
  --layer-label "L63_$VARIANT" \
  --n-latents 32768 --k 32 \
  --batch-tokens 1024 --steps 6000 \
  --lr 3e-4 --seed "$SEED" \
  $EXTRA \
  --log-every 50

# census on first-200k tokens — same protocol as the sweep (comparable numbers)
python sae_stats.py "$OUT/sae_L63_${VARIANT}_final" "$SLURM_TMPDIR/layer63-tokensorted.parquet" \
  200704 "$OUT/census" "$SLURM_TMPDIR/neuron_scale.parquet" \
  | tee "$OUT/stats.json" \
  || echo "STATS FAILED $VARIANT"

( cd "$OUT" && find . -type f -exec sha256sum {} \; > "$OUT/sha256.txt" )
echo "L63 RETRAIN DONE $VARIANT"
