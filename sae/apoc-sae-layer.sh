#!/bin/bash
#SBATCH -J marin-sae
#SBATCH -p gpu
#SBATCH --gres gpu:nvidia_a100-pcie-40gb:1
#SBATCH -a 0-63%8
#SBATCH -t 02:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/sae-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/sae-%A_%a.err
# One layer per task: stage slice from durable -> train TopK SAE -> durable runs/.
# SMOKE GATE: sbatch -a 0-1 with SMOKE=1 first (200 steps); release full array
# only after fvu falls and tok/s is sane.
set -euo pipefail
DUR=/data/SPCS-IOT/marin-32b
ATLAS=/data/scratch/apx186/marin32b/atlas
RESORT=/data/scratch/apx186/marin32b/atlas-resorted
RUNS=$DUR/runs
TMPD=/data/scratch/apx186/marin32b/tmp
mkdir -p "$RUNS" "$TMPD"

LL=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
echo "node $(hostname -f) layer $LL task $SLURM_ARRAY_TASK_ID"

STEPS=${STEPS:-6000}
if [ "${SMOKE:-0}" = "1" ]; then STEPS=200; fi

# ---- stage node-local slice (scratch staging copy -> $SLURM_TMPDIR; GPU nodes
# have no HF egress, so tasks read the scratch copy, cleaned up on job end) ----
cp "$RESORT/layer$LL-tokensorted.parquet" "$SLURM_TMPDIR/" \
  || { cp "/data/scratch/apx186/marin32b/layer$LL-tokensorted.parquet" "$SLURM_TMPDIR/"; }
cp /data/scratch/apx186/marin32b/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet "$SLURM_TMPDIR/"
ls -la "$SLURM_TMPDIR"

# ---- env: bolmo_cu118.sif userbase (torch 2.7.1 cu126), eai-sparsify
# PRE-INSTALLED during staging (GPU nodes have NO egress — never pip here).
# env_apoc: point SAE_VENV at the prepared venv/userbase. ----
source "${SAE_VENV:-/data/scratch/apx186/marin32b/sae-venv/bin/activate}"
cd /data/scratch/apx186/marin32b

# ---- train ----
python train_sae.py \
  --layer-parquet "$SLURM_TMPDIR/layer$LL-tokensorted.parquet" \
  --scale-parquet "$SLURM_TMPDIR/neuron_scale.parquet" \
  --out-dir "$RUNS/L$LL" \
  --layer-label "L$LL" \
  --n-latents 32768 --k 32 \
  --batch-tokens 1024 --steps "$STEPS" \
  --log-every 50 --save-every 1000

# ---- manifest outputs ----
( cd "$RUNS/L$LL" && find . -type f -exec sha256sum {} \; > "$RUNS/L$LL.sha256" )
echo "TASK DONE layer $LL"
