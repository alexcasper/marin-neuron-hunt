#!/bin/bash
#SBATCH -J marin-topup
#SBATCH -p gpu
#SBATCH -A pilot_gpu
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH -a 0-43%8
#SBATCH -c 4
#SBATCH --mem=45G
#SBATCH -t 02:30:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/topup-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/topup-%A_%a.err
# env_apoc adaptation of ops_sh apoc-topup.sh: resume each layer's 6k final,
# +6k steps at lr 1e-4 to 12k ABSOLUTE; saves ONLY step9000 + step12000;
# then stats_12k.json + census_12k.parquet. Task order = topup-order.txt
# (worst fvu first; task 0 = L63 scale-collapse probe).
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
RUNS=$SCR/sae-32k
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
mkdir -p "$RUNS"

# apoc has no SLURM_TMPDIR: node-local staging with cleanup trap
STAGED=$(mktemp -d /tmp/marin_topup_${SLURM_ARRAY_TASK_ID}_XXXXXX)
trap 'rm -rf "$STAGED"' EXIT

ORDER=$SCR/topup-order.txt
LINE=$((SLURM_ARRAY_TASK_ID + 1))
LL=$(sed -n "${LINE}p" "$ORDER")
[ -n "$LL" ] || { echo "no layer for task $SLURM_ARRAY_TASK_ID"; exit 1; }
LDIR="$RUNS/layer=$LL"
INIT="$LDIR/sae_L${LL}_final"
[ -d "$INIT" ] || { echo "missing $INIT"; exit 1; }
echo "node $(hostname -f) layer $LL task $SLURM_ARRAY_TASK_ID (topup 6000->12000, lr 1e-4)"

# ---- stage node-local slice ----
cp "$SCR/atlas-resorted/layer$LL-tokensorted.parquet" "$STAGED/"
cp "$SCR/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet" "$STAGED/"
df -h /tmp | tail -1

# ---- train: absolute step timeline 6000 -> 12000, save at 9000 + 12000 ----
apptainer exec --nv -B /data/scratch/apx186:/data/scratch/apx186 \
  -B "$STAGED:$STAGED" \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" "$SIF" \
  python3 -u "$SCR/train_sae.py" \
  --layer-parquet "$STAGED/layer$LL-tokensorted.parquet" \
  --scale-parquet "$STAGED/neuron_scale.parquet" \
  --out-dir "$LDIR" \
  --layer-label "L$LL" \
  --n-latents 32768 --k 32 \
  --batch-tokens 1024 --steps 12000 --start-step 6000 \
  --lr 1e-4 --save-at 9000,12000 \
  --init-from "$INIT" \
  --log-every 50

# ---- stats + census on the 12k final ----
apptainer exec --nv -B /data/scratch/apx186:/data/scratch/apx186 \
  -B "$STAGED:$STAGED" \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" "$SIF" \
  python3 -u "$SCR/sae_stats.py" "$LDIR/sae_L${LL}_step12000" \
  "$STAGED/layer$LL-tokensorted.parquet" \
  200000 "$LDIR/census_12k" "$STAGED/neuron_scale.parquet" \
  > "$LDIR/stats_12k.json" \
  || echo "STATS FAILED layer $LL (training artifacts kept)"

# ---- refresh manifest ----
( cd "$LDIR" && find . -type f -exec sha256sum {} \; > "$LDIR/sha256.txt" )
echo "TOPUP DONE layer $LL"
