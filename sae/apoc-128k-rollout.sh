#!/bin/bash
#SBATCH -J marin-128k-roll
#SBATCH -p gpu
#SBATCH --gres gpu:nvidia_a100_80gb_pcie:1
#SBATCH -a 0-60%6
#SBATCH -t 10-00:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/128k-roll-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/128k-roll-%A_%a.err
# 128k ROLLOUT (fires ONLY if the probe rule triggers: median val-fvu-at-plateau
# improvement >= 0.05 absolute vs matched 64k). All 61 layers except probe set
# (57,45,63) via 128k-rollout-layers.txt. Convergence protocol identical to the
# probe. FINALS -> durable /data/SPCS-IOT/marin-32b/sae-128k/ (trained artifacts,
# not re-pullable). Intermediates: scratch 3k-grid with keep-grid 9000 cull.
# Layer dir on durable gets: final + stats.json + census.parquet + val_log.jsonl.
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
DUR=/data/SPCS-IOT/marin-32b
TMPD=$SCR/tmp
mkdir -p "$DUR/sae-128k" "$TMPD"

ORDER=$SCR/128k-rollout-layers.txt
LINE=$((SLURM_ARRAY_TASK_ID + 1))
LL=$(sed -n "${LINE}p" "$ORDER")
[ -n "$LL" ] || { echo "no layer for task $SLURM_ARRAY_TASK_ID"; exit 1; }
OUT=$SCR/128k-run/layer=$LL
DOUT=$DUR/sae-128k/layer=$LL
mkdir -p "$OUT" "$DOUT"
echo "node $(hostname -f) 128k rollout layer $LL task $SLURM_ARRAY_TASK_ID"

cp "$SCR/atlas-resorted/layer$LL-tokensorted.parquet" "$SLURM_TMPDIR/"
cp "$SCR/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet" "$SLURM_TMPDIR/"

source "${SAE_VENV:-$SCR/sae-venv/bin/activate}"
cd "$SCR"

# stage from DURABLE atlas copy? No — atlas lives on scratch too (65-day purge,
# re-pullable). Scratch staging copy is the primary source; if purged, re-pull
# that layer from HF first (see apoc-stage-atlas.sh pattern, single-layer).
if [ ! -f "$SCR/atlas-resorted/layer$LL-tokensorted.parquet" ]; then
  echo "resorted file missing — re-pull + re-resort layer $LL from HF"
  python3 - "$LL" <<'EOF'
import sys, subprocess
LL = f"{int(sys.argv[1]):02d}"
BASE = "https://huggingface.co/datasets/Station-house/marin-32b-neuron-atlas/resolve/main"
raw = f"/data/scratch/apx186/marin32b/atlas/pilot/fineweb-10m-sparse/sparse/layer={LL}/sorted-00000.parquet"
os.makedirs(os.path.dirname(raw), exist_ok=True)
for t in range(5):
    r = subprocess.run(["curl", "-sfL", "-C", "-", "--retry", "3", "--retry-delay", "5", "-o", raw,
                        f"{BASE}/pilot/fineweb-10m-sparse/sparse/layer={LL}/sorted-00000.parquet"])
    if r.returncode == 0: break
EOF
  python3 - "$SCR/atlas/pilot/fineweb-10m-sparse/sparse/layer=$LL/sorted-00000.parquet" \
    "$SCR/atlas-resorted/layer$LL-tokensorted.parquet" <<'EOF'
import sys
import duckdb
src, dst = sys.argv[1], sys.argv[2]
con = duckdb.connect(); con.execute("SET memory_limit='10GB'")
con.execute("SET temp_directory='/data/scratch/apx186/marin32b/tmp'")
con.execute(f"""COPY (SELECT token_idx, neuron, gate, up FROM read_parquet('{src}')
                 ORDER BY token_idx) TO '{dst}' (FORMAT PARQUET, ROW_GROUP_SIZE 131072)""")
EOF
fi

python train_sae.py \
  --layer-parquet "$SLURM_TMPDIR/layer$LL-tokensorted.parquet" \
  --scale-parquet "$SCR/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet" \
  --out-dir "$OUT" \
  --layer-label "L${LL}w128" \
  --n-latents 131072 --k 32 \
  --batch-tokens 1024 --steps 50000 \
  --lr 3e-4 --seed 20261027 \
  --val-rows 1920000 --val-tokens 30000 --val-every 3000 \
  --plateau-window 6000 --plateau-thresh 0.02 \
  --save-every 3000 --keep-grid 9000 \
  --log-every 200

# FINAL + support files -> durable (trained artifacts). Step ckpts stay scratch.
FINAL=$(ls -d "$OUT"/sae_L${LL}w128_*final* 2>/dev/null | head -1)
if [ -n "$FINAL" ]; then
  mkdir -p "$DOUT/$(basename "$FINAL")"
  cp -r "$FINAL"/. "$DOUT/$(basename "$FINAL")"/
  cp "$OUT"/val_log.jsonl "$DOUT"/ 2>/dev/null || true
  cp "$OUT"/stats.json "$DOUT"/ 2>/dev/null || true
  ( cd "$DOUT" && find . -type f -exec sha256sum {} \; > "$DOUT/sha256.txt" )
else
  echo "NO FINAL CHECKPOINT layer $LL"
fi
echo "128K ROLLOUT DONE layer $LL"
