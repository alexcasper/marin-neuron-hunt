#!/bin/bash
# L00 top-up: warm-start from pilot final, train the unseen remainder of layer 00.
# Pilot consumed file rows [0, 393,216,000) = 6000 batches x 1024 tokens. This script
# slices the rest via duckdb, then runs the census on sae_L00_final, then trains.
# All numbers measured: 318.4 tok/s, 13.89 GiB torch alloc at 32k/bf16/batch1024.
set -euo pipefail
cd ~/marin-sae
export LD_LIBRARY_PATH=/nix/store/78x9i5x1wpqw4kq0h39b8f35abcv156h-zlib-1.3.2/lib
export CC=/nix/store/3d1c302vw7kc8a5vknhmn34c0pd7zm6m-gcc-wrapper-15.3.0/bin/gcc
export TMPDIR=$HOME/marin-sae/tmp
mkdir -p "$TMPDIR" runs

CUT_ROWS=393216000

echo "=== [$(date -Is)] guard: census (inference, ~4 GiB) runs regardless; training batch is chosen later ==="
avail=$(free -m | awk '/^Mem:/{print $7}')
echo "mem available: ${avail} MB"
if [ "$avail" -lt 6000 ]; then echo "ABORT: <6G available, not even census is safe"; exit 2; fi

echo "=== [$(date -Is)] slice top-up parquet (rows >= $CUT_ROWS) via duckdb ==="
if [ ! -f layer00-topup.parquet ]; then
  ~/torch-rocm-env ~/marin-sae/slice_topup.py "$CUT_ROWS"
fi
du -sh layer00-topup.parquet

echo "=== [$(date -Is)] census on sae_L00_final (400k tokens) ==="
~/torch-rocm-env ~/marin-sae/sae_stats.py \
  ~/marin-sae/runs/L00-pilot/sae_L00_final \
  ~/marin-sae/layer00-tokensorted.parquet 400000 \
  | tee census_L00_final.json

echo "=== [$(date -Is)] top-up training: warm-start, ~3.85M unseen tokens ==="
# batch sized to what the host can hold RIGHT NOW: 13.89 GiB alloc measured at batch
# 1024 (pilot), scales ~linearly. 768 -> ~10.4 GiB, safe under the llama servers.
avail=$(free -m | awk '/^Mem:/{print $7}')
if [ "$avail" -ge 13500 ]; then BATCH=1024; STEPS=3762;
elif [ "$avail" -ge 10500 ]; then BATCH=768; STEPS=5015;
else echo "ABORT: only ${avail} MB available even for batch-768 (~10.4 GiB)"; exit 2; fi
echo "training at batch $BATCH, steps $STEPS (avail ${avail} MB)"
~/torch-rocm-env ~/marin-sae/train_sae.py \
  --layer-parquet ~/marin-sae/layer00-topup.parquet \
  --out-dir ~/marin-sae/runs/L00-extend --layer-label L00 \
  --n-latents 32768 --k 32 --batch-tokens $BATCH --steps $STEPS \
  --init-from ~/marin-sae/runs/L00-pilot/sae_L00_final \
  --log-every 50 --save-every 1000

echo "=== [$(date -Is)] census on extended final ==="
~/torch-rocm-env ~/marin-sae/sae_stats.py \
  ~/marin-sae/runs/L00-extend/sae_L00_final \
  ~/marin-sae/layer00-tokensorted.parquet 400000 \
  | tee census_L00_extend.json

echo "=== [$(date -Is)] DONE ==="
