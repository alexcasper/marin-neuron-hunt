#!/bin/bash
# STAGED, NOT LAUNCHED (2026-10-27): 64k-latent rung for layer 00.
# Awaiting alex's explicit go — 64k was originally H200-class; it fits dusk only
# as bf16 (27.4 GiB of 48G, measured) and ONLY on a quiet host (no bulk rsync:
# page-cache growth pushed a sizing run into reclaim-crawl once already).
#
# To fire:  ssh dusk 'ALEX_GO=1 nohup ~/marin-sae/run_L00_64k.sh > ~/marin-sae/L00-64k.log 2>&1 &'
set -euo pipefail
cd ~/marin-sae
if [ "${ALEX_GO:-0}" != "1" ]; then
  echo "STAGED ONLY: set ALEX_GO=1 to launch (alex's call, not ours)."; exit 3
fi
export LD_LIBRARY_PATH=/nix/store/78x9i5x1wpqw4kq0h39b8f35abcv156h-zlib-1.3.2/lib
export CC=/nix/store/3d1c302vw7kc8a5vknhmn34c0pd7zm6m-gcc-wrapper-15.3.0/bin/gcc

avail=$(free -m | awk '/^Mem:/{print $7}')
echo "mem available: ${avail} MB"
if [ "$avail" -lt 30000 ]; then echo "ABORT: 64k bf16 needs ~27.4G alloc; <30G available."; exit 2; fi
if pgrep -x rsync >/dev/null; then echo "ABORT: rsync running."; exit 2; fi

# 1 epoch of layer 00 = 9762 steps x 1024 tokens (9,995,501 tokens).
# Measured throughput at this width (sizing smoke, batch 1024, prefetch on): ~160 tok/s
# -> ~17.4 h wall. Checkpoints 3.6 GB x 9 + final: ~36 GB — fits 356G free.
~/torch-rocm-env ~/marin-sae/train_sae.py \
  --layer-parquet ~/marin-sae/layer00-tokensorted.parquet \
  --out-dir ~/marin-sae/runs/L00-64k --layer-label L00 \
  --n-latents 65536 --k 64 --batch-tokens 1024 --steps 9762 \
  --log-every 50 --save-every 1000
