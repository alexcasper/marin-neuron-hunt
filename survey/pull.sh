#!/usr/bin/env bash
# sh-bs6 bulk pull v2: Station-house/marin-32b-neuron-atlas -> /mnt/d/ops/sh/data
# v1 died ~20:09Z (killed, no traceback). v2: setsid-detached, 20 retries, xet high-perf.
# TRUE total (tree API, cursor pagination): 411,904,202,313 bytes / 1,062 files.
set -uo pipefail
cd /mnt/d/ops/sh
export HF_XET_HIGH_PERFORMANCE=1
for ATTEMPT in $(seq 1 20); do
  echo "=== attempt $ATTEMPT $(date -u +%FT%TZ) ==="
  .venv/bin/python - <<'EOF' && break
from huggingface_hub import snapshot_download
import time
t0 = time.time()
p = snapshot_download(
    repo_id="Station-house/marin-32b-neuron-atlas",
    repo_type="dataset",
    local_dir="/mnt/d/ops/sh/data",
    max_workers=8,
)
print(f"DONE path={p} elapsed_s={time.time()-t0:.0f}", flush=True)
EOF
  echo "=== snapshot_download exited rc=$? at $(date -u +%FT%TZ); retrying in 30s ==="
  sleep 30
done
echo "=== verify $(date -u +%FT%TZ) ==="
echo "parquet_files=$(find /mnt/d/ops/sh/data -name '*.parquet' | wc -l)"
echo "total_bytes=$(du -sb /mnt/d/ops/sh/data | cut -f1)"
