#!/usr/bin/env bash
# sh-bs6 bulk pull v3: direct curl from HF resolve URLs, resumable, parallel.
# Driven by survey/filelist.json (tree API, cursor pagination: 1,062 files / 411,904,202,313 bytes).
# Usage: pull_v3.sh <start_index>
set -uo pipefail
cd /mnt/d/ops/sh
.venv/bin/python - "$@" <<'EOF'
import json, os, subprocess, sys, time
BASE = "https://huggingface.co/datasets/Station-house/marin-32b-neuron-atlas/resolve/main"
start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
files = json.load(open("survey/filelist.json"))
todo = []
for f in files[start:]:
    dst = os.path.join("data", f["path"])
    if os.path.exists(dst) and os.path.getsize(dst) == f["size"]:
        continue
    todo.append(f)
print(f"to fetch: {len(todo)} files ({sum(f['size'] for f in todo)/1e9:.1f} GB), starting at index {start}", flush=True)

def fetch(f, tries=5):
    dst = os.path.join("data", f["path"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = f"{BASE}/{f['path']}"
    for t in range(tries):
        r = subprocess.run(["curl", "-sfL", "-C", "-", "--retry", "3", "--retry-delay", "5",
                            "-o", dst, url], capture_output=True)
        if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) == f["size"]:
            return True
        time.sleep(10 * (t + 1))
    print(f"FAILED {f['path']} rc={r.returncode}", flush=True)
    return False

from concurrent.futures import ThreadPoolExecutor
done = fail = 0
t0 = time.time()
with ThreadPoolExecutor(max_workers=6) as ex:
    for ok in ex.map(fetch, todo):
        done += 1
        if not ok:
            fail += 1
        if done % 10 == 0:
            el = time.time() - t0
            print(f"progress {done}/{len(todo)} failed={fail} elapsed={el/60:.1f}m", flush=True)
print(f"DONE fetched={done} failed={fail} elapsed_s={time.time()-t0:.0f}", flush=True)
sys.exit(1 if fail else 0)
EOF
echo "=== verify $(date -u +%FT%TZ) ==="
.venv/bin/python - <<'EOF'
import json, os
files = json.load(open("survey/filelist.json"))
bad = [f for f in files if not (os.path.exists(os.path.join("data", f["path"])) and os.path.getsize(os.path.join("data", f["path"])) == f["size"])]
total = sum(os.path.getsize(os.path.join("data", f["path"])) for f in files if os.path.exists(os.path.join("data", f["path"])))
print(f"files_ok={len(files)-len(bad)}/{len(files)} total_bytes_on_disk={total:,}")
if bad:
    print("mismatches:", *[b["path"] for b in bad[:20]], sep="\n  ")
EOF
