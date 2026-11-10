#!/bin/bash
#SBATCH -J marin-atlas-stage
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=30G
#SBATCH -t 2-00:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/stage-%j.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/stage-%j.err
# env_apoc adaptation of ops_sh apoc-stage-atlas.sh: computeshort->compute
# (1h cap), in-SIF duckdb resort (PYTHONUSERBASE=qwen35 pyuser, torch2.7.1),
# mkdirs + df -i bookends. Resumable: re-run skips complete files.
set -uo pipefail
DUR=/data/SPCS-IOT/marin-32b
ATLAS=$DUR/atlas
RESORT=$DUR/atlas-resorted
LOGS=/data/scratch/apx186/marin32b/logs
TMPD=/data/scratch/apx186/marin32b/tmp
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
BASE="https://huggingface.co/datasets/Station-house/marin-32b-neuron-atlas/resolve/main"
mkdir -p "$LOGS" "$RESORT" "$TMPD" "$ATLAS"

echo "=== inodes/bytes before ==="
df -i /data/SPCS-IOT | tail -1
df -h /data/SPCS-IOT | tail -1

# 1) pull with node python3 (stdlib only) + node curl — CPU compute nodes
#    have verified HF egress (probe 33544347)
python3 - <<'EOF'
import json, os, subprocess, time
BASE = "https://huggingface.co/datasets/Station-house/marin-32b-neuron-atlas/resolve/main"
ATLAS = "/data/SPCS-IOT/marin-32b/atlas"
files = json.load(open("/data/scratch/apx186/marin32b/filelist.json"))
todo = [f for f in files
        if not (os.path.exists(os.path.join(ATLAS, f["path"]))
                and os.path.getsize(os.path.join(ATLAS, f["path"])) == f["size"])]
print(f"to fetch: {len(todo)}/{len(files)} files", flush=True)
print("proxy env:", {k: os.environ.get(k) for k in
      ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy")}, flush=True)
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
    os.environ.pop(k, None)
r0 = subprocess.run(["curl", "-sfL", "-o", "/dev/null", "--max-time", "20",
                     BASE + "/README.md"])
print("preflight curl rc:", r0.returncode, flush=True)
import concurrent.futures as cf
import threading
done_ct, lock = [0], threading.Lock()

def fetch(f):
    dst = os.path.join(ATLAS, f["path"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        sz = os.path.getsize(dst)
        if sz == 0 or sz > f["size"]:
            os.remove(dst)
    for t in range(6):
        if t > 0 and os.path.exists(dst) and os.path.getsize(dst) == 0:
            os.remove(dst)
        r = subprocess.run(["curl", "-sfL", "-C", "-", "--retry", "3",
                            "--retry-delay", "5", "--max-time", "3600",
                            "-o", dst, BASE + "/" + f["path"]],
                           capture_output=True)
        if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) == f["size"]:
            with lock:
                done_ct[0] += 1
                if done_ct[0] % 25 == 0:
                    print(f"{done_ct[0]}/{len(todo)}", flush=True)
            return None
        time.sleep(10 * (t + 1))
    return f"FAILED {f['path']} rc={r.returncode}"

with cf.ThreadPoolExecutor(max_workers=6) as ex:
    fails = [r for r in ex.map(fetch, todo) if r]
for mline in fails[:40]:
    print(mline, flush=True)
print(f"pull done: {len(todo)-len(fails)} ok, {len(fails)} failed", flush=True)
EOF

# 2) verify sizes vs manifest
python3 - <<'EOF'
import json, os
ATLAS = "/data/SPCS-IOT/marin-32b/atlas"
files = json.load(open("/data/scratch/apx186/marin32b/filelist.json"))
missing = [f["path"] for f in files
           if not os.path.exists(os.path.join(ATLAS, f["path"]))
           or os.path.getsize(os.path.join(ATLAS, f["path"])) != f["size"]]
print(f"verify: {len(files)-len(missing)}/{len(files)} ok, {len(missing)} bad")
for m in missing[:20]: print("BAD", m)
EOF

# 3) pre-resort sparse layers by token_idx (duckdb inside SIF + userbase)
for LL in $(seq -w 0 63); do
  SRC="$ATLAS/pilot/fineweb-10m-sparse/sparse/layer=$LL/sorted-00000.parquet"
  DST="$RESORT/layer$LL-tokensorted.parquet"
  [ -f "$DST" ] && continue
  apptainer exec -B "$DUR:$DUR" -B /data/scratch/apx186:/data/scratch/apx186 \
    --env PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser "$SIF" \
    python3 - "$SRC" "$DST" <<'EOF'
import sys
import duckdb
src, dst = sys.argv[1], sys.argv[2]
con = duckdb.connect()
con.execute("SET memory_limit='10GB'")
con.execute("SET temp_directory='/data/scratch/apx186/marin32b/tmp'")
con.execute(f"""COPY (SELECT token_idx, neuron, gate, up FROM read_parquet('{src}')
                 ORDER BY token_idx) TO '{dst}' (FORMAT PARQUET, ROW_GROUP_SIZE 131072)""")
n = con.execute(f"SELECT count(*) FROM read_parquet('{dst}')").fetchone()[0]
assert n % 64 == 0
print(f"{dst}: {n} rows")
EOF
done

echo "=== inodes/bytes after ==="
df -i /data/SPCS-IOT | tail -1
df -h /data/SPCS-IOT | tail -1
echo "STAGE COMPLETE"
