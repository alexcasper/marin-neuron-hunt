#!/bin/bash
#SBATCH -J marin-atlas-stage
#SBATCH -p computeshort
#SBATCH -t 04:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/stage-%j.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/stage-%j.err
# One-shot: pull the marin-32b-neuron-atlas pilot tree from HF into durable,
# verify sizes vs manifest, then pre-resort the 64 sparse layer files by
# token_idx. Resumable: re-run skips complete files.
set -uo pipefail
DUR=/data/SPCS-IOT/marin-32b
ATLAS=$DUR/atlas
RESORT=$DUR/atlas-resorted
LOGS=/data/scratch/apx186/marin32b/logs
BASE="https://huggingface.co/datasets/Station-house/marin-32b-neuron-atlas/resolve/main"
mkdir -p "$LOGS" "$RESORT"

echo "=== inodes before ==="; df -i /data/SPCS-IOT | tail -1

# 1) pull: iterate filelist.json (path+size), skip complete, curl-resume
python3 - <<'EOF'
import json, os, subprocess, time
BASE = "https://huggingface.co/datasets/Station-house/marin-32b-neuron-atlas/resolve/main"
ATLAS = "/data/SPCS-IOT/marin-32b/atlas"
files = json.load(open("/data/scratch/apx186/marin32b/filelist.json"))
todo = [f for f in files
        if not (os.path.exists(os.path.join(ATLAS, f["path"]))
                and os.path.getsize(os.path.join(ATLAS, f["path"])) == f["size"])]
print(f"to fetch: {len(todo)}/{len(files)} files", flush=True)
for i, f in enumerate(todo):
    dst = os.path.join(ATLAS, f["path"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for t in range(5):
        r = subprocess.run(["curl", "-sfL", "-C", "-", "--retry", "3",
                            "--retry-delay", "5", "-o", dst, f"{BASE}/{f['path']}"],
                           capture_output=True)
        if r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) == f["size"]:
            break
        time.sleep(10 * (t + 1))
    else:
        print(f"FAILED {f['path']}", flush=True)
        continue
    if i % 50 == 0:
        print(f"{i}/{len(todo)}", flush=True)
print("pull done", flush=True)
EOF

# 2) verify: every filelist entry present with exact size
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

# 3) pre-resort sparse layers by token_idx (duckdb) for array-task streaming
mkdir -p "$RESORT"
for LL in $(seq -w 0 63); do
  SRC="$ATLAS/pilot/fineweb-10m-sparse/sparse/layer=$LL/sorted-00000.parquet"
  DST="$RESORT/layer$LL-tokensorted.parquet"
  if [ -f "$DST" ]; then continue; fi
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

echo "=== inodes after ==="; df -i /data/SPCS-IOT | tail -1
echo "STAGE COMPLETE"
