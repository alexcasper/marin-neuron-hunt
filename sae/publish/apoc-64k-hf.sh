#!/bin/bash
#SBATCH -J 64k-hf
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -t 04:00:00
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -o /data/scratch/apx186/marin32b/logs/64khf-%j.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/64khf-%j.err
# Push 64k-probe artifacts (finals + stats + val_log + census + sha256,
# per layer) to alxcspr/stationhouse under 64k-probe/. One commit per layer.
set -uo pipefail
: "${HF_TOKEN:?HF_TOKEN must be set in the job environment}"
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser

code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 -L \
  "https://huggingface.co/api/models/alxcspr/stationhouse" || echo curlfail)
echo "preflight: $code"
case "$code" in 200|401|403|404) ;; *) echo "EGRESS BROKEN"; exit 3 ;; esac

apptainer exec --env PYTHONUSERBASE="$PYTHONUSERBASE" "$SIF" \
  python3 - "$SCR" <<'EOF' || { echo "UPLOAD_FAILED"; exit 5; }
import os, sys, glob, time
from huggingface_hub import HfApi
SCR = sys.argv[1]
api = HfApi(token=os.environ["HF_TOKEN"])
print("auth as:", api.whoami().get("name"), flush=True)
failed = []
for d in sorted(glob.glob(f"{SCR}/64k-probe/layer=*")):
    LL = d.split("layer=")[1]
    pats = [f"sae_L{LL}w64_plateau*", f"sae_L{LL}w64_final*", "stats.json",
            "census.parquet", "val_log.jsonl", "train_log.jsonl", "sha256.txt"]
    try:
        api.upload_folder(folder_path=d, path_in_repo=f"64k-probe/layer={LL}",
                          repo_id="alxcspr/stationhouse", repo_type="model",
                          allow_patterns=pats,
                          commit_message=f"64k probe layer {LL}: plateau final + stats")
        print("ok layer", LL, flush=True)
    except Exception as e:
        time.sleep(60)
        try:
            api.upload_folder(folder_path=d, path_in_repo=f"64k-probe/layer={LL}",
                              repo_id="alxcspr/stationhouse", repo_type="model",
                              allow_patterns=pats,
                              commit_message=f"64k probe layer {LL}: plateau final + stats (retry)")
            print("ok layer", LL, "(retry)", flush=True)
        except Exception as e2:
            print(f"FAIL layer {LL}: {e2}", flush=True)
            failed.append(LL)
print("UPLOAD64K COMPLETE. failed:", failed if failed else "none", flush=True)
EOF
