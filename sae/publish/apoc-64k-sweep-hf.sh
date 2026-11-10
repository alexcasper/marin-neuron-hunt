#!/bin/bash
#SBATCH -J 64ksweep-hf
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -t 08:00:00
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -o /data/scratch/apx186/marin32b/logs/64ksweephf-%j.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/64ksweephf-%j.err
# Push finished sae-64k sweep layers (final + stats + census + val/train logs
# + sha256) to alxcspr/stationhouse under sae-64k/layer=NN. One commit per
# layer; idempotent (re-run skips already-uploaded identical files).
set -uo pipefail
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser

code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 -L \
  "https://huggingface.co/api/models/alxcspr/stationhouse" || echo curlfail)
echo "preflight: $code"
case "$code" in 200|401|403|404) ;; *) echo "EGRESS BROKEN"; exit 3 ;; esac

apptainer exec -B /data/scratch/apx186:/data/scratch/apx186 --env PYTHONUSERBASE="$PYTHONUSERBASE" "$SIF" \
  python3 - "$SCR" <<'EOF' || { echo "UPLOAD_FAILED"; exit 5; }
import os, sys, glob, time
from huggingface_hub import HfApi
SCR = sys.argv[1]
tok = open(f"{SCR}/.hftok").read().strip()
api = HfApi(token=tok)
print("auth as:", api.whoami().get("name"), flush=True)
failed = []
done = []
for d in sorted(glob.glob(f"{SCR}/sae-64k/layer=*")):
    LL = d.split("layer=")[1]
    if not glob.glob(f"{d}/sae_L{LL}w64_final*"):
        print("skip layer", LL, "(no final yet)", flush=True)
        continue
    pats = [f"sae_L{LL}w64_final*", "stats.json", "census.parquet",
            "val_log.jsonl", "train_log.jsonl", "sha256.txt"]
    try:
        api.upload_folder(folder_path=d, path_in_repo=f"sae-64k/layer={LL}",
                          repo_id="alxcspr/stationhouse", repo_type="model",
                          allow_patterns=pats,
                          commit_message=f"64k sweep layer {LL}: final + stats + census")
        print("ok layer", LL, flush=True); done.append(LL)
    except Exception as e:
        time.sleep(60)
        try:
            api.upload_folder(folder_path=d, path_in_repo=f"sae-64k/layer={LL}",
                              repo_id="alxcspr/stationhouse", repo_type="model",
                              allow_patterns=pats,
                              commit_message=f"64k sweep layer {LL}: final + stats + census (retry)")
            print("ok layer", LL, "(retry)", flush=True); done.append(LL)
        except Exception as e2:
            print(f"FAIL layer {LL}: {e2}", flush=True); failed.append(LL)
print(f"UPLOAD64KSWEEP COMPLETE uploaded={len(done)} failed={failed if failed else 'none'}", flush=True)
EOF
