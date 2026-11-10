#!/bin/bash
#SBATCH -J hf-upload-44
#SBATCH -p computeshort
#SBATCH -A pilot_gpu
#SBATCH -t 01:00:00
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -o /data/scratch/apx186/marin32b/logs/hfupload44-%j.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/hfupload44-%j.err
# Push the 44 topup layers' BEST copies (12k finals) to alxcspr/stationhouse.
# One upload_folder commit per layer (44+1 commits, under the 128/h rate cap).
# HF_TOKEN from env only — never echoed or written.
set -uo pipefail
: "${HF_TOKEN:?HF_TOKEN must be set in the job environment}"
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser

echo "node $(hostname -f) $(date -u)"
code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 -L \
  "https://huggingface.co/api/models/alxcspr/stationhouse" || echo curlfail)
echo "preflight huggingface.co: $code"
case "$code" in
  200|401|403|404) ;;
  *) echo "EGRESS BROKEN (code=$code)"; exit 3 ;;
esac

apptainer exec --env PYTHONUSERBASE="$PYTHONUSERBASE" "$SIF" \
  python3 - "$SCR" <<'EOF' || { echo "UPLOAD_FAILED"; exit 5; }
import os, sys, time
from huggingface_hub import HfApi
SCR = sys.argv[1]
api = HfApi(token=os.environ["HF_TOKEN"])
print("auth as:", api.whoami().get("name"), flush=True)
layers = [l.strip() for l in open(f"{SCR}/topup-order.txt") if l.strip()]
print(len(layers), "layers to upload", flush=True)
failed = []
for LL in layers:
    d = f"{SCR}/sae-32k/layer={LL}"
    try:
        api.upload_folder(
            folder_path=d,
            path_in_repo=f"sae-32k/layer={LL}",
            repo_id="alxcspr/stationhouse", repo_type="model",
            allow_patterns=[f"sae_L{LL}_step12000*", "stats_12k.json",
                            "census_12k.parquet", "train_log.jsonl", "sha256.txt"],
            commit_message=f"layer {LL}: best copy (12k topup final)")
        print("ok layer", LL, flush=True)
    except Exception as e:
        # one retry after a pause (transient 429/5xx)
        time.sleep(60)
        try:
            api.upload_folder(folder_path=d, path_in_repo=f"sae-32k/layer={LL}",
                              repo_id="alxcspr/stationhouse", repo_type="model",
                              allow_patterns=[f"sae_L{LL}_step12000*", "stats_12k.json",
                                              "census_12k.parquet", "train_log.jsonl", "sha256.txt"],
                              commit_message=f"layer {LL}: best copy (12k topup final, retry)")
            print("ok layer", LL, "(retry)", flush=True)
        except Exception as e2:
            print(f"FAIL layer {LL}: {e2}", flush=True)
            failed.append(LL)
api.upload_file(path_or_fileobj=f"{SCR}/hf-README-64.md", path_in_repo="README.md",
                repo_id="alxcspr/stationhouse", repo_type="model",
                commit_message="README: full 64-layer provenance + fvu table")
print("ok README", flush=True)
print("UPLOAD44 COMPLETE. failed:", failed if failed else "none", flush=True)
EOF
