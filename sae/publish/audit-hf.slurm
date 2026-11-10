#!/bin/bash
#SBATCH -J audit-hf
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:20:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/audithf-%j.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/audithf-%j.err
apptainer exec -B /data/scratch/apx186:/data/scratch/apx186 --env PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser /data/scratch/apx186/bolmo/sif/bolmo_cu118.sif python3 - <<'EOF'
from huggingface_hub import HfApi
tok = open("/data/scratch/apx186/marin32b/.hftok").read().strip()
api = HfApi(token=tok)
ci = api.upload_file(
    path_or_fileobj="/data/scratch/apx186/marin32b/interp-12k-audit-20261106.tgz",
    path_in_repo="sae-32k/interp-12k-audit-20261106.tgz",
    repo_id="alxcspr/stationhouse", repo_type="model",
    commit_message="12k interp audit bundle 2026-11-06")
print("COMMIT:", ci)
print("URL:", "https://huggingface.co/alxcspr/stationhouse/commit/" + ci.commit_url.split("/")[-1] if hasattr(ci,"commit_url") else ci.oid)
print("OID:", ci.oid)
EOF
