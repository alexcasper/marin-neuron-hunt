#!/bin/bash
#SBATCH -J preview-hf
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -t 01:00:00
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -o /data/scratch/apx186/marin32b/logs/previewhf-%j.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/previewhf-%j.err
set -uo pipefail
: "${HF_TOKEN:?missing}"
apptainer exec --env PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser /data/scratch/apx186/bolmo/sif/bolmo_cu118.sif python3 - <<"EOF"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
p = "/data/scratch/apx186/marin32b/sae-32k/layer=57/feature-preview-L57.md"
api.upload_file(path_or_fileobj=p, path_in_repo="sae-32k/layer=57/feature-preview-L57.md",
                repo_id="alxcspr/stationhouse", repo_type="model",
                commit_message="L57 feature preview")
print("PREVIEW UPLOADED")
EOF
