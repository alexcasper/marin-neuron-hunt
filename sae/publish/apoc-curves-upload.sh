#!/bin/bash
#SBATCH -J curves-hf
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -t 02:00:00
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -o /data/scratch/apx186/marin32b/logs/curveshf-%j.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/curveshf-%j.err
set -uo pipefail
: "${HF_TOKEN:?HF_TOKEN must be set}"
apptainer exec --env PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser /data/scratch/apx186/bolmo/sif/bolmo_cu118.sif python3 - <<"EOF"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(folder_path="/data/scratch/apx186/marin32b/curves",
                  path_in_repo="curves", repo_id="alxcspr/stationhouse",
                  repo_type="model", commit_message="curves pack (64 png+csv + overlay)")
print("folder upload ok")
print("CURVES UPLOAD COMPLETE", len(files))
EOF
