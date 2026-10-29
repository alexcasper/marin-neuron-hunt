#!/bin/bash
# Upload the 20 already-good SAE layers to HF repo alxcspr/stationhouse.
# Runs on APOC login/compute (egress proven). HF_TOKEN comes from the
# environment — NEVER echoed, never written to the repo or logs.
# Usage: HF_TOKEN=hf_... sbatch -p computeshort -t 04:00:00 apoc-hf-upload.sh
set -uo pipefail
: "${HF_TOKEN:?HF_TOKEN must be set in the job environment}"
REPO=alxcspr/stationhouse
TYPE=model
SCR=/data/scratch/apx186/marin32b
SRC=$SCR/sae-32k
LOGS=$SCR/logs
mkdir -p "$LOGS"
LAYERS="00 01 02 03 04 05 06 07 08 09 10 11 37 38 39 40 41 56 57 58"

pip install --quiet "huggingface_hub>=0.25" 2>&1 | tail -1

export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
python3 - <<'EOF'
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
info = api.repo_info("alxcspr/stationhouse", repo_type="model")
print("repo exists (no creation attempted): private =", info.private)
EOF

LAYERS="00 01 02 03 04 05 06 07 08 09 10 11 37 38 39 40 41 56 57 58"
FAILED=""
for LL in $LAYERS; do
  D="$SRC/sae-32k/layer=$LL"
  [ -d "$D" ] || { echo "MISSING dir layer=$LL"; FAILED="$FAILED $LL"; continue; }
  echo "=== uploading layer=$LL"
  for f in stats.json census.parquet train_log.jsonl sha256.txt; do
    [ -f "$D/$f" ] || { echo "MISSING file layer=$LL/$f"; FAILED="$FAILED $LL/$f"; continue; }
    python3 - "$D/$f" "sae-32k/layer=$LL/$f" <<'EOF' || FAILED="$FAILED $LL/$f"
import sys, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(path_or_fileobj=sys.argv[1], path_in_repo=sys.argv[2],
                repo_id="alxcspr/stationhouse", repo_type="model",
                commit_message=f"layer {sys.argv[2].split('/layer=')[1][:2]}: {sys.argv[2].rsplit('/',1)[1]}")
EOF
  done
  # final checkpoint dir (bf16 safetensors + cfg)
  python3 - "$D" "sae-32k/layer=$LL" <<'EOF' || FAILED="$FAILED $LL/final"
import sys, glob, os
from huggingface_hub import HfApi
d, rin = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
ckpt = glob.glob(f"{d}/sae_L*_final")
if not ckpt:
    print("no final ckpt dir"); sys.exit(1)
for f in glob.glob(ckpt[0] + "/*"):
    name = os.path.basename(f)
    api.upload_file(path_or_fileobj=f, path_in_repo=f"{rin}/sae_L{rin.split('layer=')[1][:2]}_final/{name}",
                    repo_id="alxcspr/stationhouse", repo_type="model",
                    commit_message=f"layer {rin.split('layer=')[1][:2]}: final ckpt {name}")
EOF
done

# README at repo root
[ -f "$SRC/hf-README.md" ] && python3 - "$SRC/hf-README.md" "README.md" <<'EOF'
import sys, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(path_or_fileobj=sys.argv[1], path_in_repo=sys.argv[2],
                repo_id="alxcspr/stationhouse", repo_type="model",
                commit_message="provenance README")
EOF

echo "UPLOAD COMPLETE. failed: ${FAILED:-none}"
