#!/bin/bash
#SBATCH -J hf-upload
#SBATCH -p computeshort
#SBATCH -A pilot_gpu
#SBATCH -t 04:00:00
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -o /data/scratch/apx186/marin32b/logs/hfupload-%j.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/hfupload-%j.err
# Upload the 20 already-good SAE layers to private HF repo alxcspr/stationhouse.
# env_apoc adaptation: huggingface_hub comes from the SIF+userbase (no pip on
# node), egress preflight (per-node egress is unreliable — fail loudly, do not
# "complete" with all-FAILED), fixed layer/README path bugs from ops_sh draft.
# HF_TOKEN comes from the job environment — NEVER echoed or written anywhere.
set -uo pipefail
: "${HF_TOKEN:?HF_TOKEN must be set in the job environment}"
REPO=alxcspr/stationhouse
SCR=/data/scratch/apx186/marin32b
SRC=$SCR/sae-32k
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
PY=(apptainer exec --env PYTHONUSERBASE="$PYTHONUSERBASE" "$SIF" python3)

echo "node $(hostname -f) $(date -u)"
# ---- egress preflight: fail the job (not per-file) if this node can't reach HF
code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 30 -L \
  "https://huggingface.co/api/models/${REPO}" || echo curlfail)
echo "preflight huggingface.co: $code"
case "$code" in
  200|401|403|404) ;; # reachable (repo may 404 until created; auth'd codes fine)
  *) echo "EGRESS BROKEN on this node (code=$code) — resubmit to another node"; exit 3 ;;
esac
"${PY[@]}" -c "import huggingface_hub; print('hub', huggingface_hub.__version__)" || exit 3

# ---- create private repo (STOP-detect: fine-grained token write perms) ----
"${PY[@]}" - <<'EOF' || { echo "CREATE_REPO_FAILED — token likely lacks write/create perm; STOP"; exit 4; }
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo("alxcspr/stationhouse", repo_type="model", private=True, exist_ok=True)
who = api.whoami()
print("repo ready (private), authenticated as:", who.get("name"))
EOF

LAYERS="00 01 02 03 04 05 06 07 08 09 10 11 37 38 39 40 41 56 57 58"
FAILED=""
for LL in $LAYERS; do
  D="$SRC/layer=$LL"
  [ -d "$D" ] || { echo "MISSING dir layer=$LL"; FAILED="$FAILED $LL"; continue; }
  echo "=== uploading layer=$LL"
  for f in stats.json census.parquet train_log.jsonl sha256.txt; do
    [ -f "$D/$f" ] || { echo "MISSING file layer=$LL/$f"; FAILED="$FAILED $LL/$f"; continue; }
    "${PY[@]}" - "$D/$f" "sae-32k/layer=$LL/$f" <<'EOF' || FAILED="$FAILED $LL/$f"
import sys, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(path_or_fileobj=sys.argv[1], path_in_repo=sys.argv[2],
                repo_id="alxcspr/stationhouse", repo_type="model",
                commit_message=f"layer {sys.argv[2].split('/layer=')[1][:2]}: {sys.argv[2].rsplit('/',1)[1]}")
print("ok", sys.argv[2])
EOF
  done
  # final checkpoint dir (bf16 safetensors + cfg)
  "${PY[@]}" - "$D" "sae-32k/layer=$LL" <<'EOF' || FAILED="$FAILED $LL/final"
import sys, glob, os
from huggingface_hub import HfApi
d, rin = sys.argv[1], sys.argv[2]
api = HfApi(token=os.environ["HF_TOKEN"])
ckpt = glob.glob(f"{d}/sae_L*_final")
if not ckpt:
    print("no final ckpt dir"); sys.exit(1)
for f in sorted(glob.glob(ckpt[0] + "/*")):
    name = os.path.basename(f)
    api.upload_file(path_or_fileobj=f, path_in_repo=f"{rin}/sae_L{rin.split('layer=')[1][:2]}_final/{name}",
                    repo_id="alxcspr/stationhouse", repo_type="model",
                    commit_message=f"layer {rin.split('layer=')[1][:2]}: final ckpt {name}")
    print("ok", name)
EOF
done

# README at repo root (staged at $SCR/hf-README.md)
[ -f "$SCR/hf-README.md" ] && "${PY[@]}" - "$SCR/hf-README.md" "README.md" <<'EOF'
import sys, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_file(path_or_fileobj=sys.argv[1], path_in_repo="README.md",
                repo_id="alxcspr/stationhouse", repo_type="model",
                commit_message="provenance README")
print("ok README.md")
EOF

echo "UPLOAD COMPLETE. failed: ${FAILED:-none}"
