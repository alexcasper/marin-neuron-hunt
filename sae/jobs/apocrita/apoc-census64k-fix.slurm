#!/bin/bash
#SBATCH -J census64k-fix
#SBATCH -p gpu
#SBATCH -A pilot_gpu
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH -c 4
#SBATCH --mem=45G
#SBATCH -t 01:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/census64kfix-%j.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/census64kfix-%j.err
# Census fix for the 64k probe: stats step failed because the final ckpt is
# named sae_LNNw64_final_s<step> (val-suffix), not sae_LNNw64_final.
set -uo pipefail
SCR=/data/scratch/apx186/marin32b
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
export PYTHONPATH=$SCR
STAGED=$(mktemp -d /tmp/marin_c64k_XXXXXX)
trap 'rm -rf "$STAGED"' EXIT
cp "$SCR/atlas/pilot/fineweb-10m-sparse/summaries/neuron_scale.parquet" "$STAGED/"
for LL in 45 57 63; do
  OUT=$SCR/64k-probe/layer=$LL
  CK=$(ls -d "$OUT"/sae_L${LL}w64_final_s* 2>/dev/null | head -1)
  [ -n "$CK" ] || { echo "no final for $LL"; continue; }
  cp "$SCR/atlas-resorted/layer$LL-tokensorted.parquet" "$STAGED/"
  apptainer exec --nv -B /data/scratch/apx186:/data/scratch/apx186 \
    -B "$STAGED:$STAGED" \
    --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$PYTHONPATH" "$SIF" \
    python3 -u "$SCR/sae_stats.py" "$CK" "$STAGED/layer$LL-tokensorted.parquet" \
    200704 "$OUT/census" "$STAGED/neuron_scale.parquet" \
    > "$OUT/stats.json" 2>"$OUT/stats.err" \
    && echo "CENSUS OK $LL ($CK)" || echo "CENSUS FAILED $LL (see stats.err)"
  rm -f "$STAGED/layer$LL-tokensorted.parquet"
done
( cd "$SCR/64k-probe" && for d in layer=*; do ( cd "$d" && find . -type f -exec sha256sum {} \; > sha256.txt ); done )
echo "CENSUS64K FIX DONE"
