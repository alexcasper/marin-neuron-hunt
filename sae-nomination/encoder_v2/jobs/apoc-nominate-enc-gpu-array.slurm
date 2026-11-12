#!/bin/bash
#SBATCH -J nenc64k-gpu
#SBATCH -p gpu
#SBATCH -A pilot_gpu
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH -a 57
#SBATCH -c 4
#SBATCH --mem=45G
#SBATCH -t 02:00:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/gpu-nenc-%A_%a.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/gpu-nenc-%A_%a.err
# Encoder-attribution nomination, GPU v2 (ops_sh 2026-11-09). Replaces the
# 13:26 draft, which requested no GPU (no --gres), ran apptainer without --nv,
# and called v1, whose code hard-codes device="cpu".
#
# Pattern = apoc-64k-sweep.slurm (the job that TRAINED these 64k SAEs on
# A100-40): -p gpu + a100-pcie-40gb gres, -c 4/45G, apptainer --nv bolmo_cu118
# SIF + PYTHONUSERBASE=qwen35/pyuser (eai-sparsify), layer parquet staged to
# node-local /tmp. (ops_sh's sae/*.sh copies reference a sae-venv that never
# existed on scratch -- do not use them as the env template.)
#
# Script = nominate_enc_pass_v2.py: vectorized attribution (v1's per-token
# Python loop was the bottleneck) + FIXED shuffle control (v1 searchsorted on an
# unsorted ctrl_idx -> most control latents never accumulated; L57's
# "permuted p50 0.000" is that artifact). Equivalence to v1's real-data math
# tested (test_nominate_v2.py). Outputs -> layer=NN/nominate_enc_v2/ (v1 outputs
# untouched). Same 200k-token sample, K=5 perms, 8000-latent cap as v1.
#
# GATE ORDER:
#  1. Smoke (this default, -a 57). Pass = SEL-CHECK passes; real-pass stats
#     match v1 L57 (top1_ge_0.5 == 40, top1_p50 ~0.064, tokens 200,704);
#     control "rows" > 0; note wall_s for sizing.
#  2. Full: sbatch -a 0-63%8 apoc-nominate-enc-gpu-array.slurm
#     (all 64, so every layer has the fixed control incl. 00/02/57).
set -euo pipefail
SCR=/data/scratch/apx186/marin32b
LL=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
echo "node $(hostname -f) nominate-enc v2 layer $LL task $SLURM_ARRAY_TASK_ID"
test -f "$SCR/nominate_enc_pass_v2.py" || { echo "MISSING $SCR/nominate_enc_pass_v2.py (deploy first)"; exit 3; }

SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
export PYTHONPATH=$SCR
STAGED=$(mktemp -d /tmp/marin_nenc_${SLURM_ARRAY_TASK_ID}_XXXXXX)
trap 'rm -rf "$STAGED"' EXIT
cp "$SCR/atlas-resorted/layer$LL-tokensorted.parquet" "$STAGED/"
df -h /tmp | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

apptainer exec --nv -B /data/scratch/apx186:/data/scratch/apx186 \
  -B "$STAGED:$STAGED" \
  --env PYTHONUSERBASE="$PYTHONUSERBASE" --env PYTHONPATH="$PYTHONPATH" "$SIF" \
  env NOM_LAYER="$SLURM_ARRAY_TASK_ID" NOM_MODE=attribution \
      NOM_PARQ="$STAGED/layer$LL-tokensorted.parquet" SLURM_CPUS_PER_TASK=4 \
  python3 -u "$SCR/nominate_enc_pass_v2.py"

echo "TASK DONE layer $LL"
