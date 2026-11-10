#!/bin/bash
#SBATCH -J a100-80-smoke
#SBATCH -p gpu
#SBATCH -A pilot_gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1
#SBATCH -c 4
#SBATCH --mem=45G
#SBATCH -t 00:10:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/a80smoke-%j.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/a80smoke-%j.err
echo "node $(hostname -f) $(date -u)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
SIF=/data/scratch/apx186/bolmo/sif/bolmo_cu118.sif
export PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser
apptainer exec --nv --env PYTHONUSERBASE="$PYTHONUSERBASE" "$SIF" python3 - <<'EOF'
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
p = torch.cuda.get_device_properties(0)
print(f"name={p.name} cap={p.major}.{p.minor} mem={p.total_memory/2**30:.1f}GiB sms={p.multi_processor_count}")
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
y = (x @ x).sum().item()
print("CUDA OP OK", y)
EOF
echo "A80 SMOKE DONE"
