#!/bin/bash
#SBATCH -J probedec
#SBATCH -p gpu
#SBATCH -A pilot_gpu
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH -c 4
#SBATCH --mem=30G
#SBATCH -t 00:20:00
#SBATCH -o /data/scratch/apx186/marin32b/logs/probedec-%j.out
#SBATCH -e /data/scratch/apx186/marin32b/logs/probedec-%j.err
apptainer exec --nv -B /data/scratch/apx186:/data/scratch/apx186 --env PYTHONUSERBASE=/data/scratch/apx186/qwen35/pyuser --env PYTHONPATH=/data/scratch/apx186/marin32b /data/scratch/apx186/bolmo/sif/bolmo_cu118.sif python3 -u /data/scratch/apx186/marin32b/probe_decode6.py
