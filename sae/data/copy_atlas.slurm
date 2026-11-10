#!/bin/bash
#SBATCH -J mv-atlas
#SBATCH -p compute
#SBATCH -A pilot_gpu
#SBATCH -t 04:00:00
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -o /data/scratch/apx186/marin32b/logs/mvatlas-%j.out
set -e
mkdir -p /data/scratch/apx186/marin32b/atlas
cp -a /data/SPCS-IOT/marin-32b/atlas/. /data/scratch/apx186/marin32b/atlas/
echo COPY_DONE
find /data/scratch/apx186/marin32b/atlas -type f | wc -l
