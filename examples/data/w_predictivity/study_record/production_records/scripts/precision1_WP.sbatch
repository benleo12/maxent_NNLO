#!/bin/bash
#SBATCH -A m4599
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH -t 00:30:00
#SBATCH -J wpred_p1WP
#SBATCH -o /pscratch/sd/b/bassi/jhep_w_predictivity_20261114/logs/production-%j.log
set -euo pipefail
W=/pscratch/sd/b/bassi/jhep_w_predictivity_20261114
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
srun --cpu-bind=none bash -c 'python3 /pscratch/sd/b/bassi/jhep_w_predictivity_20261114/scripts/run_pool.py --install-subdir install_v6 --pin-workers --workers 128 --shard "$SLURM_PROCID" --shards "$SLURM_NTASKS" --manifest /pscratch/sd/b/bassi/jhep_w_predictivity_20261114/precision1_WP/manifest.json'
echo PRODUCTION_COMPLETE
