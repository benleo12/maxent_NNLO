#!/bin/bash
#SBATCH -A m4599
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -t 00:10:00
#SBATCH -J wpred_check
#SBATCH -o /pscratch/sd/b/bassi/jhep_w_predictivity_20261114/logs/check-%j.log
set -euo pipefail
W=/pscratch/sd/b/bassi/jhep_w_predictivity_20261114
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$W/technical"
python3 "$W/scripts/check_pointwise.py" "$W"
echo POINTWISE_COMPLETE
