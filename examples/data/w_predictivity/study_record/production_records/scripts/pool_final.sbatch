#!/bin/bash
#SBATCH -A m4599
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -t 00:10:00
#SBATCH -J wpred_poolFinal
#SBATCH -o /pscratch/sd/b/bassi/jhep_w_predictivity_20261114/logs/poolFinal-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python3 /pscratch/sd/b/bassi/jhep_w_predictivity_20261114/scripts/pool_final.py /pscratch/sd/b/bassi/jhep_w_predictivity_20261114
