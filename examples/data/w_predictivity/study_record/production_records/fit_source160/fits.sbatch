#!/bin/bash
#SBATCH -A m4599
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -t 00:30:00
#SBATCH -J wpred_fits
#SBATCH -o /pscratch/sd/b/bassi/jhep_w_predictivity_20261114/logs/fits-%j.log
set -euo pipefail
module load python/3.13-26.11.0
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export WPRED_ROOT=/pscratch/sd/b/bassi/jhep_w_predictivity_20261114
python "$WPRED_ROOT/scripts/fit_pool.py" --root "$WPRED_ROOT" --runs 160 --workers 8
