#!/bin/bash
#SBATCH -A m4599
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -t 00:20:00
#SBATCH -J wpred_timing
#SBATCH -o /pscratch/sd/b/bassi/jhep_w_predictivity_20261114/logs/calibration-%j.log
set -euo pipefail
W=/pscratch/sd/b/bassi/jhep_w_predictivity_20261114
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 "$W/scripts/run_pool.py" --pilot --pilot-tag pilot50k --pilot-points 10000 --pilot-iterations 5 --workers 24
echo CALIBRATION_COMPLETE
