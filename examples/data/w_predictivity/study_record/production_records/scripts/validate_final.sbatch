#!/bin/bash
#SBATCH -A m4599
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -t 00:05:00
#SBATCH -J wpred_unblind
#SBATCH -o /pscratch/sd/b/bassi/jhep_w_predictivity_20261114/logs/unblind-%j.log
set -euo pipefail
module load python/3.13-26.11.0
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
W=/pscratch/sd/b/bassi/jhep_w_predictivity_20261114
python "$W/scripts/validate.py" "$W" --runs 160 --reference "$W/reference_final" --output "$W/validation_final" --allow-precision-misses
python "$W/scripts/summarize_results.py" "$W/validation_final/RESULTS.json" > "$W/validation_final/READER_SUMMARY.json"
echo VALIDATION_FINAL_COMPLETE
