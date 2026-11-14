#!/bin/bash
#SBATCH -A m4599
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -t 00:10:00
#SBATCH -J wpred_poolP1
#SBATCH -o /pscratch/sd/b/bassi/jhep_w_predictivity_20261114/logs/poolP1-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
W=/pscratch/sd/b/bassi/jhep_w_predictivity_20261114
python3 "$W/scripts/aggregate.py" --root "$W" --group V --replicas 64 \
  --channel-counts '{"LO":128,"R":512,"V":128,"RR":1024,"RV":256,"VV":64}' \
  --output "$W/reference_p1"
python3 "$W/scripts/precision_only.py" "$W/reference_p1" --output "$W/reference_p1/PRECISION_ONLY.json"
