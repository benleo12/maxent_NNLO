#!/bin/bash
#SBATCH -A m4599
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 00:30:00
#SBATCH -J wpred_fast
#SBATCH -o /pscratch/sd/b/bassi/jhep_w_predictivity_20261114/logs/build-%j.log
set -euo pipefail
W=/pscratch/sd/b/bassi/jhep_w_predictivity_20261114
LHAPDF_PREFIX=/global/cfs/cdirs/m4599/Projects/sherpa/bassi/software/LHAPDF-6.5.5/local
export OMP_NUM_THREADS=1
cmake -S "$W/source" -B "$W/build_v4" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_Fortran_FLAGS=-ffree-line-length-none -DLHAPDF_ROOT_DIR="$LHAPDF_PREFIX" \
  -DOPENMP=OFF -DPINEAPPL=OFF -DAPPLFAST=OFF -DDOKAN=OFF \
  -DCMAKE_INSTALL_PREFIX="$W/install_v4"
cmake --build "$W/build_v4" -j32 --target NNLOJET nnlojet_w
mkdir -p "$W/install_v4/bin" "$W/install_v4/lib"
cp "$W/build_v4/NNLOJET" "$W/install_v4/bin/"
cp "$W/build_v4/libnnlojet_core.so" "$W/build_v4/libnnlojet_interface.so" "$W/build_v4/driver/process/W/libnnlojet_w.so" "$W/install_v4/lib/"
sha256sum "$W/install_v4/bin/NNLOJET" "$W/install_v4/lib/"*.so > "$W/build_v4_checksums.txt"
echo BUILD_SUCCESS
cd "$W/technical"
python3 "$W/scripts/check_pointwise.py" "$W"
python3 "$W/scripts/run_pool.py" --pilot --pilot-tag pilot_fastpath --install-subdir install_v4 --workers 24
python3 "$W/scripts/check_pilots.py" "$W" --pilot-tag pilot_fastpath
python3 "$W/scripts/run_pool.py" --pilot --pilot-tag pilot50k_final --install-subdir install_v4 --pilot-points 10000 --pilot-iterations 5 --workers 24
echo FINAL_CALIBRATION_COMPLETE
