#!/bin/bash
#SBATCH -A m4599
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -t 00:30:00
#SBATCH -J wpred_build
#SBATCH -o /pscratch/sd/b/bassi/jhep_w_predictivity_20261114/logs/build-%j.log
set -euo pipefail
W=/pscratch/sd/b/bassi/jhep_w_predictivity_20261114
LHAPDF_PREFIX=/global/cfs/cdirs/m4599/Projects/sherpa/bassi/software/LHAPDF-6.5.5/local
export OMP_NUM_THREADS=1
cmake -S "$W/source" -B "$W/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_Fortran_COMPILER=gfortran -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_Fortran_FLAGS=-ffree-line-length-none -DLHAPDF_ROOT_DIR="$LHAPDF_PREFIX" \
  -DOPENMP=OFF -DPINEAPPL=OFF -DAPPLFAST=OFF -DDOKAN=OFF \
  -DCMAKE_INSTALL_PREFIX="$W/install"
cmake --build "$W/build" -j32 --target NNLOJET nnlojet_w
mkdir -p "$W/install/bin" "$W/install/lib"
cp "$W/build/NNLOJET" "$W/install/bin/"
cp "$W/build/libnnlojet_core.so" "$W/build/libnnlojet_interface.so" "$W/build/driver/process/W/libnnlojet_w.so" "$W/install/lib/"
sha256sum "$W/install/bin/NNLOJET" "$W/install/lib/"*.so > "$W/build_checksums.txt"
echo BUILD_SUCCESS
