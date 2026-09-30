#!/bin/bash
# Diphoton prior v5 = CKKW-L-merged MG5 (q qbar -> aa, q g -> aa q) + Pythia gg box:
#   1. combine the 1M-event pieces of each merged slice (aa_merge_slices.py)
#   2. solve with the v5 slice set            -> aa_eventlevel_weights_v5.npz
#   3. figures for v5 (main set, no suffix)   -> fig_aa_<key>.pdf
#   4. figures for the v4 stress case         -> fig_aa_<key>_v4.pdf  (from the kept v4 weights)
# usage: bash aa_v5_pipeline.sh [merge|solve|fig|priors|fig4|all]
set -eu
PROJ="${PROJ:-$(cd "$(dirname "$0")" && pwd)}"      # the analysis tree (this directory by default)
PY="${PY:-python3}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
W=$HOME/aa_workdir; M=$W/merged; B=$W/box
mkdir -p $W/v5
step=${1:-all}

if [ "$step" = merge ] || [ "$step" = all ]; then
  for S in A B C D; do
    pieces=$(ls $M/aa_prior_v5mg_${S}.npz $M/aa_prior_v5mg_${S}[0-9].npz 2>/dev/null || true)
    [ -n "$pieces" ] || { echo "no pieces for slice $S"; continue; }
    $PY $PROJ/aa_merge_slices.py $W/v5/aa_prior_v5_mg${S}.npz $pieces
  done
  for s in a b c d e; do
    [ -f $B/aa_prior_v5box_$s.npz ] && cp -p $B/aa_prior_v5box_$s.npz $W/v5/aa_prior_v5_box$s.npz
  done
  ls -la $W/v5/
fi

V5=$(ls $W/v5/aa_prior_v5_*.npz | tr '\n' ',' | sed 's/,$//')
export AA_PRIOR="$V5"
echo "AA_PRIOR=$AA_PRIOR"

if [ "$step" = solve ] || [ "$step" = all ]; then
  cd $PROJ && AA_WEIGHTS_OUT=aa_eventlevel_weights_v5.npz $PY aa_eventlevel_solve.py 2>&1 | tee $W/v5/solve_v5.log
fi
if [ "$step" = fig ] || [ "$step" = all ]; then
  cd $PROJ && AA_WEIGHTS=aa_eventlevel_weights_v5.npz AA_FIG_SUFFIX="" $PY fig_aa_eventlevel.py 2>&1 | tee $W/v5/fig_v5.log
fi
if [ "$step" = priors ] || [ "$step" = all ]; then
  # the two-priors figure (v4 stress case vs v5), needs both weight files
  cd $PROJ && AA_PRIOR_V5="$V5" AA_WEIGHTS_V5=aa_eventlevel_weights_v5.npz $PY fig_aa_priors.py 2>&1 | tee $W/v5/fig_priors.log
fi
if [ "$step" = fig4 ] || [ "$step" = all ]; then
  # the v4 stress case: its own slice set and its own (kept) weights
  cd $PROJ && AA_PRIOR="" AA_PRIOR_SET=v4 AA_WEIGHTS=aa_eventlevel_weights_v4.npz AA_FIG_SUFFIX="_v4" $PY fig_aa_eventlevel.py 2>&1 | tee $W/v5/fig_v4.log
fi
echo "pipeline step '$step' done  $(date)"
