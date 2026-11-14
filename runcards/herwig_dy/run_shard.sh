#!/bin/bash
set -euo pipefail
shard=$1
events=$2
keep=${3:-0}
mkdir -p /work/runs/run_${shard}
cd /work/runs/run_${shard}
sed -e "s/:NumberOfEvents 1000/:NumberOfEvents ${events}/" \
    -e "s/:PrintEvent 1000/:PrintEvent ${events}/" \
    -e "s/:Seed 731041/:Seed $((731101+shard))/" /work/Herwig.in > Herwig.in
Herwig read -i "$HERWIGPATH" --repo "$HERWIG_REPO" Herwig.in > read.log 2>&1
mkfifo DYHerwig.hepmc
python3 /work/collect_stream.py DYHerwig.hepmc events.npz --card Herwig.in --shard "$shard" --keep "$keep" > extract.log 2>&1 &
reader=$!
set +e
Herwig run DYHerwig.run -N "$events" --repo "$HERWIG_REPO" > run.log 2>&1
generated=$?
if [ "$generated" -ne 0 ]; then kill "$reader" 2>/dev/null; fi
wait "$reader"
extracted=$?
set -e
rm DYHerwig.hepmc
if [ "$generated" -ne 0 ] || [ "$extracted" -ne 0 ]; then
    echo "Failed shard $shard: generator=$generated reader=$extracted"
    exit 1
fi
python3 - "$events" <<'PY'
import json, sys
from pathlib import Path
d=json.loads(Path('events.json').read_text())
assert d['source_records']==int(sys.argv[1]), d
assert d['source_weight_sum']==d['source_records'], d
assert .2 < d['selected_records']/d['source_records'] < .6, d
print(d['source_records'], d['selected_records'], d['elapsed_seconds'])
PY
