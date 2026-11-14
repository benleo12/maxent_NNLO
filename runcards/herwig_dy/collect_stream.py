"""Stream generator records into the public compact DY event schema."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyhepmc
from maxent_upgrade.hepmc import collect_dy

parser = argparse.ArgumentParser()
parser.add_argument("pipe")
parser.add_argument("output")
parser.add_argument("--card", required=True)
parser.add_argument("--shard", type=int, required=True)
parser.add_argument("--keep", type=int, default=0)
args = parser.parse_args()
start = time.monotonic()
source = dict(generator="Herwig 7.3.0", shard=args.shard,
              card_sha256=hashlib.sha256(Path(args.card).read_bytes()).hexdigest(),
              full_records="streamed, with explicitly counted subset retained")
with pyhepmc.open(args.pipe, format="hepmc2") as reader:
    if args.keep:
        with pyhepmc.open(str(Path(args.output).with_suffix(".hepmc")), "w") as writer:
            arrays = collect_dy(reader, lepton_pid=11, qed_off=True,
                                selected_writer=writer, selected_limit=args.keep,
                                source=source)
    else:
        arrays = collect_dy(reader, lepton_pid=11, qed_off=True, source=source)
arrays["generator_event_number"] = arrays["event_id"].copy()
arrays["event_id"] += args.shard * 10000000
arrays["shard_id"] = np.full(len(arrays["event_id"]), args.shard, dtype=np.int64)
np.savez_compressed(args.output, **arrays)
metadata = json.loads(str(arrays["record_provenance_json"]))
metadata["elapsed_seconds"] = time.monotonic()-start
metadata["unique_id_rule"] = "generator_event_number + shard_id * 10000000"
Path(args.output).with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
print(json.dumps(metadata), flush=True)
