"""Extract a QED-off Drell--Yan HepMC sample, fit it and attach selected weights.

Install the optional reader with: pip install '.[hepmc]'
Run: python examples/hepmc_drell_yan.py input.hepmc output_directory --qed-off

Use a sufficiently populated positive-weight sample with the released DY
selection. The output HepMC contains the selected events and preserves their
original weights. Central and coherent variation weights are appended in pb.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maxent_upgrade import upgrade_with_uncertainties, weight_diagnostics
from maxent_upgrade.hepmc import attach_weights, extract_dy
from maxent_upgrade.provenance import canonical_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--lepton-pid", type=int, choices=[11, 13], default=11)
    parser.add_argument("--qed-off", action="store_true", required=True,
                        help="confirm that lepton QED radiation was disabled during generation")
    parser.add_argument("--bootstrap", type=int, default=20)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    events = extract_dy(args.input, lepton_pid=args.lepton_pid, qed_off=args.qed_off)
    np.savez_compressed(output / "prior.npz", **events)
    result = upgrade_with_uncertainties(events, "dy", n_boot=args.bootstrap,
                                        progress=lambda name, fit: print(f"{name}: {fit.summary()}", flush=True))
    record = json.loads(str(events["record_provenance_json"]))
    result.provenance["event_record"] = record
    absolute = result.arrays()
    np.savez_compressed(output / "posterior.npz", **events, **absolute)
    variations = {f"maxent_{source}_{index + 1:03d}_pb_per_event": vector
                  for source in result.weights
                  for index, vector in enumerate(absolute["weight_" + source])}
    attach_weights(args.input, output / "posterior.hepmc", events["event_id"], absolute["weight_upgraded"],
                   provenance=result.provenance, expected_source_sha256=record["source"]["sha256"], variations_pb=variations)
    (output / "statistics.json").write_text(canonical_json(weight_diagnostics(result.central.weights, events["weight"])) + "\n")
    print(f"Saved selected records, fitted arrays and all uncertainty weights in {output}")
    print("All new HepMC weights are in pb per event. Do not multiply it by the generator cross section.")


if __name__ == "__main__":
    main()
