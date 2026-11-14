# From an event record to fitted weights

The optional Drell–Yan adapter reads HepMC2 or HepMC3 records, extracts the released
13 TeV fiducial selection, and attaches fitted weights by unique event number.
The core fitting workflow still needs only NumPy.

```bash
python -m pip install '.[hepmc]'
python examples/hepmc_drell_yan.py my_events.hepmc dy_records --qed-off
```

Use `--lepton-pid 13` for muons (electrons are the default). The explicit
`--qed-off` declaration means lepton QED radiation was disabled **at generation**.
The adapter does not switch it off retrospectively. The released table uses
pre-QED leptons, so bare or dressed leptons from a QED shower are different inputs.

The adapter selects exactly one prompt opposite-sign pair of the requested
flavour. It rejects tau and hadron decay ancestry, permits incoming beam protons,
and fails on ambiguous prompt pairs. It applies `66 < mll < 116` GeV,
`pT > 27` GeV and `|eta| < 2.5` for both leptons. Nominal generator weights must
be finite and positive. It preserves their values and unique event numbers,
converts extracted momenta to GeV, and keeps both lepton four-vectors so later
observables can be defined without extracting the records again. There is no
jet clustering in this adapter.

The output directory contains:

- `prior.npz`: selected arrays, original weights, IDs and extraction provenance;
- `posterior.npz`: the same arrays, all fitted variation weights and fit provenance;
- `posterior.hepmc`: selected full records with named central and coherent variation weights;
- `statistics.json`: global effective statistics and concentration.

The new HepMC weight **`maxent_pb_per_event` is already an absolute contribution
in pb**. Its sum is the fitted fiducial cross section. Do not multiply it by the
generator cross section again. Original event weights, particle momenta, vertex
connections and metadata remain in the selected records. The original generator
cross-section metadata is preserved; the new rate and units are separately
recorded in the `maxent_upgrade` run attribute. Every available variation is also attached as a named weight, for example
`maxent_scale_001_pb_per_event`, `maxent_stat_001_pb_per_event` and
`maxent_window_001_pb_per_event`; PDF members appear when supplied. Index 001 is
the first entry of that source in `posterior.npz`. Each weight vector carries
its own total rate, recorded in the run metadata. Build downstream histograms
for each named vector before forming an envelope or covariance; no per-event
envelope weights are constructed. The `variations_pb=` mapping of `attach_weights`
can be used to attach only a chosen subset.

A mismatched source hash, missing ID or duplicate ID aborts attachment. The output
is written atomically after checking all requested IDs. Files containing different
run headers must be handled one run at a time, because a HepMC3 writer serializes
one run header. If runs reuse event numbers, assign a unique mapping when combining
them and retain that mapping.

## Stream directly from a generator

`collect_dy` accepts a record iterator, retaining compact numeric columns instead
of full event records. A nonseekable pipe needs an explicit input format:

```python
import numpy as np
import pyhepmc
from maxent_upgrade.hepmc import collect_dy

with pyhepmc.open("events.fifo", format="hepmc2") as records, pyhepmc.open("selected.hepmc", "w") as writer:
    events = collect_dy(records, lepton_pid=11, qed_off=True,
                        selected_writer=writer, selected_limit=200,
                        source={"generator": "name and version", "card_sha256": "..."})
np.savez_compressed("prior.npz", **events)
```

Use `format="hepmc3"` for that format. Automatic format detection requires a
seekable file. The optional writer above preserves the first 200 selected full
records; the arrays retain **all** selected events. Metadata records all-event
and selected counts and weight sums. Check these against the generator's completed
run. A cross section found in the last record is saved as metadata, not used as a
final integration result without checking the generator summary.

`dy_observables(event, ...)` exposes the single-record extraction step.
`extract_dy(path, ...)` additionally fingerprints a regular source file.
`attach_weights(path, output, event_ids, weights_pb, ...)` works with a permuted
array order because association uses event IDs. For a new process, replace the
observable extraction and selection; the generic array-based fit is unchanged.

## Record round-trip check

On 13 November 2026, the adapter read a real Herwig 7.3 pilot: 1000 generated
records, total nominal weight 1000, and 365 selected electron pairs. No prompt-pair
ambiguities occurred. The first 200 selected records were extracted, fitted to
three released NNLO mass moments, and written with reversed array/ID order.
Every original ID, weight, particle four-momentum, PDG/status and vertex connection
was preserved; attached weights agreed with the fit to relative tolerance `1e-14`.
This checks record handling and weight association. A 200-event mass fit does not
validate the full DY prediction or establish generator independence by itself.

The pilot used Born `pp -> e+ e-` at 13 TeV, `50 < mee < 2200` GeV, CT18NNLO for
the hard process, QCD showering with default hard matrix-element corrections,
hadronization and MPI. Lepton QED radiation was disabled. The tests additionally
cover unit conversion, secondary leptons, duplicate/missing IDs and preserved
run/event attributes with deterministic small records.
