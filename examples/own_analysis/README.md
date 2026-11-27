# A Higgs shower through your analysis and Rivet

This example runs a real Pythia `gg -> H -> gamma gamma` sample through custom
observable extraction, a fit to released NNLO rapidity moments, HepMC weight
attachment and the **actual Rivet library**. It checks every fitted weight against
direct event sums, including after filtering the record file. It also runs a
wrong normalization deliberately and verifies that the discrepancy is detected.

This is an interoperability example with a small shower sample. It does not
reproduce the paper's production, establish NNLO accuracy for photon cuts or
fit a measured distribution. The fixed-order inputs are physical released NNLOJET
results; the generated sample is a small real shower, not synthetic kinematics.

## Run

Install the Python package and optional reader from the repository checkout:

```sh
python -m pip install '.[hepmc]'
python examples/own_analysis/run.py /tmp/higgs_rivet_example --generate 1000 --seed 91357
```

The output directory must be new. This additionally needs a C++17 compiler,
`pythia8-config`, `rivet-config` and Rivet 3 built with HepMC3. Pythia runs with
one stream and no parallel event production. A thousand complete shower records
occupy roughly 220 MB before compression on the tested installation. The script
compiles its small generator and analysis driver into the output directory;
it does not modify the installed libraries.

If you already have positive-weight inclusive Higgs-to-diphoton HepMC3 records,
Pythia is unnecessary:

```sh
python examples/own_analysis/run.py /tmp/my_higgs_analysis --input my_higgs.hepmc
```

The sample must retain its Higgs/photon ancestry, contain exactly two Higgs-decay
final photons per event, and cover the inclusive production phase space. Photon
conversion in the shower must be disabled for this particular observable.
The generated example uses Pythia's Monash tune, shower, MPI and hadronization,
13 TeV proton beams, nominal Higgs mass 125 GeV, and a forced diphoton decay.
The generator source fixes all nondefault settings and the random seed.
Bitwise replay requires the same generator version and build; another compatible
build should pass the normalization checks on its own events.

The C++ driver uses Rivet's `AnalysisHandler` and ordinary projections, named-weight
multiplexing, counters and histograms. This also allows execution on installations
whose `rivet` Python launcher references an unavailable historical Python version.
It is not a replacement histogrammer emulating Rivet. The supplied C++ analysis can
also be adapted to a normal Rivet plugin.

## What the user supplies

`measure(event, to_GeV)` in `run.py` defines an event column named `h_y_abs` from
the sum of the two Higgs-decay photon four-vectors. This is an inclusive Higgs
rapidity observable; no photon cuts enter its fit. It also stores the diphoton
transverse momentum and flags for later selections.

`target_table()` takes degrees 1 and 2 of the released NNLO `y_abs` tower,
renames that observable to `h_y_abs`, and retains its linear clipped map on
`[0,4]`. It imposes exactly these two degrees. Six scale members carry their own
rapidity moments and inclusive rates. Each rate is multiplied by
`BR(H -> gamma gamma) = 0.00227`, once, so every fitted vector already describes
the diphoton channel in pb. The written `targets.json` can be used directly with
`maxent-upgrade fit prior.npz targets.json posterior.npz`.

The photon selection requires `105 < m_gg < 160` GeV, leading/subleading
`pT/m_gg > 0.35/0.25`, and both photons inside `|eta| < 2.37` excluding
`1.37 < |eta| < 1.52`. The additional cut is `pT_gg > 45` GeV. There is **no
isolation requirement** in this deliberately simple user analysis; it is not
called an ATLAS fiducial implementation. The Python callback and Rivet analysis
independently implement these same definitions. No selection is refitted.

## Normalization that survives HepMC and filtering

`weight_upgraded` and every `weight_scale` row are absolute pb per event. The
HepMC file retains original generator weights and adds one named stream per fitted
vector. Its `GenCrossSection` entries for those streams are their respective sums
over the records actually written. A filtered file therefore carries its filtered
rate, without changing the individual fitted weights or their original fit.

The example runs both of these valid Rivet analyses:

- **Absolute:** fill with the named weight and perform no final cross-section
  rescaling. Bin sums already have units of pb.
- **Conventional:** apply `crossSection()/picobarn/sumOfWeights()` in `finalize()`.
  Rivet uses the cross section associated with each named stream. Because the
  attached rate describes this file, this preserves the same absolute result.

Each mode checks the all-record, photon-selected and additional-cut counters,
and all six diphoton-pT bins, for the central member and every scale member.
It runs on both the inclusive record file and a file retaining only the photon-
selected records. Attachment receives deliberately reversed IDs and arrays to
check association by ID. A full-record audit checks original weights, particle
four-vectors and vertex connections.

`fiducial_wrong_full_rate.yoda` is an intentional negative control: it normalizes
the filtered file to the inclusive rate, which incorrectly removes the acceptance.
The check must detect this for every member. Do not copy that override into an
analysis. Likewise, each disjoint file shard needs its own partial fitted rate;
do not assign the full parent rate to every shard or average their absolute sums.

Original cross-section slots retain their input values. A scalar original
cross section applies to the original weight slots; an absent original rate is
marked `-1` (unknown). No uncertainty is inferred for the new `GenCrossSection`
error fields: they are zero placeholders. The scale vectors remain separate;
Rivet's event `sumW2` describes fixed-weight sampling, not response of the fit.

## Outputs and executed check

`qa.json` records versions, input/code hashes, all direct predictions, every
comparison and the negative control. `targets.json`, `prior.npz` and
`posterior.npz` expose the numerical inputs and fitted weights. Full and filtered
HepMC files, high-precision YODA files, build logs and generation logs make the
round trip inspectable. The repository's `qa_result.json` is the compact record
of an executed run; it contains no full event archive.

The executed 1,000-event example used Pythia 8.310, Rivet 3.1.10 and pyhepmc
2.16.1. Central inclusive, photon-selected and additional-cut rates were
0.08958710633, 0.04775023874 and 0.01740847148 pb. Raw Rivet sums agree at floating-
point precision. Conventional normalization agrees within `5e-11` pb, limited
by HepMC's serialized cross-section precision. Every named variation passes.

Rivet's analysis and multiweight design is described in
[Rivet 3, SciPost Phys. 8 (2020) 026](https://arxiv.org/abs/1912.05451).
