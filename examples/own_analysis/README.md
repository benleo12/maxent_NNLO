# Higgs decay to two photons in Rivet

Start with inclusive Pythia Higgs events, fit two rapidity moments from NNLOJET and apply photon cuts in Rivet. The fit uses the Higgs rapidity before the photon cuts. The example checks that Rivet obtains the same fiducial cross section and diphoton transverse-momentum bins as direct sums of the fitted event weights, for the central prediction and each scale choice.

## Run the example

From the repository checkout, install the optional HepMC reader and run

```sh
pip install '.[hepmc]'
python examples/own_analysis/run.py /tmp/higgs_rivet_example --generate 1000 --seed 91357
```

Supply a new output directory. The example needs a C++17 compiler, `pythia8-config`, `rivet-config` and Rivet 3 built with HepMC3. It builds the small generator and analysis driver in the output directory. The uncompressed generated sample occupies about 220 MB.

To start from your own positive-weight inclusive Higgs-to-diphoton events instead, run

```sh
python examples/own_analysis/run.py /tmp/my_higgs_analysis --input my_higgs.hepmc
```

This analysis identifies the Higgs through its two final decay photons. The records must retain their Higgs ancestry and cover inclusive production. For this example, disable photon conversion in the shower. The supplied generator uses 13 TeV proton beams, a 125 GeV Higgs, a forced diphoton decay and the Monash tune with showering, multiparton interactions and hadronization.

## Observable and fixed-order inputs

`measure(event, to_GeV)` in `run.py` returns `h_y_abs`, the absolute rapidity of the sum of the two photon four-momenta. It also records the diphoton transverse momentum and the selection decisions. `target_table()` pairs this column with degrees 1 and 2 of the released NNLO absolute-rapidity moments. Both use the same linear map on `[0,4]` with clipping at the endpoints. The central and six varied cross sections each include the Higgs diphoton branching fraction once. The resulting weights are in pb per event. To repeat this fit on a compatible sample, reuse the saved `targets.json` with `maxent-upgrade fit`.

The fiducial selection requires `105 < m_gg < 160` GeV, leading and subleading `pT/m_gg > 0.35` and `0.25`, and both photons inside `|eta| < 2.37` excluding `1.37 < |eta| < 1.52`. The selection has no photon isolation requirement. An additional cut requires `pT_gg > 45` GeV. The event measurement and the Rivet analysis implement these cuts independently and use the same inclusive fit.

## Absolute cross sections and selected files

The HepMC writer preserves the generator weights and appends one named weight for each fit. Each new `GenCrossSection` entry gives the sum of that weight over the events written. A file containing only events passing the photon cuts therefore carries the fiducial cross section. Disjoint files each carry their own partial rate.

The example runs Rivet twice on both the inclusive and filtered files.

- With absolute event weights, each bin sum is already in pb.
- With the usual Rivet normalization, `crossSection()/picobarn/sumOfWeights()` in `finalize()` uses the cross section for each weight stream. It recovers the same bin sums.

The program checks every counter and bin against direct sums for each scale. It also normalizes a filtered file to the inclusive rate on purpose. This would remove the acceptance, and the test detects the error for every scale choice. It checks event association by reversing the supplied ID order and verifies the original weights, four-momenta and vertex connections after writing.

An existing cross-section entry is preserved. An unknown original rate is marked `-1`. The zero errors attached to new `GenCrossSection` entries are placeholders. The separate fitted scale weights provide the scale variations. Rivet's `sumW2` gives the sampling variance at fixed event weights.

## Results

The output directory contains `targets.json`, the input and fitted NPZ samples, full and filtered HepMC files and Rivet histograms in YODA format. `qa.json` records the versions, input and code hashes, direct predictions and all comparisons. In the supplied run, the usual Rivet normalization reproduced the direct sums within the precision of the cross sections stored by HepMC. [Executed comparison](qa_result.json)

The analysis uses the Rivet `AnalysisHandler`, projections, counters and histograms. It can also be adapted to a Rivet plugin. [Rivet 3](https://arxiv.org/abs/1912.05451)
