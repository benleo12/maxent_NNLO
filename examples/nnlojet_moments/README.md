# Calculate moments with NNLOJET

A new calculation needs a working NNLOJET installation with the [moment patch for v1.0.2](../../nnlojet_patch/generic/README.md), its process library and the PDFs specified in the card. The example measures the absolute Higgs rapidity in inclusive gluon fusion. Its [run card](higgs.run) specifies the process, all six NNLO contributions, PDFs, scales and integration settings. The [calculation file](calculation.json) books the first two normalized moments of the NNLOJET observable `abs_yh` on a linear map from 0 to 4. The same measurement on the shower events uses the `y_abs` column. Values above the map endpoint are clipped for the polynomial and remain in the cross section.

The checkout includes a compatible inclusive Higgs shower sample. Run

```sh
maxent-upgrade match examples/data/applications/ggh_prior.npz examples/nnlojet_moments/calculation.json upgraded.npz \
  --nnlojet /path/to/bin/NNLOJET --work-dir higgs_runs --jobs 4
maxent-upgrade check upgraded.npz --selection
```

To use your own prior, replace the first argument with its path. It must contain the absolute rapidity of the Higgs before cuts on its decay products. This card supplies a production rate. For a forced decay, include the appropriate branching fraction once when converting that rate into a prediction for the decay channel. The output will identify the saved table. It can be used with `fit` on a second shower sample.

## Change the process or measurements

1. Supply a physical run card for a process implemented in your installation. Choose its infrared-safe observables, parton distributions, scales and fiducial cuts. Each requested scalar observable must be defined on every contributing configuration that passes the cuts.
2. In `axes`, map each column in the prior to its registered NNLOJET observable name. For a jet observable defined to be zero in events with too few jets, you can supply `{"observable":"ptj2","undefined":0}` as its entry. Use the identical convention in the prior. Without an explicit value, an undefined observable stops the integration. Define the range, map and any recoil window in `config`. The same definitions apply to the integration and the fit. In `degrees`, supply the desired degrees, including sparse lists or ordered pairs for products. [File formats and a product-moment example](../../docs/NEW_PROCESSES.md)
3. List independent seeds for each contribution. The driver assigns disjoint seed ranges to different contributions. The scale names default to the SCALES lines in the run card. To override them, give `scale_labels` in the same order, central first. The native output unit defaults to fb and is converted to pb. If the installation writes a different unit, set `output_unit` accordingly.

The driver adds integrated histograms to the run card and executes each contribution and seed in its own directory. It averages equal-statistics production runs within a contribution, sums the contributions and propagates the correlated numerator and denominator errors into the normalized moments. Use enough independent runs for the desired numerical precision. `--jobs` sets the number of simultaneous NNLOJET processes. Set `OMP_NUM_THREADS` for the cores available to each process.

Repeat the command with the same calculation and work directory to reuse completed runs after an interruption. List external run-card inputs under `dependencies`. Relative inputs are copied to the same relative paths in each run directory. To assign a different local path, supply a mapping such as `{"inputs/grid":"../shared/grid"}`. Use `--replicas N` for additional joint numerical variations. Scale variations use the common integration points of each run. For the independent-group interface to other integrators, see the [user guide](../../docs/NEW_PROCESSES.md).
