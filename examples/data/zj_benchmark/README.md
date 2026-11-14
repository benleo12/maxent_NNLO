# A fixed-budget comparison of moment and histogram reweighting

All methods use the same positive-weight prior in
`60 <= pTj2 <= pTj1 < 1000` GeV and the same fixed-order integration budget.
Distributions are normalized over that hard two-jet region. The fixed-order
reference is the real contribution to NLO Z+jet, which supplies **LO Z+2-jet**
accuracy in this selection. This comparison does not establish NNLO accuracy
for jets or test the low-recoil transition.

## Read the independent comparisons

```bash
python examples/plot_zj_benchmark.py --output zj_benchmark_plots
```

This uses only the committed aggregate caches. It writes the overview figure,
`metrics.csv`, `local_support.csv` and `analysis.json`. The stored arrays retain
all 21 methods/resolutions, seven coherent scale choices, 32 paired statistical
replicas and the full 88-component prediction vectors. The independent reference
includes its covariance. Deterministic scale changes remain separate from the
statistical covariance. The local table covers every cell of the 8-by-8 grid,
including the sparse cells omitted from the displayed supported-region distance.

The summary records the validity of the optional angular reference separately
from the primary jet quantities. An invalid angular reference has NaN array
entries and null JSON metrics, with its reason retained. Those entries are not
used to assess the jet fits.

All resolutions were fixed before inspecting the independent validation sample:

- separate degree-four moments, with joint degrees one through four;
- full separate/joint degree-one, two and three fits;
- histogram marginal matching by iterative proportional fitting, and direct
  joint-histogram ratios, with 2, 3, 4, 5, 7 and 10 bins per axis;
- the original prior.

Both moment matching and histogram matching are entropy projections with
different measurement functions. Every method sees the same training integration.
`joint_residuals` measures un-imposed or imposed moment discrepancies; the cut,
balance and two-dimensional histogram predictions are checked separately against
independent integration points. Agreement with an imposed target is closure, not
an independent test of that prediction.

An exact histogram fit can fail if a positive-target cell contains no prior
events. The summary retains the reason and missing target probability. A failure
in a cell containing a tiny fraction of the rate does not establish a large
inclusive error or invalidate the histogram approach. Failed array entries remain
NaN, available JSON diagnostics use null, and the number of successful bootstrap
replicas is reported. The plots do not silently select a successful resolution.

## Refit another prior

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python examples/zj_benchmark.py my_zj_prior.npz my_benchmark
python examples/plot_zj_benchmark.py --results my_benchmark --output my_benchmark_plots
```

Supply `weight`, `ptj1`, `ptj2`, `njet` and `pimdphi` arrays. Momenta are in GeV and
`pimdphi = pi - Delta_phi(lepton,lepton)` is in radians. Before this script, apply
the released Z+jet lepton/jet definitions in `moments/zj_nlo_moments.json`, using
jets after shower and MPI but before hadronization for the paper's prior. The
script then applies the common two-jet hard-region cuts. An overall normalization
of the prior cancels. It does not reconstruct or redefine jets.

The default is the complete 32-replica protocol. `--bootstrap` can reduce the
replica count for a workflow check; that changes its statistical precision and is
recorded in the output. Each replica jointly resamples the training integrations
and Poisson-resamples the prior events. The independent validation integration is
first opened after all fits are frozen. The output records input hashes and the
implementation identity; no output is written into the committed data directory.

`definitions.json` contains the exact cuts, resolution grids and observable
boundaries. `PROTOCOL.txt` preserves the predeclared scientific protocol.
`PRODUCTION_NOTES.txt` records the original production environment and jobs; its
machine paths are historical records, not prerequisites for the public scripts.
The original benchmark used package commit b437a68. The portable script uses the
installed/current package and fingerprints its implementation.
