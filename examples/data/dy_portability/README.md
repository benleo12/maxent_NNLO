# Full-sample Drell–Yan generator comparison

```bash
python examples/plot_dy_portability.py --data examples/data/dy_portability --output dy_portability_plots
```

This writes `dy_portability.pdf` and `.png` using only NumPy and matplotlib.
The command draws the generator and fixed-order spectra from the full samples used in the paper,
with 3,583,332 selected MG5+Pythia events and 572,823 Herwig events. The compact public fitting
demonstrations use 112,978 and 98,808 events, respectively.

## What the panels compare

Both columns use the full fiducial normalization at each scale. The upper panels show the original LO+PS samples, their reweighted distributions and the ATLAS measurements. The middle row divides each shower by fixed order in the hard region. The lower row divides it by data across the measured range.

The dilepton recoil was fitted. The six hard bins above 60.6 GeV test how closely reweighting reproduces that input. The inset resolves the recoil peak below 30 GeV. For the data comparison MG5+Pythia uses the ATLAS bins, and Herwig integrates eight intervals whose edges coincide with both histograms. [ATLAS recoil and comparison tables](../paper_distributions/README.md)

Lepton acoplanarity depends on the directions of both leptons and was held out of the fit. The fixed-order comparison uses four intervals in the hard region above 0.107. For the ratio to ATLAS, MG5+Pythia uses all 36 measured bins. Herwig uses four groups of complete ATLAS bins, with the uncertainty from splitting the generator bins included in its reweighted band. [Fixed-order calculation](../paper_distributions/drell_yan/phistar_fixed_order.md) · [Herwig in the measured bins](../paper_distributions/drell_yan/herwig_atlas_groups.md)

The reweighted-to-fixed-order ratios use the same scale choice in numerator and denominator. Their numerical bars combine finite event statistics with the spread from 20 target-integration replicas. The gray experimental band gives the uncertainty in each native ATLAS bin.

## Numerical and production record

Each generator has cached mass, rapidity, full and hard recoil, decay angle,
leading/subleading lepton momentum and phi-star distributions. Arrays include
the central shape, prior, six scale alternatives, twenty integration replicas,
two window alternatives and a common-order fit. The common-order check fixes
mass/rapidity/recoil degrees to 12/11/16 for both generators; the main curves use
their automatically selected orders. Reports retain all local statistics and
moment residuals, including sparse bins.

`*_weights.provenance.json` identifies the exact full prior, transformed moments,
effective configuration, implementation and variation inputs. The original saved
ensemble event weights are in pb; the separate common-order event weights are
unit-area. The compact histograms here are unit-area **densities**, as recorded
in `normalization_manifest.json`. The manifest also records the original archive
members and verifies the extracted metadata without distributing full weights.

`SHA256.json` fingerprints these caches and production records. The frozen
protocols, directly queried Herwig settings and original analysis source
(`PRODUCTION_ANALYSIS.txt`) are archived here. The source's `/work` paths describe
the original production container. Generator cards, sampling rules and the
streaming event adapter are documented in [HERWIG_GENERATION.md](../HERWIG_GENERATION.md)
and [DY_GENERATION.md](../DY_GENERATION.md). To perform a fresh public fit, use
`examples/real_drell_yan.py`, optionally with `--prior` for either compatible input.
