# Full-sample Drell–Yan generator comparison

```bash
python examples/plot_dy_portability.py --data examples/data/dy_portability --output dy_portability_plots
```

This writes `dy_portability.pdf` and `.png` using only NumPy and matplotlib.
The command draws the generator and fixed-order spectra from the full samples used in the paper,
with 3,583,332 selected MG5+Pythia events and 572,823 Herwig events. The compact public fitting
demonstrations use 112,978 and 98,808 events, respectively.

## What the panels compare

Every distribution is divided by its own full fiducial cross section, including
probability outside the displayed bins. No distribution is renormalized within
the plot range. Each scale member is normalized with its own fiducial rate.

The upper panels show both showers before and after reweighting. Each lower
fixed-order comparison uses the same reference bins as the upper spectrum.
The third recoil panel divides the reweighted shapes by the ATLAS measurement.
MG5+Pythia uses the native ATLAS bins. The released Herwig spectrum and the
measurement share eight complete intervals between 2 and 300 GeV. The gray
experimental band gives the pointwise errors on the original ATLAS bins.
The two model bands vary the fitted scales. MG5+Pythia also has finite-event
error bars. [Measured recoil and binned predictions](../paper_distributions/README.md)

The full recoil spectrum uses a linear 0–500 GeV axis, with an inset below 30 GeV. Its low-recoil shower bins
end at 60 GeV. The six fixed-order bins start at 60.615 GeV. The lower recoil panel
enlarges those six bins. The fixed-order spectrum and the recoil moments share 298
integration seeds, so this comparison tests how closely the reweighting reproduces
its input. Seeds 226 and 235 lack matching normalization inputs and were excluded.

The decay-angle reference combines 64 independent NNLO integrations. Both rows
show all 20 bins of the absolute Collins–Soper cosine. The transverse-momentum
cut on each lepton produces an acceptance edge near 0.8 at the Z pole. The
fixed-order scale and integration errors grow there. The angular measurement
published alongside the ATLAS recoil spectrum concerns dilepton acoplanarity.
The paper compares that observable to data separately.

Each colored scale band in the fixed-order ratios uses the same scale choice in numerator and
denominator. The colored error bars combine event sampling at fixed weights with
the scatter of twenty target-integration bootstrap fits. The reference integration
errors and its scale envelope are shown separately around one. The caches also
contain the transition variations.

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
