# Full-sample Drell–Yan generator comparison

```bash
python examples/plot_dy_portability.py --data examples/data/dy_portability --output dy_portability_plots
```

This writes `dy_portability.pdf` and `.png` using only NumPy and matplotlib.
The command redraws the full-sample manuscript comparison: 3,583,332 selected
MG5+Pythia events and 572,823 independently generated Herwig events. The compact
public fitting demonstrations use 112,978 and 98,808 events, respectively; they
are separate fits and need not reproduce the full-sample numerical curves.

## What the panels compare

Every distribution is divided by its own full fiducial cross section, including
probability outside the displayed bins. No distribution is renormalized within
the plot range. Each scale member is normalized with its own fiducial rate.

The hard-recoil comparison is a constraint-closure check. Its fixed-order
histograms and moments share the same integrations: 298 paired seeds, with
seeds 226 and 235 omitted because they lack matching normalization inputs. The
middle panel uses identical bin boundaries for every numerator and denominator.
Upper and lower recoil panels join the cached low-recoil bins ending at 60 GeV
to the common hard bins beginning at 60.615 GeV. The uncomputed interval is left
blank; values are not interpolated across it. All upper hard-recoil curves
therefore use identical bins.

The angle reference combines 64 independent NNLO integrations and was not used
to fit the moments. The region `0.75 < |cos(theta_CS)| < 0.9` is shaded and its
fixed-order ratios are omitted because of the fixed-order shoulder. The complete
arrays remain available, with their integration errors. The lower panels divide
each posterior by its own prior and retain the same global normalization.

In the middle panels, each colored scale band comes from seven ratios formed
with the **same scale choice in numerator and denominator**. Colored error bars
combine fixed-weight event-sampling errors with the scatter of twenty complete
target-integration bootstrap fits. The reference's integration errors and scale
envelope are shown separately around one. These are distinct uncertainty sources;
the event-sampling term does not refit a Poisson-resampled prior. Lower-panel
bands divide the posterior scale envelope by the central prior. Window variations
are stored in the caches but are not included in these displayed scale bands.

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
