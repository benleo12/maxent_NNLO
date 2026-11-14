# W decay information: the full-production comparison

These compact results retain the full W+ and W− productions, all twenty frozen
candidates, seven coherent scale choices and the covariance of declared model
differences. They have different event statistics from the 80,000-event fitting
tutorial. The original W study in `../../w_joint` remains a separate result.

The full priors contain 522,365 W+ and 490,046 W− selected events. Their source
files are approximately 197 MB per charge and are not bundled here. Exact
full-prior re-solving requires those originals, whose hashes are preserved in
`../study_record/PRIOR_PHYSICS.json`. The compact 80,000-event fixtures run the
complete workflow independently; they do not reproduce every full-study number
bit for bit.

Redraw the independent charge-asymmetry comparison and exploratory old-cut map:

```bash
python examples/plot_w_predictivity.py --output w_study_plots
```

This creates `charge_asymmetry.pdf/png`, `old_cut_ledger.pdf/png`,
`acceptances.csv` and `report.json`. The table retains every model and scale,
while the plots show the prior, two recoil prescriptions and the selected
additional decay constraints. Values are read from the supplied metric caches;
the plotter does not repeat the fit or reconstruct a covariance approximation.

## Candidates and their purpose

| Name | Declared change |
|---|---|
| `P` | Original shower shape with a separate NNLO rate for each charge |
| `anchor` | Production and existing angular constraints with blended recoil |
| `R` | The same constraints with direct windowed recoil integrals |
| `D` | Higher degree in the existing polar-angle window |
| `M` | Low-order W mass–angle correlations |
| `N`, `G`, `W` | Declared narrow, nominal and wide mass-gated angular windows |
| `J4` | Rapidity–angle products through joint degree four |
| `H0`, `H` | Hard-recoil spin functions, respectively without and with rapidity dependence |
| `S0`, `S` | Recoil-suppressed spin functions, respectively without and with rapidity dependence |
| `R+H+S` | The combined choice selected by the frozen training rule for both full samples |

The angular/window/joint singleton controls use the blended anchor unless
prefixed by `R+`. Both spin families always use the direct windowed recoil
anchor. The difference between `R` and `anchor` is therefore a separate
transition-prescription control; the difference between `R+H+S` and `R` isolates
the added decay information. Exact functions, windows, orders and compatible
combinations are implemented in `../../../w_physical_inputs.py` and recorded in
`../feature_definitions.json`.

The fixed sequence considers `D`, `M`, `H`, then `S`. A block must be resolved at
every coherent scale, with the full training residual covariance, and its fit
must pass the imposed-input numerical check. The two alternating-seed training
halves independently select the same combination. Validation quantities never
enter that decision. A replacement prior must make its own choice; the compact
example does so with the same public implementation.

## Arrays and uncertainty interpretation

The raw prediction caches retain the 112 booked integrals for every candidate:
the baseline and six scalar selections, three eleven-bin lepton spectra, and
72 explicitly exploratory old-cut ledger entries. Central covariance is split
into common-integral and refitted-prior contributions. Declared pair records
retain cross-covariance, so a model difference is formed before its numerical
uncertainty. Seven scale columns remain coherent sensitivity variations.

`WP_metrics.npz` and `WM_metrics.npz` contain the normalized selections and
ledger fractions with their full ratio-Jacobian covariance. `asymmetry.npz`
contains the charge asymmetries and paired changes formed from absolute charge
rates. Independent-reference covariance is separate throughout. The scalar
selection conditioned on hard recoil has a `qT>50 GeV` denominator; the other
acceptances use the original baseline. Numerical errors do not include PDF,
shower or transition-model uncertainty.

The new 28 and 32 GeV lepton thresholds, transverse-mass and combined selections
were declared before reference inspection. The old 27/30 GeV ledger is an
exploratory explanation of earlier behavior. Its map shows contributions to
the accepted fraction in disjoint recoil–angle cells; it is not a fitted 2D
distribution or a closure certificate. Displayed lepton pseudorapidity follows
the massless convention: the actual booked quantity is charged-lepton rapidity.

Source checksums, frozen choices, precision-only stopping records, common-run
accumulators and scale-order audits accompany the scientific export. The
integration measurement cache was tested with `OPENMP=OFF` and one integration
thread per process. Reproduction parallelizes independent seed processes; the
cached measurement wrapper has not been validated for threaded integration.

`../PUBLIC_FILE_MAP.json` maps every original delivered artifact to its public
location and checks its SHA256 identity. `../study_record/` preserves the exact
campaign sources, technical history and independent reviews, including the
failed reference-only v5 check that was corrected before scientific production.
The immutable v4 training and v6 reference snapshots identify the versions that
actually produced the inputs. Read `../study_record/FINAL_SCIENCE_SUMMARY.md`
and the [reference precision record](../reference/README.md) alongside the plots:
the new selections improve, while the old 27 GeV acceptance retains a bias.
