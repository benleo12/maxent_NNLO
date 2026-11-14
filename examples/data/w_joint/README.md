# Independent W predictions and joint rapidity-angle moments

This study separates the choice of constraints from the predictions used to
test them. Training moments select among a fixed sequence; a separate NNLO
integration then predicts lepton distributions, tighter-cut acceptances and the
charge asymmetry. The independent lepton data are not used to select constraints.

## Redraw the public comparisons

```bash
python examples/plot_w_joint.py --output w_joint_plots --full-scans
```

The compact caches contain every model, the full reference covariance and seven
coherent scale choices. The command writes the charge-asymmetry figure, a
machine-readable and a TeX acceptance table, a W− rapidity-angle redistribution
map, and optional model/cut scans. The map is the same-event ratio `J2/A`, with
sparse cells masked; it is not closure against an independent two-dimensional
NNLO reference. Run `examples/plot_w_joint_map.py` for the map alone.
Training covariance, prior-sample covariance and deterministic scale variations
are retained separately. Neither a successful fit nor an uncertainty envelope
replaces the independent prediction comparison.

The result is mixed. The selected combination reduces the RMS asymmetry residual
across the baseline eta bins from 0.00476 to 0.00300. With lepton pT>30 GeV it
instead changes 0.00687 to 0.00830, and the tighter-cut acceptances remain offset
from the independent calculation. Adding joint moments reduces the central
asymmetry residual relative to angular-only constraints at all three cuts. Every
model and cut remains in the release; none was selected using these predictions.
`charge_asymmetry.json` retains the paired statistical covariance of changes
relative to the prior and angular-only fits (`difference_from_prior` and
`difference_from_angle`), including the common-event and common-input correlations.

## Fit your own sample

```bash
python examples/w_joint_refinement.py wp my_wp_prior.npz wp_selected
python examples/w_joint_refinement.py wm my_wm_prior.npz wm_selected
```

Supply positive `weight`, `m_w`, `y_abs`, `pt_w` and `ctw1` arrays in the table's
7 TeV fiducial selection: lepton pT>20 GeV, neutrino pT>25 GeV, mT>40 GeV and
|eta_l|<2.5, before lepton QED radiation. The angle is **`ctw1 = 1 + cos(theta*)`**,
with the signed charged-lepton Collins–Soper angle and its longitudinal direction
specified in the table. It is not the absolute angle. Keep `eta_l` and `pt_l` to
produce the example's lepton histograms, and event IDs to preserve associations.
`eta_l.npz` is a **signed** pseudorapidity histogram on [-2.5,2.5]. When `eta_l`
is absent, the runner derives it from `l_born`, or accepts `y_l` as the massless
pre-QED lepton convention. The charge-asymmetry figures instead use `abs(eta_l)`;
combine the two charges' cross-section weights on the same absolute-eta bins,
rather than normalizing each charge separately.

The runner also reads the authors' built-prior layout, applying the same cuts
and deriving the angle from `l_born` and `nu`. That route records source row IDs
and its selection. A canonical input is assumed already selected. A new process
needs its own definitions and fixed-order inputs; the fitting solver is unchanged.

`--model` selects an explicit member of the frozen sequence:

| Model | Constraints |
|---|---|
| `B` | mass, absolute W rapidity and resolved recoil |
| `A` | `B` plus windowed decay-angle moments through degree six |
| `J2` | `A` plus rapidity-angle products through degree two |
| `J4` | `A` plus rapidity-angle products through degree four |
| `selected` | the model chosen from training information for that charge |

The default reproduces the frozen study choice: `A` for W+ and `J2` for W−.
It does not repeat the joint-degree selection for a new prior. Check supplied
joint moments and independent predictions before carrying that choice to another
sample; the complete `B/A/J2/J4` sequence remains available.

The joint products include `(k,0)`: the distribution of rapidity **inside the
angular window** is not fixed by its global marginal. The angular tower already
fixes that window's rate, so the mixed block uses `include_rate=False` to avoid a
duplicate constraint. The recoil moments were booked directly on log[25,500]
GeV and must not be transformed from the older [25,2500] interval.

The new `A`, `J2` and `J4` targets use the same directly booked angular window:
the feature is `w(c) T_n(c)`, including its rate. The older paper configuration
blends prior and fixed-order angular information through a different target
construction. Its unchanged weights are retained only as
`old_blended_angle_anchor_mean`, with the same event ordering verified and the
overall rate set to the new training normalization. This anchor is distinct from
`A`; its original target-input uncertainties are not recomputed in this study.

Each run writes `posterior.npz` with central and coherent variation weights,
provenance and retained input arrays, plus local-analysis histogram files when
the lepton observables are available. Scale alternatives use one complete input
vector at a time. Integration replicas resample complete seed vectors within
each independent perturbative channel and then sum the channel means. They keep
numerator/denominator correlations; individual moments are never independently
jittered. `--bootstrap` and `--seed` are recorded in the output. The runner does
not open the independent validation files while fitting.

The fresh histogram files' `sample_error` conditions on the fitted weights; it
does not include a refit of a resampled prior. The cached study's prior covariance
is a separate linear-response calculation, retained apart from target-integration
covariance. Its production source and Jacobian checks accompany the booking bundle.
The supplied fresh W-study workflow propagates scale and target-integration
uncertainties. It has no dedicated PDF-member, shower or moved-window inputs;
those require additional compatible variations. The fixed windows and selected
constraints remain unchanged across its replicas.

The training JSON files contain configurations, selection, basis/window maps,
all seven mean accumulator vectors, their full statistical covariance, frozen
model decisions and source identities. The paired `*_training_runs.npz` files
retain the original per-channel/per-seed accumulators. Validation files remain
separate. Their `*_validation_runs.npz` files retain per-channel counts: valid
rows have `seed >= 0`, while unused padded rows have seed -1 and NaN arrays.
Use each channel's actual `n_runs` when resampling; a precision extension in one
channel does not add seeds to the others. `PROTOCOL.txt` records the predeclared
decisions and the independent precision-only extension.

The [booking bundle](../../../nnlojet_patch/w_validation/BOOKING_README.txt)
contains the actual training/validation cards, added measurement functions, a
portable installer for the baseline NNLOJET patch, source hashes and the pooling
recipe. Its archived orchestration paths document the production environment;
new production uses a local NNLOJET build and independently adapted grids.
The [study sources](../../../nnlojet_patch/w_validation/study_sources/README.txt)
archive the exact selection, propagation, pooling and precision-audit programs,
both model-freeze records and the completed validation marker. Original export
readers/plotters are in `../../../nnlojet_patch/w_validation/export_sources/`;
their hashes are preserved in `PRODUCTION_ARTIFACT_SHA256.txt`. The adapted
public entry points are the scripts in `examples/`.

`FINAL_MANIFEST.json` records the exact charge-dependent channel counts, raw
input and frozen-weight hashes, source snapshot, freeze times and propagation
conventions. `SHA256.json` fingerprints the files distributed in this data
directory. Large raw-log archives remain production records outside this compact
release; the supplied seed accumulators reproduce the released reference means,
covariances and resampling without them.

## Checking the public interface

The `wp_public_equivalence.json` and `wm_public_equivalence.json` records compare
the public reader and solver against all four frozen study fits on the complete
original priors. Selected row hashes agree exactly; maximum relative weight
differences are below 2.3e-9. The two `*_joint_demo_check.json` files record
fresh selected-model runs with six scale alternatives and twenty complete
integration replicas. Every positive weight member sums to its own recorded
cross section, and signed lepton-eta histograms and unique event IDs are retained.
These are interface and normalization checks; they do not assess agreement with
independent physics predictions.
