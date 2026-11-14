# Examples

Start with the committed real Drell–Yan fitting example:

```bash
python examples/real_drell_yan.py --output dy_demo --plot
```

It creates `posterior.npz`, `recoil.npz`, `angle.npz`, `acceptance.npz`,
`report.json` and, with `--plot`, `comparison.pdf`. The report contains local
statistics, comparison ratios and source provenance. The [data recipe](data/DY_GENERATION.md)
documents the generator sample, inverse-inclusion weights and independent angle
reference. This is a fresh fit, not a plot of precomputed posterior weights. Repeat it
on the [independent Herwig sample](data/HERWIG_GENERATION.md) with
`--prior examples/data/herwig_public_prior.npz --output herwig_demo`.

For full event records, [the HepMC example](../docs/HEPMC.md) reads, fits and
reattaches central and coherent variation weights by ID. The optional extension
tutorials [custom_observables.py](custom_observables.py) and
[custom_uncertainties.py](custom_uncertainties.py) use synthetic data to explain
new observables, joint constraints and uncertainty inputs. They also demonstrate
[held-out joint checks and local statistics](../docs/VALIDATION.md).

For a complete training, frozen-fit and independent-prediction workflow, run:

```bash
python examples/constraint_workflow.py output/constraint_tutorial
```

The [tutorial](../docs/CONSTRAINT_TUTORIAL.md) explains the resulting weights,
PDF/PNG figures, local HTML report and complete covariance. This small controlled
Born-decay example makes the workflow explicit without presenting it as QCD
evidence. Its public [study interface](../docs/STUDY_API.md) also supports
user-defined physical blocks and common-run integration inputs.

The full paper configurations below require a compatible positive-weight prior
inside the released table's selection. The authors' full event productions are
available on request.

All commands below run from the repository root after `python -m pip install .`.
Plotting is optional and uses `python -m pip install '.[plot]'`.

## Released paper configurations

```bash
python examples/paper_examples.py --list
python examples/paper_examples.py dy --inspect
python examples/paper_examples.py dy my_dy_prior.npz dy.npz
```

The inspect command lists the actual required arrays, selected maps and windows.
All input arrays remain in the output, in their original order. `weight_upgraded`
contains the new weights **in pb**. The direct `upgrade` call returns unit-normalized
weights, which must be multiplied by the desired cross section.

| Example name | Constraints | Further use |
|---|---|---|
| `dy` | mass, absolute rapidity, dilepton recoil | plot the unconstrained absolute Collins–Soper angle |
| `wp`, `wm` | mass, absolute rapidity, W recoil | combine charges into the lepton asymmetry |
| `wp-angle`, `wm-angle` | the same, plus decay-angle moments | test the role of the decay distribution in acceptance |
| `ggh` | absolute Higgs rapidity and recoil | inspect leading-jet momentum, jet multiplicity and fiducial acceptance |
| `aa` | diphoton mass, absolute Collins–Soper angle and recoil | inspect the unconstrained photon separation |
| `zj-no-joint` | mass, rapidity and separate jet moments | the control without joint jet constraints |
| `zj` | the same, plus joint jet moments | inspect jet balance and unconstrained lepton acoplanarity |
| `zj-acoplanarity` | the same, plus acoplanarity moments | the added constraint shown in orange in the paper |
| `dy-nnlo-zj` | Drell–Yan with the letter's NNLO Z+jet recoil moments | NNLO input for the hard recoil |
| `dy-joint` | Drell–Yan with joint mass and decay-angle moments | the letter's joint-moment example |

The [five-application input guide](../docs/INPUTS.md) gives the event-array
schemas and extraction instructions. The following sections show how to use
those arrays and the resulting weights.
Full fiducial definitions, jet and isolation settings are in
[the moment-table guide](../moments/README.md#what-your-prior-must-provide).

## Independent comparisons

Redraw the full-sample generator comparison, two-jet benchmark and independent W study:

```bash
python examples/plot_dy_portability.py --output dy_portability_plots
python examples/plot_zj_benchmark.py --output zj_benchmark_plots
python examples/plot_w_joint.py --output w_joint_plots --full-scans
```

The [generator comparison](data/dy_portability/README.md) uses the full Pythia
and Herwig productions, with matched hard-recoil bins and the independent angular
reference. These curves have different event statistics from the compact public
fitting examples.

The [two-jet benchmark](data/zj_benchmark/README.md) compares moment, marginal
histogram and joint-histogram reweighting at a common integration budget. It
retains every resolution, failed fit, local cell count and independent prediction.
Use `zj_benchmark.py` to refit a compatible prior.

The [W study](data/w_joint/README.md) releases its frozen production, angular
and joint targets, full training covariance, original seed accumulators and
independent lepton/cut/charge-asymmetry predictions. Use `w_joint_refinement.py`
to apply the frozen configurations to a compatible prior with coherent scale
and integration variations. Its `selected` option reproduces the study choice;
it does not select a new joint degree for an arbitrary prior.

These plotting commands use compact committed caches; fitting a new prior
requires event arrays.

## Drell–Yan

The required arrays are `mll`, `y_abs`, `pT_ll` and `weight`. They describe Born-level
leptons before QED radiation in the 13 TeV selection. Keep any additional analysis
observables in the file, for example `abs_costh` for the decay-angle distribution.

```bash
python examples/paper_examples.py dy dy_prior.npz dy.npz
python examples/plot_result.py dy.npz pT_ll --label 'Dilepton transverse momentum' --unit GeV --log-x --range 2 500
python examples/plot_result.py dy.npz abs_costh --label '|cos(theta*)|'
```

For the letter's input or its joint moments, replace `dy` with `dy-nnlo-zj` or
`dy-joint`. These are separate from the `zj` application, which is an NLO calculation
with explicit jet constraints.

## W production and the charge asymmetry

For each charge provide `m_w`, `y_abs`, `pt_w` and `weight` in the 7 TeV fiducial
selection. Added decay-angle moments require **`ctw1 = 1 + cos(theta*)`**, with the
signed Collins–Soper angle of the charged lepton, for both charges. This shifted
variable is part of the table definition.

```bash
python examples/paper_examples.py wp wp_prior.npz wp.npz
python examples/paper_examples.py wm wm_prior.npz wm.npz
python examples/paper_examples.py wp-angle wp_prior.npz wp_angle.npz
python examples/paper_examples.py wm-angle wm_prior.npz wm_angle.npz
```

The lepton asymmetry needs the two charges' absolute cross sections. For an
additional event array `eta_l`, already included in each input file:

```python
import numpy as np

with np.load("wp.npz") as plus, np.load("wm.npz") as minus:
    edges = np.linspace(0, 2.5, 11)
    sp = np.histogram(abs(plus["eta_l"]), edges, weights=plus["weight_upgraded"])[0]
    sm = np.histogram(abs(minus["eta_l"]), edges, weights=minus["weight_upgraded"])[0]
    asymmetry = np.divide(sp - sm, sp + sm, out=np.full_like(sp, np.nan), where=(sp + sm) > 0)
```

Repeat with the angle-constrained outputs to make the comparison. Do not normalize
the two charge distributions separately to unit area before forming the asymmetry.

## Inclusive Higgs and cuts after reweighting

Provide an inclusive prior with `y_abs`, `pT_H` and `weight`. Keep the leading-jet
momentum, jet count and a boolean analysis-selection array for subsequent analysis.
The default recoil transition begins at 54 GeV.

```bash
python examples/paper_examples.py ggh higgs_prior.npz higgs.npz --cross-section 48.58
```

The paper uses a total production cross section of 48.58 pb at N3LO. Without this
explicit override, the output uses the table's NNLO production cross section.
For a boolean array `fiducial` computed from the decay products:

```python
with np.load("higgs.npz") as sample:
    accepted = sample["fiducial"].astype(bool)
    weights = sample["weight_upgraded"]
    acceptance = weights[accepted].sum() / weights.sum()
    fiducial_production_pb = weights[accepted].sum()
```

Multiply by the decay branching fraction when predicting a decay-channel cross
section from a sample with a forced decay. Retain the inclusive normalization when
histogramming accepted events. For the ATLAS jet-count comparison, the other Higgs
production modes must also be supplied separately, as described in the paper.

A prior with matrix-element corrections is reweighted by the same command. The
prior changes, not the moments or solver. The letter's 37 GeV setup uses
`load_moments("ggh", window="down", qt_basis="booked")`.

## Diphotons and prior dependence

Provide `m_aa`, `abs_costh_aa`, `pt_aa` and `weight` in the 8 TeV photon selection,
with the specified isolation. Preserve the relative slice weights when combining
generation slices. An unconstrained array such as `dphi_aa` can be plotted afterwards.

```bash
python examples/paper_examples.py aa merged_prior.npz aa_merged.npz
python examples/paper_examples.py aa two_to_two_prior.npz aa_two_to_two.npz
python examples/plot_result.py aa_merged.npz dphi_aa --label 'Photon azimuthal separation' --unit rad
```

The same fixed-order input is used for both priors. The retained effective count
and weight spread expose the statistical cost of inadequate hard-emission coverage.
Agreement in a constrained distribution does not establish correct correlations.
The plotting example normalizes to unit area, as in the paper's data comparisons.

## Z+jet and joint constraints

Provide `mll`, `y_abs`, `ptj1`, `ptj2` and `weight` for events with at least one jet
in the table's selection. Set `ptj2 = 0` for events without a second jet. The added
acoplanarity constraint reads **`pimdphi = pi - Delta_phi(lepton, lepton)`**, in radians.

```bash
python examples/paper_examples.py zj-no-joint zj_prior.npz zj_separate.npz
python examples/paper_examples.py zj zj_prior.npz zj_joint.npz
python examples/paper_examples.py zj-acoplanarity zj_prior.npz zj_acoplanarity.npz
```

Use the same three commands with a prior generated with lepton cuts to reproduce
the setup of the prior-coverage comparison. For the balance panel, histogram
`ptj2 / ptj1` for events with `ptj1 > 60` GeV and use the corresponding weights.
Normalize the selected distribution when making a unit-area comparison.

The Python equivalents make the constraint choices explicit:

```python
load_moments("zj", mixed=[])                           # separate jet moments
load_moments("zj")                                     # includes their joint moments
load_moments("zj", recoil=["ptj1", "ptj2", "aco"])      # adds acoplanarity
```

## Uncertainties and truncation

For the moved transition regions, every paper configuration accepts `--window`:

```bash
python examples/paper_examples.py ggh higgs_prior.npz higgs_down.npz --window down
python examples/paper_examples.py ggh higgs_prior.npz higgs_up.npz --window up
```

These standalone commands select moments afresh. To construct a band, hold the
central moment orders fixed, as in [bands_with_released_moments.py](bands_with_released_moments.py).

The same calculation is available directly from the installed command:

```bash
maxent-upgrade reweight wp my_wp_prior.npz wp_bands.npz --uncertainties --recoil pt_w ctw_window
maxent-upgrade histogram wp_bands.npz pt_w wp_recoil.npz --bins 30 --range 0 150
python examples/plot_result.py wp_bands.npz pt_w --band pdf --range 0 150
```

The new output includes event arrays and coherent replacement weights in pb for
each variation. `--sources scale window` selects fewer sources. The histogram
command produces separate bands and weighted-sample statistical errors, for any
observable present in the event file. Use `--unit-area` for normalized shapes.
The old band script above retains its documented shape-weight/rate format.

For a complete example with your own observable names and uncertainty inputs:

```bash
python examples/custom_uncertainties.py
python examples/plot_result.py custom_bands.npz mass --label Mass --unit GeV
```

This synthetic example uses `upgrade_ensemble` and `Variation` without a released
process table. Its target and variations are illustrative, not QCD predictions.

The older `bands_with_released_moments.py` script provides the scale, input-statistical, matching and released PDF
re-solves. Its saved event weights sum to one. Matching `sigma_*_pb` arrays contain the
cross section for each scale, statistical replica or PDF member. For example,
`scale_w * sigma_scale_pb[:, None]` gives absolute event weights in pb.
The letter’s NNLO Z+jet table has no run replicas or moved windows, so that
part of the input cannot supply those variations. Shower
variations require the corresponding weights or alternate samples from the user's
generator and a new solve for each prior.

The paper's two-dimensional jet-correlation comparison also runs without the
large event samples:

```bash
python examples/plot_joint_validation.py
```

It compares the separate and joint fits with an independently integrated Z+2-jet
reference. [Binned data and provenance](data/README.md) are included. For other
observables, form the same two-dimensional histograms from your posterior event
weights and fixed-order reference, with common bins and normalization.


For a truncation study, repeat the solve with explicit counts, keeping the
other constraints fixed:

```python
central = upgrade(events, moments, config)
orders = dict(central.report["moment_selection"]["chosen"])
for degree in (2, 4, 8, 12):
    trial = {**orders, "mll": degree}
    result = upgrade(events, moments, {**config, "fixed_orders": trial})
```

The highest requested degree cannot exceed the available moments.

## Comparing perturbative orders and mass maps

The same solver can compare NLO with NNLO moment inputs, or Breit–Wigner with
linear mass maps. These are additional controls users can perform with suitably
booked fixed-order inputs. The manuscript's mass-map illustrations show prior
geometry and conditioning; they are not fitted linear-map comparisons.

For an NLO-input comparison, construct the moments from an NLO calculation with
the same observable definitions and selection, then call `upgrade` on the same
prior. Use the NLO cross section for an absolute prediction. Changing a table's
order label does not change its perturbative input.

For a linear-map comparison using fixed-order points:

```python
from copy import deepcopy
from maxent_upgrade import compute_fo_moments, load_moments, upgrade

_, linear_config = load_moments("dy")
linear_config = deepcopy(linear_config)
linear_config["born"]["mll"]["map"] = "lin"
linear_moments = compute_fo_moments(fixed_order_events, linear_config)
linear_result = upgrade(events, linear_moments, linear_config)
```

Estimate production uncertainties with independent integrator runs, as described in
the [new-process guide](../docs/NEW_PROCESSES.md#3-supply-the-moments). Relabeling
existing moments with a different map is not a valid map comparison. These
optional controls require target moments for the chosen order and map, together
with a compatible prior.

## Your own process

Use [custom_observables.py](custom_observables.py) for separate and joint Born
moments, and [example_moments.py](example_moments.py) for a recoil window. Both are
synthetic, require no physics input files and use `compute_fo_moments` followed by
`upgrade`. The [new-process guide](../docs/NEW_PROCESSES.md) explains how to replace
the toy arrays with your own inputs and automate the same procedure.

`example_synthetic.py` exercises the older histogram interface. It is retained for
compatibility and is not the method used for the paper's results.
