# Examples

Start with [custom_observables.py](custom_observables.py) for a complete run without
external files. It uses synthetic data and shows what joint moments add. For
physics applications, provide your own positive-weight prior inside the released
table's selection. The paper's large event samples are available from the authors.

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
contains the new weights **in pb**. The Python API instead returns unit-normalized
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

The following sections show what to supply and what to do with the weights.
Full fiducial definitions, jet and isolation settings are in
[the moment-table guide](../moments/README.md#what-your-prior-must-provide).

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
`load_moments("ggh", window="down", qt_basis="booked")` in Python.

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
That script provides the scale, input-statistical, matching and released PDF
re-solves. Its saved event weights sum to one. Matching `sigma_*_pb` arrays contain the
cross section for each scale, statistical replica or PDF member. For example,
`scale_w * sigma_scale_pb[:, None]` gives absolute event weights in pb.
The letter’s NNLO Z+jet table has no run replicas or moved windows, so that
part of the input cannot supply those variations. Shower
variations require the corresponding weights or alternate samples from the user's
generator and a new solve for each prior.

For the truncation example, repeat the solve with explicit counts, keeping the
other constraints fixed:

```python
central = upgrade(events, moments, config)
orders = dict(central.report["moment_selection"]["chosen"])
for degree in (2, 4, 8, 12):
    trial = {**orders, "mll": degree}
    result = upgrade(events, moments, {**config, "fixed_orders": trial})
```

The highest requested degree cannot exceed the available moments.

## NLO input and the linear mass-map control

The paper also compares NLO and NNLO Drell–Yan input and compares the Breit–Wigner
and linear mass maps. Their additional target tables are not released. The same
solver supports both comparisons when those inputs are supplied.

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
existing moments with a different map is not a valid map comparison. Exact
reproduction of these two controls needs the authors' target inputs and prior.

## Your own process

Use [custom_observables.py](custom_observables.py) for separate and joint Born
moments, and [example_moments.py](example_moments.py) for a recoil window. Both are
synthetic, require no physics input files and use `compute_fo_moments` followed by
`upgrade`. The [new-process guide](../docs/NEW_PROCESSES.md) explains how to replace
the toy arrays with your own inputs and automate the same procedure.

`example_synthetic.py` exercises the older histogram interface. It is retained for
compatibility and is not the method used for the paper's results.
