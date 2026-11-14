# maxent_upgrade

**Fixed-order QCD information in the shower events you already have.**

Reweight a positive-weight LO+PS sample using fixed-order moments of selected
observables. The package selects the resolved moments, solves for positive event
weights and reports their agreement with the input. Event kinematics stay intact.

The same solver works with any compatible prior and user-defined observables.
A new process needs its observable definitions and fixed-order moments, with no
changes to the solver. Ready-to-use inputs accompany the paper for Drell–Yan,
W±, Higgs, diphoton and Z+jet production.

[Quick start](#quick-start) · [All paper examples](examples/README.md) ·
[Prior input schemas](docs/INPUTS.md) · [New processes](docs/NEW_PROCESSES.md) · [API reference](docs/REFERENCE.md) ·
[Validation](docs/VALIDATION.md) · [Event records](docs/HEPMC.md) ·
[Constraint-study tutorial](docs/CONSTRAINT_TUTORIAL.md) ·
[Reproduction](REPRODUCE.md)

## Install

Python 3.9 or newer and NumPy are sufficient. From a checkout:

```bash
python -m pip install .
```

Optional plotting support: `python -m pip install '.[plot]'`.

## Run a real QCD example

A compact Drell–Yan sample and fixed-order references are included in the checkout:

```bash
git clone https://github.com/benleo12/maxent_NNLO.git
cd maxent_NNLO
python -m pip install '.[plot]'
python examples/real_drell_yan.py --output dy_demo --plot
```

This performs a new fit to the released NNLO moments on 112,978 real shower
events. It saves event weights, separate uncertainty sources, a constrained
hard-recoil comparison, an unconstrained decay-angle comparison against an
independent NNLO integration, and the acceptance of a tighter lepton cut.
`dy_demo/comparison.pdf` shows both comparisons; `report.json` records local
effective statistics, ratios and input provenance. The fit needs no network
access or external event sample after installation.

The example distinguishes propagated input errors from physics validation.
The independent angle is allowed to disagree: it was not fitted. The
[sample recipe](examples/data/DY_GENERATION.md) explains the sampling weights,
generator settings and which reference is independent. Omit `--plot` if only
NumPy is installed.

Repeat the same workflow on the independent Herwig input:

```bash
python examples/real_drell_yan.py --prior examples/data/herwig_public_prior.npz --output herwig_demo --plot
```

Its 98,808 events include the original weights, inclusion probabilities, IDs
and lepton four-vectors. The [Herwig recipe](examples/data/HERWIG_GENERATION.md)
records the production settings and sampling rule.

## Decide which information to add

A small executable [constraint-study tutorial](docs/CONSTRAINT_TUTORIAL.md) walks
through fitting separate moments, diagnosing an omitted correlation, freezing
the chosen physical block and predicting an independent reference:

```bash
python examples/constraint_workflow.py output/constraint_tutorial
```

It writes event weights, readable PDF/PNG figures and a local HTML report, with
the full numerical covariance and provenance beside them. This controlled
Born-decay illustration explains the workflow; it is not an NNLO calculation.
The [advanced interface](docs/STUDY_API.md) accepts arbitrary declared event
functions and common-run fixed-order integrals, using the same solver as
`upgrade`. Paired model changes retain their shared target and event fluctuations.

## Quick start

Inspect the required arrays and selection, then reweight your sample:

```bash
maxent-upgrade inspect dy
maxent-upgrade reweight dy my_prior.npz posterior.npz
```

`my_prior.npz` contains one array per event for `mll`, `y_abs`, `pT_ll` and a
strictly positive `weight`. Values use GeV. Apply the table's fiducial selection
before reweighting. Other event arrays are preserved in the output.

`posterior.npz` adds **`weight_upgraded`, in pb**, summing to the table's fiducial
cross section. Use these as replacement event weights. Pass `--unit-area` for
weights summing to one, or `--cross-section PB` to set another normalization.
The command also works as `python -m maxent_upgrade`.

The direct interface is equally short:

```python
import numpy as np
from maxent_upgrade import load_moments, sigma_fid, upgrade

with np.load("my_prior.npz") as sample:
    events = dict(sample)

moments, config = load_moments("dy")
result = upgrade(events, moments, config)
print(result.describe())

shape_weights = result.weights                          # sum = 1
weights_pb = shape_weights * sigma_fid("dy")["value"]   # sum = cross section in pb
```

`result.weights` already contains the prior weights. Do **not** multiply it by
`events["weight"]` again. For a relative reweighting factor, use
`result.weights / (events["weight"] / events["weight"].sum())`.

## Predictions with uncertainties

Add `--uncertainties` to re-solve the available scale choices, independent-run
bootstrap replicas, moved transition windows and PDF members:

```bash
maxent-upgrade reweight dy my_prior.npz posterior.npz --uncertainties
maxent-upgrade histogram posterior.npz pT_ll recoil.npz --bins 30 --range 0 300
python examples/plot_result.py posterior.npz pT_ll --range 1 300 --log-x --band scale
```

Every variation carries its own total cross section and positive replacement
event weights. The central moment orders stay fixed. Histogram bands keep the
sources separate: scale/window envelopes, integration-statistical standard
deviation and the CT18 68% Hessian band where available. The file also includes
the usual weighted-sample statistical error. Unavailable inputs are reported.
Shower variations can be supplied as generator weights through the direct interface.

```python
from maxent_upgrade import upgrade_with_uncertainties

result = upgrade_with_uncertainties(events, "dy", n_boot=20)
hist = result.histogram(events["pT_ll"], np.linspace(0, 300, 31))
# hist["central"], hist["scale_low"], hist["scale_high"], ... are in pb/GeV
np.savez_compressed("posterior.npz", **events, **result.arrays())
```

Use `--sources scale window` to request fewer sources, or `--bootstrap 100` for
more integration replicas. Storing weights costs 8 bytes per event per variation.
The same workflow supports new processes through `upgrade_ensemble`; the complete
[custom uncertainty example](examples/custom_uncertainties.py) runs without any
external data. See the [uncertainty reference](docs/REFERENCE.md#uncertainty-bands)
for normalization, shower inputs and the saved-array format.

## Try it without an event sample

```bash
python examples/custom_observables.py
python examples/plot_result.py custom_result.npz mass --label Mass --unit GeV
```

This short, self-contained example builds synthetic prior and target samples,
constrains two distributions and adds their joint moments. It prints the change
in their correlation and saves the weights. The optional plot shows the prior,
posterior and known target. The data are illustrative, not a QCD prediction.

## The paper examples

| Input | Process | Default constrained observables |
|---|---|---|
| `dy` | Drell–Yan, 13 TeV, NNLO | dilepton mass, absolute rapidity, recoil |
| `wp`, `wm` | W±, 7 TeV, NNLO | lepton–neutrino mass, absolute W rapidity, recoil |
| `ggh` | gluon-fusion Higgs, 13 TeV, NNLO | absolute Higgs rapidity, recoil |
| `aa` | diphotons, 8 TeV, NNLO | diphoton mass, Collins–Soper angle, recoil |
| `zj` | Z+jet, 13 TeV, NLO | dilepton mass and rapidity, two jet momenta and their joint moments |

[The example guide](examples/README.md) includes all five applications, both W
charges, the added decay-angle and acoplanarity constraints, separate and joint
jet constraints, the prior comparisons, uncertainty re-solves, and the letter's
NNLO Z+jet input. A single script makes the choices explicit:

```bash
python examples/paper_examples.py --list
python examples/paper_examples.py wp-angle --inspect
python examples/paper_examples.py wp-angle my_wp_prior.npz wp_angle.npz
```

The [two-jet benchmark](examples/data/zj_benchmark/README.md) includes ordinary
histogram comparators, every tested resolution and independent cut/balance
predictions, with a portable fitting script and compact plotting caches. The
[full-sample generator comparison](examples/data/dy_portability/README.md) can
also be redrawn directly from its numerical caches. The
[independent W study](examples/data/w_joint/README.md) provides frozen angular
and joint configurations, correlated training inputs and independent predictions,
with a public fitting command and complete comparison caches.

The tables include their fixed-order definitions and provenance. The compact DY fitting inputs and independent joint-validation histograms are
distributed here. The authors' full event productions are not. You can use
your own compatible prior, or request the authors' samples for numerical
reproduction. See [REPRODUCE.md](REPRODUCE.md).

## A new process or observable

The solver accepts arrays and moment constraints. Observable names are chosen by
you, and the prior can come from any generator that supplies positive weights
and adequate phase-space coverage.

```python
from maxent_upgrade import compute_fo_moments, upgrade

config = {
    "born": {
        "mass": {"range": (60.0, 140.0), "map": "lin"},
        "cos_theta": {"range": (-1.0, 1.0), "map": "lin"},
    }
}
moments = compute_fo_moments(fixed_order_events, config, n_max=8)
result = upgrade(prior_events, moments, config)
```

Here `prior_events` and `fixed_order_events` each contain `mass`, `cos_theta` and
`weight` arrays. Their event counts may differ. Fixed-order weights may be signed.
You can also accumulate the moment sums directly in your fixed-order code and
pass the resulting dictionary, without exporting its phase-space points.

The [new-process guide](docs/NEW_PROCESSES.md) gives the full path, including
recoil windows, correlations, uncertainty inputs and automation. The
[fixed-order recipe](MOMENT_CONSTRUCTION.md) gives the precise measurement functions.

## Reading the result

Saved fits include a [provenance record](docs/PROVENANCE.md) with configuration,
input and moment-table hashes, selected orders and weight normalization.
[Local statistics and joint checks](docs/VALIDATION.md) help decide whether an
analysis needs more events or additional correlation constraints. An optional
[HepMC adapter](docs/HEPMC.md) extracts the DY selection and attaches central
and named coherent variation weights to selected full records by event ID.


- **Moment agreement:** achieved moments compared with the input Monte Carlo
  uncertainties. A successful minimization alone does not establish that a prior
  is physically adequate.
- **Retained statistics:** `result.effN_retained` compares the effective event
  count with that of the prior. It is the useful measure for weighted or sliced samples.
- **Weight spread:** the range of the normalized weight ratio `q/p`. Inspect
  sparsely populated regions of observables important to your analysis as well.

Accuracy belongs to the constrained distributions and their stated kinematic
regions, up to moment truncation. Inclusive NNLO input supplies NNLO Born moments
and NLO hard-recoil moments. The low-recoil region relies on the shower.
Unconstrained distributions inherit the prior's conditional distributions, and
missing phase space cannot be generated by reweighting. Both sides must use the
same cuts, units, maps, windows and observable definitions.

## Code and reference material

| File | Purpose |
|---|---|
| [`reweight.py`](maxent_upgrade/reweight.py) | build constraints, select moments, solve and report |
| [`constraints.py`](maxent_upgrade/constraints.py) | fixed-order moments and transition functions |
| [`basis.py`](maxent_upgrade/basis.py) | variable maps and Chebyshev polynomials |
| [`solver.py`](maxent_upgrade/solver.py) | the convex-dual Newton solver |
| [`result.py`](maxent_upgrade/result.py) | weights and readable diagnostics |
| [`released.py`](maxent_upgrade/released.py) | released tables and their variations |
| [`cli.py`](maxent_upgrade/cli.py) | inspect inputs and reweight NPZ files |

[Moment definitions and provenance](moments/README.md) ·
[NNLOJET patches](nnlojet_patch/README.md) · [Run cards](runcards/README.md) ·
[Changelog](CHANGELOG.md)

The earlier histogram interface remains available for existing scripts, documented
in the [API reference](docs/REFERENCE.md#legacy-histogram-input).

## Citation

B. Assi, *Maximum-entropy reweighting of color-singlet samples* (2026).
The manuscript describes the method and the five applications.

For the letter's NNLO Z+jet recoil moments (`load_moments("dy", nnlo_zj=True)`),
also cite B. Assi, R. Poncelet and J. Whitehead, *Positive-weight moment matching
at NNLO and beyond* (2026), IFJPAN-IV-2026-23, MCNET-26-35.

MIT license. See [LICENSE](LICENSE).
