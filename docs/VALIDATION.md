# Check a prediction before using it

A small moment residual establishes agreement with the supplied constraints.
For an analysis, also inspect local statistics and observables that were not
used to determine the weights. Propagating input errors and validating the
prediction answer different questions.

## Local statistics

```python
from maxent_upgrade import bin_diagnostics, weight_diagnostics

print(weight_diagnostics(result.weights, events["weight"]))
local = bin_diagnostics(events["my_observable"], analysis_edges, result.weights)
```

Each bin reports its event count, effective count `(sum w)^2/sum(w^2)`, largest
fraction of its weight carried by one event, and largest fraction of `sum(w^2)`
carried by one event. These expose a tail whose apparent precision depends on a
few events even when the global effective count is large. Their values do not
depend on whether the weights are normalized to one or to a cross section.
For a weighted prior, the retained effective fraction can exceed one: the fit
can make the weights more uniform.

An optional nonnegative `reference=` array supplies one expected value per bin.
Then `empty_with_reference` marks bins with positive reference content and no
sample events. An occupied bin can still contain incomplete phase space or
incorrect conditional distributions; these checks cannot certify support.

## Decide whether to add a joint block

First fit the separate distributions that matter for the analysis. Then evaluate
candidate joint moments **without imposing them**:

```python
from maxent_upgrade import load_moments, upgrade, joint_moment_residuals

moments, separate_config = load_moments("zj", mixed=[])
separate = upgrade(events, moments, separate_config)
joint_targets, joint_config = load_moments("zj")
before = joint_moment_residuals(events, separate.weights, joint_targets, joint_config)

joint = upgrade(events, joint_targets, joint_config)
after = joint_moment_residuals(events, joint.weights, joint_targets, joint_config)
```

Each row records the product moment, target, residual, input error and a
fixed-weight sample error. The conditional window mean and the window's rate
are reported separately. `residual_in_input_errors` is descriptive: the rows
are correlated, the fixed-order targets can share integration runs, and fitting
introduces further correlations. Do not combine these ratios into a chi-square
without their covariance.

Add a block when the remaining correlation matters for the intended observable
and the discrepancy is resolved by the supplied calculations and sample. Check
its effect on the intended observable, local effective statistics, and a joint
histogram or independent moment set that was not used for fitting. If a newly
imposed moment agrees, that is constraint closure. Agreement with an independent
reference is a separate validation. This procedure does not automatically select
arbitrary correlations or invent the required fixed-order calculation.

`examples/custom_observables.py` demonstrates the diagnostic before and after a
joint fit. `examples/plot_joint_validation.py` shows the paper's independent
Z+2-jet reference and its cell statistics. The latter plots released results;
use `paper_examples.py` with your own event sample to perform that fit.

## Keep uncertainty sources and physics comparisons separate

`upgrade_with_uncertainties` propagates available scale choices, independent-run
replicas, moved windows and PDF members. The same coherent weight variation
must be used across every bin and any changed analysis cut. A band does not
include every possible error in the shower's correlations or missing event
configurations. The sample error conditions on the fitted weights; it does not
include resampling and refitting the prior.

The [real Drell–Yan example](../examples/real_drell_yan.py) makes this distinction
visible. Hard recoil tests the fitted spectrum using histograms from the moment
production. Its unconstrained decay-angle reference comes from a separate NNLO
integration. Differences there are reported as physics results, not converted
into software test failures or tuned away.
