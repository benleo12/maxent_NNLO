# Supply your own prior and moments

`maxent-upgrade fit prior.npz targets.json upgraded.npz` accepts any process with
a positive LO+PS prior covering the target phase space. Each target names an observable,
its polynomial degree, its normalized fixed-order average and its absolute numerical
or combined error. Include `stat_error` separately when the quoted total also contains
scale variation. Use the same event definitions, units and cuts in both calculations.
The optional `selection` string records the fiducial cuts. Apply them upstream.

The [Drell–Yan target file](../examples/own_prior/targets.json) is a complete
example. It takes two absolute-rapidity moments from the released NNLO table, including
its seven scale choices and corresponding cross sections. It can be fitted to either
of the two generators in the checkout. Replace the column names, maps and target
rows for another physical process.

## Identify the measurement function

Define each ID once in `config`. A Born or recoil ID is the input event column.
For a joint ID, supply the two column names in the same order as the degree pair.
The joint block inherits their maps. For an axis that appears only in a joint block,
declare its interval and map under the block's `range` and `map` objects.

```json
{
  "born": {
    "mass": {"map": "lin", "range": [60, 140]},
    "angle": {"map": "lin", "range": [-1, 1]}
  },
  "mixed": {
    "mass_angle": {
      "observables": ["mass", "angle"], "n": 2
    }
  }
}
```

The target row
`{"observable":"mass_angle","degree":[1,2],"value":0.03,"error":0.002}`
specifies ⟨T₁(u_mass)T₂(u_angle)⟩. Supply the desired product degrees up to the
declared maximum. Individual observables can also supply a sparse set of degrees.
For an even correction, give degrees 2 and 4 alone. The fit then has no odd-degree
constraint. Supply target rows in any order. Automatic selection keeps the supplied
degrees up to the highest one it resolves against the prior. It needs an
`error` per central tower moment. To impose every supplied degree yourself, set
`"moment_selection":false` inside `config`. You may then omit the errors for a
whole tower. Input variations are refitted at the same degrees and can omit their
errors. A windowed or joint target may also require its fixed-order window fraction. Use `lin`, `log` or
`["bw", pole_mass, width]`. The coordinates clip to the declared interval.
`T_n(cos θ) = cos(nθ)` defines the Chebyshev convention.

For recoil, a smooth window can turn on fixed order above the singular region.
For example, declare `"recoil":{"pt":{"range":[20,500],"map":"log","profile":{"a":20,"b":40,"c":500}}}`
alongside the Born definitions. For this tower, each supplied moment is
⟨w T_n⟩/⟨w⟩. Provide the window fraction ⟨w⟩ separately under
`"rates":{"pt":{"value":0.31,"error":0.002}}` in the target file. The same rule
holds for a joint block carrying a window on either axis. Fixed-order booking and
reweighting use the same map, window and endpoints. The paper documents the physics
choice of transition for each process.

## Book targets or supply the numbers

Write the observable definitions in `observables.json`. With a signed fixed-order
sample, book all requested averages and their errors once:

```sh
maxent-upgrade moments observables.json fixed_order.npz targets.json --orders 6 --cross-section 12.4
maxent-upgrade fit prior.npz targets.json upgraded.npz
```

The two samples need one finite value per constrained observable per record,
and a `weight` column. Only the shower prior requires strictly positive individual
weights. Signed contributions in the integration sample must give a positive fiducial
total and positive window integrals. The fixed-order record ordering is independent
of that of the prior. If the records carry common scale choices, supply their absolute
integration weights in `weight_scales[choice, record]`, with row zero equal to
`weight`, and string `scale_labels[choice]` if available. These weights must have a
common overall normalization across choices. Member cross sections are the central
rate times the ratio of signed sums. An integrator with separately normalized samples
can instead supply each varied rate directly.

An integrator can write the target rows without exporting its records. The values
are ratios of the signed weighted sums, normalized by the fiducial integral (or the
window integral for windowed moments). Use independent-run errors when the integrator
pairs real and subtraction records. [Fixed-order definitions](../MOMENT_CONSTRUCTION.md)

## Return variations to your analysis

The optional `variations` object groups member inputs by source. Each member gives
a `label`, its own `moments` and, for absolute predictions, its `cross_section_pb`.
The central map definitions apply to every row. Each varied input must identify
exactly the same degrees. Include all window fractions for that member. To vary
only a normalization, supply a label and a cross section. For shower variations on
the same events, supply the name of the positive replacement prior column as
`prior_weight_column`. Statistical replicas from independent integrations use the
source `stat` (at least two members). For a covariance-based calculation using
common-run integrals see the [correlated-input interface](STUDY_API.md).

The output records the number of fitted moments per ID in `moment_orders` and their
actual degrees in `moment_degrees_json`, for example `{"angle":[2,4]}`. For the
optional `fixed_orders` setting, give a count of supplied tower degrees in ascending
order. A count of one in this example imposes degree two.

The output contains all input fields plus `weight_upgraded` and, per source,
`weight_<source>[member,event]` and `labels_<source>`. Sum a weight array over
selected rows for the corresponding fiducial cross section. Without a supplied
central cross section the outputs sum to one. No second fit is needed when the
analysis selection changes.

## Check the moments returned by the fit

```sh
maxent-upgrade check dy_weights.npz
maxent-upgrade check dy_weights.npz --source scale --member "muR = 2 m_ll, muF = 2 m_ll"
```

The second command uses a scale label from the absolute-rapidity example. Each
report compares the prior and fitted averages of the **same measurement function**
with its target. A windowed row shows the mean of the window times the polynomial
over the full fiducial sample. A blended row includes the complementary prior
contribution. Window fractions appear as separate rows when imposed. The signed
residual is the achieved mean minus the target. These dimensionless quantities
also apply to an output normalized in pb.

The NPZ field `constraint_checks_json` stores the central rows and each source's
labeled variation rows. Each row identifies its observable, degree, measurement
kind and hard or soft treatment. `input_error` gives the supplied total error
in feature units. For a target depending on both a window moment and its fraction,
the report propagates their marginal errors in quadrature at fixed prior. An
unavailable error appears as `null` in the file and a dash in the display.
A joint significance test needs the integration covariance.

For hard constraints the package checks whether a weighted average of the actual
prior events could give the requested values. A contradiction may involve a
linear combination of moments even when each target lies inside its own range.
A separating bound is checked on every event before rejecting a target. A
nonconstant feature exactly at its minimum or maximum would require zeroing
events away from that endpoint. The numerical convex-hull check can also be
inconclusive, in which case the package reports that status. A feasible target
can retain a residual from the regularized fit. Inspect the achieved values in
the saved check. Gaussian soft targets with positive widths can trade agreement
for a smaller change to the prior.

The same workflow is available in memory:

```python
from maxent_upgrade import book_targets, fit_targets

targets = book_targets(fixed_order, config, orders=6, cross_section_pb=12.4)
ensemble = fit_targets(prior, targets)
weights = ensemble.arrays()["weight_upgraded"]
```

For arbitrary event functions, covariance from common integration runs or variations
that deliberately change the measurement map, use the [direct API](REFERENCE.md) and
[correlated-input interface](STUDY_API.md). The package also supplies a [joint-moment
example](../examples/custom_observables.py) and an [uncertainty
example](../examples/custom_uncertainties.py).
