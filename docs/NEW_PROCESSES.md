# New processes and observables

The solver has no process-specific branches. It needs a positive prior, a set of
measurement functions, and their target expectation values. The released process
names belong to the table loader, not to the reweighting algorithm.

[Run the complete synthetic example](../examples/custom_observables.py) first.
It shows two custom observables, their separate moments and their joint moments.
For an existing released process, you can instead select additional constraints
already present in its table. See [the paper examples](../examples/README.md).

## 1. Define the event arrays

Choose observables that your prior and fixed-order calculation can evaluate with
the same definitions, cuts and units. Each dictionary holds one value per event:

```python
prior = {
    "mass": prior_mass,
    "cos_theta": prior_angle,
    "weight": prior_weights,
}
fixed_order = {
    "mass": fo_mass,
    "cos_theta": fo_angle,
    "weight": fo_weights,
}
```

The two samples may have different sizes. Prior weights must be strictly positive.
Fixed-order phase-space weights may have either sign. The prior must populate the
phase space needed by the target, including configurations that can cross analysis
cuts during showering. Names such as `mass` are ordinary dictionary keys. Rename
or add them freely, and use the same names in `config`.

Extra prior arrays are carried along for your analysis. They receive the same
weights but supply no constraints. For example, keeping `jet_multiplicity` in
`prior` does not request a fixed-order calculation of it.

## 2. Choose where fixed order is reliable

For a Born observable with a reliable fixed-order distribution over its selected
range, give an interval and a map:

```python
config = {
    "born": {
        "mass": {"range": (60.0, 140.0), "map": "lin"},
        "cos_theta": {"range": (-1.0, 1.0), "map": "lin"},
    }
}
```

The available maps are `lin`, `log` and `("bw", mass, width)`. The logarithmic map
requires a positive interval. Values outside the interval are clipped, identically
on the prior and fixed-order sides. Their detailed distribution there is not
resolved by that set of moments.

For a recoil observable, use fixed order only in a suitable window. This example
turns it on between 20 and 40 GeV and ends the constrained range at 500 GeV:

```python
config["recoil"] = {
    "recoil_pt": {
        "range": (20.0, 500.0),
        "map": "log",
        "soft_lo": 20.0,
        "profile": {"a": 20.0, "b": 40.0, "c": 500.0},
    }
}
```

Add `recoil_pt` arrays to both samples. The default `blend` transition combines
fixed-order moments in the window with complementary moments from the prior.
Above the hard cutoff, the implemented target is the **rescaled prior share**,
not a separately supplied fixed-order tail rate. The recoil measurement functions
are constant below the lower edge. Constraints on other correlated observables
can still affect the marginal distribution there.

The window is a physics choice made when computing the input moments. It is not
inferred from a variable name or freely adjustable afterwards. The color-singlet
matching-scale criterion in `matching_scale` is a guide for the applications in
the paper, not a universal prescription for arbitrary observables. Born observables
can also need restricted windows near acceptance edges, as in the W example.

## 3. Supply the moments

If the fixed-order points are available in Python:

```python
from maxent_upgrade import compute_fo_moments, upgrade

moments = compute_fo_moments(fixed_order, config, n_max=8)
result = upgrade(prior, moments, config)
print(result.describe())
```

Otherwise accumulate the same weighted sums inside the fixed-order integrator.
For each observable, map its value to `u` and accumulate
`sum(weight * T_n(u)) / sum(weight)`. Recoil and joint constraints include their
specified window factors. The [fixed-order recipe](../MOMENT_CONSTRUCTION.md)
and [NNLOJET patch](../nnlojet_patch/README.md) give the exact definitions.

The solver accepts the resulting plain dictionary. For example, a Born block is:

```python
moments = {
    "born": {
        "mass": {
            "values": mass_moments,
            "stat_errors": mass_mc_errors,
            "scale_errors": mass_scale_errors,
            "errors": np.hypot(mass_mc_errors, mass_scale_errors),
        },
        "cos_theta": {
            "values": angle_moments,
            "stat_errors": angle_mc_errors,
            "scale_errors": angle_scale_errors,
            "errors": np.hypot(angle_mc_errors, angle_scale_errors),
        },
    }
}
```

Lists start at degree one. The normalization is separate. Errors determine how
many moments are resolved for the user's prior, so production inputs must include
reliable uncertainty estimates. `compute_fo_moments` uses the weighted-point error
estimate. For correlated real-emission and subtraction points, estimate statistical
errors using the integrator's independent runs, as in the released tables.

## 4. Add correlations when needed

Separate moments constrain separate distributions. To supply information about
an angular distribution at different masses, add products of their polynomials:

```python
config["mixed"] = {
    "mass_angle": {
        "observables": ("mass", "cos_theta"),
        "range": {"mass": (60.0, 140.0), "cos_theta": (-1.0, 1.0)},
        "map": {"mass": "lin", "cos_theta": "lin"},
        "n": 2,
    }
}
```

This requests four products with degrees `(1, 1)`, `(1, 2)`, `(2, 1)` and `(2, 2)`.
Recompute the moments with this configuration before reweighting.

When a joint block involves recoil windows, give the same profiles in its
`profile` dictionary and set each coupled recoil constraint's `transition` to
`"windowed"`. The joint measurement functions then contain the product of both
windows, and the corresponding targets come from fixed order. The solver rejects
a joint block coupled to a blended recoil constraint. The Z+jet tables already
supply a consistent configuration.

## 5. Use the weights in an analysis

```python
weights_pb = result.weights * target_cross_section_pb
counts_pb, edges = np.histogram(prior["mass"], bins=mass_edges, weights=weights_pb)
```

Every returned weight belongs to the input event at the same index. Apply it to
that event's complete record, including its detector response if available.
`result.weights` sums to one and already includes the prior's relative weights.
The ratio to the normalized prior is useful when an analysis accepts multiplicative
factors, but absolute predictions also require the target cross section.

A cut applied afterwards selects a subset of these weights. Do not renormalize the
subset unless a shape-only result is intended. Accurate acceptance also depends on
the relevant joint distributions and on any unconstrained decay or radiation
variables. Correct individual marginals alone do not guarantee it.

## 6. Automate repeated use

Once the observable definitions and target moments are fixed, the same call can
process samples from different generators, tunes or production batches:

```python
def reweight_sample(events, moments, config, sigma_pb):
    result = upgrade(events, moments, config)
    print(result.describe())
    return result.weights * sigma_pb
```

Retain the exact configuration with the moments. Save the event ordering and the
solver's diagnostic output with each result. The selected moment count can differ
between priors because the correction required by each prior differs.

For uncertainty re-solves on one prior, keep the central set of moment orders:

```python
fixed_orders = result.report["moment_selection"]["chosen"]
shifted_config = {**config, "fixed_orders": fixed_orders, "lam0": result.report["lam"]}
shifted_result = upgrade(prior, shifted_moments, shifted_config)
```

Use a cold start when a changed map or window changes the measurement functions.
The [band example](../examples/bands_with_released_moments.py) demonstrates scale,
input-statistical, transition and PDF variations using the released tables.

Automation covers the numerical reweighting. The user specifies the physics input:
which observables to constrain, where fixed order is reliable, and whether the
prior describes the remaining correlations adequately. A converged solve verifies
the constraints, not the accuracy of every observable in the event.
