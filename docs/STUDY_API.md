# A supported constraint-study workflow

`NumericalInput`, `ConstraintBlock`, `fit_constraints`, `plan_observable`,
`load_frozen` and `validate_prediction` provide the scientific workflow used by
the [small executable tutorial](CONSTRAINT_TUTORIAL.md). They do not define a
new fit objective: `fit.central` is the existing `UpgradeResult`, and all solves
use the existing prior-standardized maximum-entropy solver. Existing moment
tables, `upgrade` configurations and uncertainty ensembles remain supported.

## Numerical inputs

```python
from maxent_upgrade import NumericalInput

training = NumericalInput(
    labels, mean, covariance, sample_id="training-production-001",
    role="training", normalization=0, scale_names=scale_names,
    metadata={"process": process, "cuts": cuts, "sources": source_hashes},
    run_ids=run_ids,
)
```

`mean` is `(integral, coherent input)`; column zero is central. The normalization
entry is a positive cross section in pb at every coherent input. `covariance`
is the complete **central covariance of the raw integral estimates**. The
normalization integral, fitted functions and diagnostic functions must share
that vector. There is no fallback that silently drops their cross-covariance.
Scales remain deterministic alternatives; they are not treated as additional
Gaussian observations.

`training.normalized` contains integrals divided by their own coherent total
rate. `normalization_jacobian` and `normalized_covariance` include the common
denominator. The original raw covariance is retained for absolute predictions.

For independent equal-budget runs within each additive channel:

```python
training = NumericalInput.from_runs(
    labels, runs, seeds=seeds, channels=channels,
    sample_id="training-production-001", scale_names=scale_names,
    metadata=production_metadata,
)
```

`runs` has shape `(channel, padded run, integral, coherent input)` and `seeds`
has shape `(channel, padded run)`. Each real row must be finite. Padding is all
NaN with integer seed `-1`; unequal channel counts are supported. Each channel
needs at least two independent runs. Channel means and covariances of means are
summed, which assumes independent additive contributions. Seeds must be globally
distinct unless genuinely independent RNG `stream_namespaces` are explicitly
declared per channel. Merely renaming a channel does not establish independence.

Different precision batches for the **same** physical contribution are not
additive channels and must not be summed as such. Correlated channels or
unequal-budget estimates need appropriate external pooling and a complete
covariance passed to the direct constructor.

`bootstrap_runs(runs, seeds=..., channels=..., replicas=..., seed=...)` resamples
whole central vectors within channels, preserving normalization and observable
correlations. `prediction.numerical_replicas({input.fingerprint: replicas})`
evaluates the first-order fit response on those common raw draws. This is not
an exact nonlinear refit of every replica; use actual re-solves to check the
linear approximation when shifts are large or constraints are poorly conditioned.

## Physical blocks

Directly booked functions use `ConstraintBlock.from_input(name, features,
training, labels=..., metadata=...)`. `features` is `(event, function)`, in the
same event order as the prior. The named raw integral is divided by the total
rate, so a windowed function is booked as `w*f`, with its window rate booked
once if required.

The advanced constructor represents targets as `offset + matrix @ r`, where
`r` is the normalized common vector. Its optional `prior_derivative[i,k]` is
the derivative of target `k` with respect to normalized prior probability `p_i`,
evaluated at the central numerical input. Additive constants in this derivative
do not matter; it is centered internally. This term is required for a
prior-dependent blended continuation. Feature definitions themselves must be
fixed under resampling; prior-adaptive maps are not supported by this response.

Record the physical definitions, units, frame convention, maps, windows and any
angular basis in block metadata. Their numerical event columns, target map and
ordering are also hashed. An arbitrary-feature interface does not make an
inconsistent set of marginal/joint windows physically compatible. In particular,
direct joint recoil constraints should use a compatible direct windowed recoil
anchor; changes from a blended continuation should be a separate control.

`fit_constraints(prior_weights, training, blocks, prior_id=..., event_ids=...)`
returns positive unit-area weights, coherent-input solves and `fit.central`.
There is no blind automatic block selection. Define and inspect a physically
motivated candidate sequence, then record the actual decision. The target rate
multiplies the fitted shape for absolute predictions.

## What the statistical response includes

Write the fitted functions as `Phi`, their covariance under the posterior as
`Cq`, and the prior feature variances as `vp`. In physical feature units the
response Hessian is `Cq + l2*diag(vp)`. A predicted function `h` responds to target
changes through `Cov_q(h,Phi) @ inverse(Hessian)`. The default scalar ridge is
the same `1e-4` used by `upgrade`.

`fit.diagnose(block)` subtracts the supplied diagnostic target **before** forming
the covariance. Thus the numerical derivative is `(response @ L - S) @ J`, with
`L` the fitted-target map, `S` the diagnostic-target map and `J` the raw-to-
normalized Jacobian. The cross-covariance between fitted and diagnostic targets
is retained automatically. The finite-prior term includes refitting, declared
prior-dependent targets and the derivative of the solver's empirical-prior
standardization/ridge.

This is a first-order numerical calculation, not a theory-uncertainty or coverage
model. Prior sampling uses independent weighted event-row/Poisson influence
covariance. Fixed sampling quotas, correlated records, integration subtraction
groups or other sampling designs require their own validated covariance or
replica treatment. Inverse-inclusion weights alone do not specify that design.

The central small Hessian and feature means are cached per unchanged fit. Large
event-by-feature response arrays are not cached a second time for every
histogram. At `l2=0`, dependent features cause a clear response error instead of
an unreported pseudoinverse. Unsupported constant features are rejected.

Input closure is separate from this diagnostic. Nonzero residuals with zero
input variance are recorded explicitly and have no finite MC pull. A fitted
column is removed from a diagnostic only after verifying its event function and
target definition, not merely its name.

`residual_test` normalizes the covariance to a correlation matrix and reports
the retained rank, eigenvalues, cutoff and discarded residual components. The
default rank cutoff is `1e-6`, a declared numerical choice rather than a universal
calibration. Nonzero discarded-mode residuals prevent a clean unresolved
conclusion. A block with no unfitted columns is not a validation test. Coherent
input residuals are assessed separately using the central statistical covariance;
the default training rule requires `p < 0.01` at every input variation to call
the supplied block resolved.

## Freeze, then predict

```python
from maxent_upgrade import plan_observable, validate_prediction

plan = {"lepton_eta": plan_observable(
    eta_bin_indicators, labels=eta_labels,
    definition={"edges": edges.tolist(), "cuts": declared_cuts},
)}
fit.freeze("fit.npz", selection=all_candidate_outcomes, validation_plan=plan)
prediction = fit.predict(
    eta_bin_indicators, labels=eta_labels, definition=plan["lepton_eta"]["definition"],
    validation_key="lepton_eta",
)
comparison = validate_prediction(
    prediction, independent_reference, independent_of=[training.sample_id],
    definition=plan["lepton_eta"]["definition"],
)
```

The plan records actual event-column hashes as well as labels and physical
definitions. Reusing a freeze with changed cuts or bins fails. A freeze also
records every input/feature hash, event IDs/weights, ridge, implementation,
selection decision, normalization and timestamp. Supply all candidate outcomes,
thresholds and controls in the selection record. Existing freeze files cannot
be overwritten. `load_frozen(path, prior_weights, training, blocks, prior_id=...,
event_ids=...)` verifies reconstruction and restores weights **without solving**.

Independent references have `role="validation"`, a distinct sample identity and
their own run IDs. The validation call requires an explicit independence
declaration and rejects known run overlap. The software verifies the declared
inputs and unchanged choices; it cannot establish that an analyst never looked
at an external reference. It never uses held-out predictions to select a block.

## Derived predictions and paired changes

`Prediction.mean` is `(coherent input, observable)`. Its separate
`covariance_numerical` and `covariance_prior` matrices include correlations
between observable bins. `new_prediction.difference(anchor_prediction)` pairs
shared target and event responses before forming the difference covariance.

`stack_predictions([...])` preserves common-source cross-covariance. Use
`prediction.transform(function, jacobian, labels=..., definition=...)` for a
ratio, acceptance or charge asymmetry. The function acts on all coherent rows;
the supplied Jacobian acts at the central point. Validate the underlying raw
integrals first, then transform prediction and reference. Deterministic scale
vectors remain matched across charge samples and models.

Shared numerical samples with different partial definitions are rejected: use a
full common vector. Distinct samples with overlapping run IDs are also rejected.
Shared event samples must have identical event ordering and row sets; align IDs
before pairing, rather than accidentally treating reordered events as independent.

## Reports

`maxent_upgrade.study_report.write_study_report` accepts fitted candidates, their
diagnostics and one common binned validation comparison. The executable tutorial
shows a complete call. It saves PDF/PNG figures, a lightweight local HTML page,
strict JSON and NPZ arrays. All candidates remain in the numerical report;
the spectrum defaults to prior, anchor and selected fit. Larger training scans
are paginated. No plotting dependency is imported during numerical fitting.

Red denotes the selected fit, blue the comparison anchor, gray the starting
shape and black the independent reference. Input-variation envelopes, numerical
covariance and independent-reference errors keep separate meanings. Local
sample counts and concentration should accompany any physically sparse region;
the existing `bin_diagnostics` helper supplies them without claiming support
certification.
