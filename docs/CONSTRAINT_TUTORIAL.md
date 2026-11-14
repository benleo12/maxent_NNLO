# From a missing correlation to an independent prediction

Start with a small example that runs without external data:

```bash
python -m pip install '.[plot]'
python examples/constraint_workflow.py output/constraint_tutorial
```

Open `output/constraint_tutorial/report/report.html`. The command fits three
candidates, chooses between the declared constraint blocks using training data,
saves that decision, then evaluates an independent reference. To run the two
stages separately:

```bash
python examples/constraint_workflow.py output/separate_stages --stage train
python examples/constraint_workflow.py output/separate_stages --stage validate
```

Use a new output directory for a new training decision. Existing frozen records
are not overwritten. Validation can be repeated from the saved records without
refitting or reopening the selection.

## What the example teaches

The example is a controlled massless Born decay, **not an NNLO calculation or a
shower sample**. A boson of mass 80 GeV has production rapidity `y` and polar
angle `c = cos(theta)`. Its daughter has
`eta_l = y + atanh(c)` and `pT_l = 40 sqrt(1-c*c)` GeV. The known target density
contains an odd angular contribution proportional to `y*c`. The starting
density lacks that correlation.

The candidates have clear roles:

| Candidate | Information imposed |
|---|---|
| Prior | The target total rate; the starting shape is retained |
| Separate moments | Separate production-width and even polar-angle moments |
| Joint moments | The same moments, plus the rapidity-dependent odd angular term |

The supplied training diagnostic also contains higher rapidity–angle products.
It tests the information left after each fit, using the covariance shared by
all integrals and the response of the fitted weights. A fitted column is omitted
from this test: matching an input is a different check from predicting a new
quantity. Every candidate remains in the report, including the candidates that
were not selected.

The independent prediction is the lepton pseudorapidity spectrum after
`pT_l > 27 GeV`. No lepton-bin or acceptance indicator is used as a fitted
constraint. Separate marginals can move this prediction in the wrong direction
because pseudorapidity depends on the production–decay correlation. Adding the
declared joint term changes that correlation. The remaining disagreement and
the finite statistics in the tails stay visible.

Three coherent input columns illustrate how deterministic variations travel
through the workflow. They are production-shape variations of this controlled
density, **not physical renormalization/factorization-scale variations**. Real
QCD inputs can supply their coherent scale vectors through the same interface.

## Read the outputs

| File | Purpose |
|---|---|
| `weights.npz` | Original IDs and weights; absolute pb weights for every candidate and coherent input |
| `candidate_0.npz`, … | Frozen weights, multipliers, input/feature hashes and the declared validation plan |
| `selection.json` | Every candidate diagnostic and the training-only decision |
| `training.npz`, `validation.npz` | Distinct common-run integrals, full covariance, seed arrays and source metadata |
| `report/training.pdf` | Imposed-input agreement and remaining correlation evidence |
| `report/predictions.pdf` | Independent spectrum, matched-input ratios and a paired model-change panel |
| `report/report.json`, `report/arrays.npz` | Values, separate covariance components, source responses and provenance |

Event weights in `weights.npz` are **pb per event**. Sum them directly for an
absolute histogram; do not apply the old sample's cross-section normalization
again. `metadata` lists the candidate and coherent-input ordering.

The selected fit is red, its comparison anchor blue, the starting shape gray
and the independent reference black. Error bars on a prediction combine its
target-integration error and its refitted-prior sampling error. The reference
error is separate. Filled bands show coherent input variations, never a
Gaussian contribution added to those statistical errors.

The lower panel is a *paired* difference between two fits to the same inputs and
events. It subtracts their responses before forming the covariance. Adding the
two fits' errors as if they were independent would give a different answer.

Local counts, effective counts and largest-event variance fractions are saved
under `provenance.local_statistics`. They describe the supplied sample; they do
not establish that an unobserved region is adequately populated or modeled.

## Use your own process

Keep the workflow and replace the example's event columns and integrals. Book
the candidate physical functions and their diagnostics in a common integration
vector, including the total rate. Preserve the full covariance, not just an
error on each moment. Choose the maps, windows and candidate blocks before
opening the independent prediction sample.

Most directly booked blocks need no hand-built selector matrix:

```python
from maxent_upgrade import ConstraintBlock, fit_constraints

angular = ConstraintBlock.from_input(
    "polarization", event_angular_functions, training,
    labels=("spin_0", "spin_1", "spin_2"),
    metadata={"definition": "declared angular functions", "window": angular_window},
)
fit = fit_constraints(
    events["weight"], training, [production, angular],
    prior_id="my-generator-sample", event_ids=events["event_id"],
)
print(fit.central.summary())
diagnostic = fit.diagnose(unfitted_correlations)
```

The existing `upgrade(events, moments, config)` remains the short path for
standard moment towers and released applications. The explicit-block interface
is useful when a study needs arbitrary physical functions and their complete
common-run covariance. Both use the same solver and default regularization.
See the [advanced workflow reference](STUDY_API.md) for normalized targets,
blended targets, coherent resampling and derived observables.
