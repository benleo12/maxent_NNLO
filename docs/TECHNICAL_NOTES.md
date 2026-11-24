# Numerical prescriptions and supporting checks

This record supplies the detailed prescriptions behind the paper's fits and
the numerical studies supporting them. The shorter paper keeps the physical
argument and its predictions. The solver, inputs and reported results are
unchanged by this documentation release.

For running an analysis, start with the [example catalogue](../examples/README.md),
[W tutorial](W_CONSTRAINT_TUTORIAL.md) or [study interface](STUDY_API.md).
The sections below specify the numerical choices and point to the underlying
records.

## Moment degree and numerical solution

For the baseline individual spectra, compare the fixed-order and prior
moments using the same map and window. Keep all degrees through the highest
computed degree for which the absolute moment difference exceeds the assigned
fixed-order error. That error combines the scatter of independent integration
runs and half the seven-point scale envelope in quadrature. For recoil, the
comparison uses window-weighted moments normalized within the common window.
The rule selects a usable cutoff among the computed moments. It is not a
two-sample significance test or a search for the smallest possible basis.

The solver centers and standardizes the measurement functions with the prior,
then minimizes the convex dual with damped Newton steps and the coefficient
penalty `l2=1e-4`. The gradient is the achieved-minus-target moment vector and
the unpenalized Hessian is the posterior covariance of the functions. A finite
unregularized solution requires targets inside the relative interior of their
convex hull on the prior events. Coefficients are unique after removing
constant and linearly dependent combinations. Regularization can change
moment closure, so residuals are checked against the numerical input errors.

In the reported central-scale jet comparison the largest moment residual is
0.46 times its target statistical error. For the selected W fits every residual
is below 0.05 times its central input error at all seven scale choices. These
are numerical checks of the fitted constraints, not independent tests of the
predicted distributions.

If accumulated recoil moments use a wider map interval, the moments for the
fitted interval are obtained by an exact linear transformation of the
polynomials. Applying it to every integration run and scale member also
transforms the input errors consistently. The independent W and jet studies
accumulate their chosen measurement functions directly. The acoplanarity
moments in the jet application come from integration of the binned NLO result.

## W constraint selection

The [physical functions](../examples/w_physical_inputs.py) and
[selection implementation](../examples/w_selection.py) define the executable
prescription. The full-production
[protocol](../examples/data/w_predictivity/study_record/PROTOCOL.md),
[training record](../examples/data/w_predictivity/study_record/fits160/FINAL_TRAINING_FREEZE.json)
and [independent results](../examples/data/w_predictivity/full_study/README.md)
retain the actual comparisons. The compact examples make their own choices
from their supplied samples.

The production and polar constraints are fixed before the spin sequence. The
rapidity-angle choice tests products through degree four and retains the
smallest joint degree for which the remaining combined discrepancy no longer
satisfies the addition rule. In the full study this adds six degree-two
rapidity-angle functions for W-minus and none for W-plus.

All spin comparisons use direct fixed-order targets for the window probability
and windowed recoil moments. Starting from these production and polar
constraints, the declared sequence tests:

1. Four higher polar moments, of degrees seven through ten.
2. Six mass-angle products, of mass degrees one and two and angular degrees
   zero through two.
3. The hard-recoil spin functions.
4. The recoil-suppressed spin functions.

At each step, the full candidate block is evaluated before being imposed.
Residual covariances include common fixed-order integration fluctuations, the
finite prior and the response of the fitted weights. The same central numerical
covariance is used for all seven coherent scale choices.

The [residual test](../maxent_upgrade/study.py) standardizes by the numerical
standard deviations and diagonalizes the resulting correlation matrix. It
retains eigenvalues above `1e-6` times the largest eigenvalue. A discrepancy is
resolved when the retained components have an approximate chi-squared tail
probability below `0.01` at every scale choice. The standardized norm in the
discarded modes is reported with tolerance `1e-8`. A nonzero discarded component
prevents declaring compatibility. It does not erase a discrepancy already
resolved by the retained modes, but the sequence remains flagged for covariance
review. A block with no retained statistical rank is inconclusive.

After an addition the full fit is repeated. Every imposed residual must lie
within its central input standard error, or the absolute numerical tolerance
`1e-10`, at every scale. Weights must be finite, positive and normalized. Both
alternating-seed halves of the training integrations must select the same
additional blocks. The fit choices precede inspection of the independent
reference. Scale members are coherent theory variations, not statistical
replicas or a calibrated probability of covering the true prediction.

The original production budget and stopping decisions are retained in the
[precision record](../examples/data/w_predictivity/study_record/PRECISION_STOPPING.md).
The [reference record](../examples/data/w_predictivity/reference/README.md)
also retains the three asymmetry bins that miss their precision target and the
valid contribution dominating one bin's variance. The plotted numerical errors
must not be read as calibrated Gaussian significances in that bin.

## Two-jet moment and histogram comparison

Both methods use the same 16,004 hard two-jet events, surviving the dilepton
and jet selections from approximately ten million generated Drell–Yan events.
The fitted region is `60 <= ptj2 <= ptj1 < 1000 GeV`. Each training and
independent reference integration uses 16 million fixed-order points. The
input is LO for two resolved jets, from the real contribution to NLO Z+jet,
with central scale `HT/2`.

The [refinement protocol](../examples/data/zj_refinement/PROTOCOL.txt),
[results](../examples/data/zj_refinement/RESULTS.txt) and
[reproduction receipt](../examples/data/zj_refinement/REPRODUCTION.json)
give the full comparison. Histogram grids start with 2, 3, 4, 5, 7 or 10 bins
per log-momentum axis. Adjacent sparse cells are merged using only original
prior counts and grid geometry until each group contains at least 25 prior
events. The fixed groups give 2, 4, 6, 8, 16 or 28 independent shape constraints.
The protocol records the strict-grid failures and the deterministic tie rule.

Numerical errors in that refinement study use 256 shared resamples of prior
events and training integrations, with all methods refitted. The subsequent
separate-degree interpretation uses 128 resamples and includes independent
reference fluctuations. The paper's correction figure uses the latter moment
points and the former histogram points. Those two ensembles are not paired
with each other. The original reference had already been examined when the
follow-up scans were specified.

The 19 common comparison cells have at least 50 prior events and contain about
97% of the reference probability. The plotted half-sum of absolute probability
differences is not renormalized within those cells. Full-grid and seven-scale
checks are recorded as well. In the original ensemble, refining the histogram
from eight to 28 constraints increases the RMS fluctuation of its correction
shape within populated cells from about 2.4% to 4.4%. The twelve-constraint
moment correction gives about 3.4%. These are not errors on every future cut
acceptance. A sufficiently fine histogram is competitive in this example.

The sparse-tail diagnostic is retained in the public results: extrapolating a
resampled moment fit to an omitted extreme event can give a very large weight
at that position. Precision in the populated cells does not establish precision
there. Higher moment degrees can also increase the fluctuations.

The [correction-origin archive](technical_notes/correction_origin/summary.json)
adds the existing separate-degree results,
[paired physical fractions](technical_notes/correction_origin/physical_fractions.json),
[scale checks](technical_notes/correction_origin/scales.json),
[protocol](technical_notes/correction_origin/protocol.json) and
[aggregate arrays](technical_notes/correction_origin/diagnostics.npz).
Its source scripts preserve the original compute-host paths and are analysis
records, not new installed entry points. The runnable package examples remain
in `examples/`. No new event production or integrations were performed to
assemble this archive.

## Response bounds and conditioning

The [archived source extracts](technical_notes/paper_details_v29.tex) preserve
the derivations and detailed explanations from manuscript v29, including the
exact covariance response, the small-correction expansion, its tighter
projection bound, the unconstrained-error decomposition and polynomial
conditioning. The extracts retain the original labels and are not a standalone
LaTeX document. The accompanying diagnostic figures are:

- [Linear response and its bound](technical_notes/fig_transport_v3.pdf).
- [Mapped densities and polynomial conditioning](technical_notes/fig_conditioning_v3.pdf).
- [Weight concentration for different priors](technical_notes/fig_cost_short_v3.pdf).

The response plot tests an approximation to the shift induced by reweighting.
It does not test agreement with QCD. In the cases retaining more than 90% of
their effective statistics the linear estimate has the correct sign and is
within 35% of the observed mean shift. The large shower-only diphoton correction
is less well described. This expansion about the prior is distinct from the
differential response of the fitted solution used to propagate numerical input
errors in the [study workflow](STUDY_API.md).

Complete polynomial bases through the same degree impose equivalent exact
constraints when targets and covariances are transformed consistently. In the
reported twelve-moment Drell–Yan mass example the prior correlation-matrix
condition number is near ten for Chebyshev polynomials and above `1e7` for
monomials. Standardization and a diagonal coefficient penalty are basis
dependent, so finite regularized implementations can differ. A change of basis
alone does not reduce the physical integration uncertainty.

Matching the logarithmic recoil map to the fitted interval keeps its
polynomials distinguishable over that interval. Extending the twenty-moment
map to 2500 GeV compresses the fitted region and places seven eigenvalues below
the regularization scale in the reported example. The shorter map's moments
are exact linear combinations of the accumulated ones.

The [archive manifest](technical_notes/archive_manifest.json) records checksums,
the source manuscript hash and the unchanged numerical-code revision. The full
tutorial text is also preserved in the source extracts. Current usage belongs
in the [example catalogue](../examples/README.md),
[event-record documentation](HEPMC.md) and [interface reference](REFERENCE.md).
