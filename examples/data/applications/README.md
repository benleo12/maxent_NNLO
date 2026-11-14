# Real Higgs, diphoton and Z+jet examples

These compact event samples make the remaining three paper applications runnable
without generating new events or requesting an authors' archive:

```bash
python examples/application_demo.py ggh --output ggh_demo --plot
python examples/application_demo.py aa --output aa_demo --plot
python examples/application_demo.py zj --output zj_demo --plot
```

Install with `python -m pip install '.[plot]'` first. Each command performs a new
fit to the released moments, followed by six coherent scale variations, twenty
common-run integration replicas and two transition variations. The Higgs input
also supplies 58 PDF members. Missing PDF/shower sources are recorded explicitly.
`--prior PATH` replaces the bundled sample; `--bootstrap N` controls the number
of integration replicas.

The outputs are `posterior.npz` (all original arrays and absolute pb weights),
two histogram files, `acceptance.npz`, `report.json` and, with `--plot`,
`comparison.pdf/png`. Every member has its own absolute normalization.

| Application | Constrained distribution | Further analysis |
|---|---|---|
| Higgs | Higgs transverse momentum | Leading-jet momentum and the stored photon fiducial selection |
| Diphotons | Pair transverse momentum | Photon separation and a subleading-photon threshold of 40 GeV |
| Z+jet | Second-jet momentum | Jet balance and a second-jet threshold of 60 GeV |

The extra distributions/cuts are predictions. **These plots have no independent
reference**, so they show the effect of reweighting and propagated inputs, not a
physics validation. Scale bands and target-integration errors appear in the
upper panels; transition/PDF inputs and fixed-weight event errors appear below.
The full refitted-prior covariance is available through the separate
[constraint-study workflow](../../../docs/STUDY_API.md), not this short interface.

The Higgs example uses the released NNLO rate, about 39.47 pb. The paper's Higgs
plots use a separate N3LO normalization of 48.58 pb; the compact example does not
silently substitute that rate. The diphoton mass map clips outside [80,700] GeV
as in its booked measurement functions; this interval is not an additional
fiducial cut. The compact Z+jet fit uses 200,000 events and the unchanged default
regularization; its central residual is below one integration MC error.
Its hardest bins still need a local-statistics check.
`input_agreement` records every central/varied fit, while `local_statistics`
records counts, effective counts and largest-event concentration. Inspect both
before using a tail prediction.

## Sampling and generation provenance

The Higgs and diphoton files contain 80,000 rows, and Z+jet contains 200,000,
selected uniformly without replacement using
seed731132 over the concatenated valid original rows in basename-sorted source
order. This rule was fixed before fitting. The
[exporter](../../export_application_fixture.py) preserves source-file index,
source row, unique event ID, original weight, source normalization and inclusion
probability. The JSON companion records every original file hash and count,
generator cross-section metadata and the final fixture hash.

The fit weight is exactly
`original_weight * source_normalization / inclusion_probability`.
For diphotons, **each source has its own `sigma_pb/n_generated` factor** before
concatenation. Omitting those factors would change the mixture of exclusive
hardness slices and merged/box contributions. Higgs and Z+jet retain the relative
weights used by the original analyses; the fitted target sets the final rate.

- **Higgs:** four existing Pythia on-shell gluon-fusion chunks, 2,000,000 inclusive
  events in total. The sample retains Higgs/leading-jet kinematics and the stored
  photon fiducial flag. The latter is applied after the inclusive fit.
- **Diphotons:** the existing merged MG5+Pythia sample and its loop-induced box
  contributions, 10,644,601 positive finite selected rows across nine sources.
  The original photon four-vectors are retained. The source generation already
  applies the paper's fiducial/isolation definitions; source normalization and
  merging metadata remain in the JSON companions.
- **Z+jet:** the no-MEC parton-level jet analysis of the existing DY production,
  with the original lepton selection and at least one selected jet; 914,685
  valid source rows. Jet definitions are anti-kT R=0.4, pT>20 GeV, |y|<4.4 and
  lepton separation>0.4. `gidx` retains the original generator association.

The compact fits have different statistics from the full paper productions.
Uniform sampling avoids adding new kinematic quotas; it does not guarantee
that all tails remain well populated. No full source archive is needed to run
these examples, and source paths in historical metadata are not runtime inputs.

The Z+jet size was chosen after a declared numerical-quality comparison of
80,000 and 200,000 uniform rows, at fixed constraint orders and three ridge
values. The smaller default-ridge fit left a 3.30-MC-error residual; the larger
one reduced it to 0.652 while retaining 91.1% of prior effective statistics.
Lowering the ridge on the smaller sample was less effective and the smallest
value failed. The complete [comparison](zj_sample_size_check.json), including
that failure, is retained. No independent prediction reference was used to
choose the size, and neither full-paper settings nor default regularization
were changed.

Continuous jet spectra exclude absent-jet sentinel values and require pT>20
GeV. Jet balance requires a second jet above 60 GeV. These are absolute spectra
with the original fitted normalization, not conditional shapes normalized to
the selected events. The report also records the unselected fractions.
