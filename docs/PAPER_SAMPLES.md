# Samples and numerical inventories used in the paper

These counts describe the full samples used in the paper, after their stated
selections. They are distinct from the smaller fixtures distributed with the
runnable examples. The original inventories are retained in
[source form](paper_inventories/tables_v30.tex), including the application
overview, accepted fractions and effective-statistics comparison that were
replaced by plots or consolidated in the shorter paper.

| Process | Sample | Selected events | Negative-weight fraction |
|---|---|---:|---:|
| Drell–Yan | MG5_aMC + Pythia prior | 3.6 million | 0 |
| Drell–Yan | Herwig prior | 570,000 | 0 |
| W+ | MG5_aMC prior | 520,000 | 0 |
| W− | MG5_aMC prior | 490,000 | 0 |
| Gluon-fusion Higgs | Pythia prior | 2.0 million | 0 |
| Gluon-fusion Higgs | MiNNLO_PS | 600,000 | 14% |
| Gluon-fusion Higgs | MC@NLO | 1.0 million | 9% |
| Diphoton | Prior with hard-emission matrix elements | 11 million | 0 |
| Diphoton | Prior with shower-generated radiation | 6.9 million | 0 |
| Z + jet | Drell–Yan prior with at least one jet | 920,000 | 0 |

The numbers retain the precision of the published inventory. The priors have
positive weights. For equal weight magnitudes, a negative-weight fraction
`f_neg` reduces the effective sample size to `(1 - 2*f_neg)^2` times the event
count. For nonuniform weights the effective count must instead be calculated
from the actual weights.

The hard two-jet benchmark is a separate selection: 16,004 events survive the
dilepton and hard-jet cuts from approximately ten million generated Drell–Yan
events. Its [protocol and results](../examples/data/zj_refinement/PROTOCOL.txt)
retain the exact comparison region and sample treatment. The
[W production record](../examples/data/w_predictivity/study_record/PROTOCOL.md)
specifies the full-study baseline, selected constraints and independent tests.

The physical generator, PDF, QED-radiation, jet and hadronization definitions
remain in the paper's Samples appendix. Generation and fixture details are
also supplied for [Drell–Yan](../examples/data/DY_GENERATION.md),
[Herwig](../examples/data/HERWIG_GENERATION.md) and the
[example catalogue](../examples/README.md).

The archived table source uses the equation and section labels of paper v30.
It is a historical source extract, not a standalone document or a claim that
all those tables remain in the current manuscript. The
[technical notes](TECHNICAL_NOTES.md) describe the numerical prescriptions and
supporting records.

The [plot inputs and checks](paper_inventories/plot_inputs_and_checks.json)
preserve the numerical values behind the two replacement plots, including
correlated changes when spin constraints are added. The
[draft review](paper_inventories/DRAFT_REVIEW.json) records their checks before
insertion into the manuscript. File hashes are listed in the
[manifest](paper_inventories/manifest.json).
