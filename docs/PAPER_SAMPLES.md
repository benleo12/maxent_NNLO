# Shower samples and worked examples

Appendix B of the paper gives the generator settings, selections, event counts
and negative-weight fractions. The accepted fractions for the tighter lepton
cuts in the charged-current study are in Appendix C, under *Production and
decay constraints for W bosons*.

The [example catalogue](../examples/README.md) gives the commands for fitting
the smaller samples distributed with the code. Each sample includes the event
weights to use in the fit. The records below describe how the samples were
selected from the full simulations.

| Process | Samples and calculations |
|---|---|
| Neutral-current Drell–Yan | [MG5_aMC + Pythia](../examples/data/DY_GENERATION.md) and [Herwig](../examples/data/HERWIG_GENERATION.md) generation and sampling. The [two-shower example](../examples/own_prior/README.md) applies the same moments to both priors. |
| Charged-current Drell–Yan | [Charged-lepton samples](../examples/data/w_predictivity/README.md) for both charges and a [worked example](W_CONSTRAINT_TUTORIAL.md) for predicting the acceptance of a tighter cut. The [study protocol](../examples/data/w_predictivity/study_record/PROTOCOL.md) specifies the cuts used for the independent NNLO comparisons. |
| Gluon-fusion Higgs, diphoton and Z + jet | [Event samples and fit commands](../examples/data/applications/README.md), including example cuts on jets and photons. |
| Z + two hard jets | [Selection and fitting procedure](../examples/data/zj_refinement/PROTOCOL.txt) and [results](../examples/data/zj_refinement/RESULTS.txt) for the moment and histogram comparison on the 16,004 selected events. |

To fit an event sample from another generator or for another process, see
[Reweight your own sample](NEW_PROCESSES.md).
