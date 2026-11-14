# Public QCD inputs

## Drell–Yan fitting example

`dy_prior.npz` contains 112,978 real shower events with original event IDs and
inverse-inclusion sampling weights. `dy_reference.npz` contains per-run
fixed-order histograms, including a decay-angle integration independent of the
moment constraints. Both files and their JSON provenance are committed.
See [DY_GENERATION.md](DY_GENERATION.md) for generator settings, selection,
units, the frozen sampling rule and reference construction. Run the new fit
with `python examples/real_drell_yan.py --output dy_demo --plot`.

## Independent generator input

`herwig_public_prior.npz` contains 98,808 selected Herwig events with generation
weights, inclusion probabilities, native and global IDs, and lepton four-vectors.
Pass it through the same `real_drell_yan.py --prior` workflow. The
[Herwig recipe](HERWIG_GENERATION.md) records the generator settings, seed rule
and the distinction between compact demonstrations and full-production plots.

## Two-jet validation histogram

`zj_joint_validation.npz` accompanies the JHEP comparison of separate and joint
jet moments. It contains binned sums, not event records. Run:

```bash
python examples/plot_joint_validation.py
```

The prior is the paper's Drell–Yan LO+PS sample without matrix-element corrections
or generation-level lepton cuts, selected to contain at least one jet (914,685
events). Jets use anti-kT, R=0.4, before hadronization and after shower and MPI.
The arrays `prior`, `marg` and `joint` contain weight sums with the same full-sample
unit normalization. Their `_w2` arrays contain sums of squared event weights.
`counts` gives the unweighted bin counts. The two posterior configurations are
`zj-no-joint` and `zj` in `paper_examples.py`.

Both axes use edges [60, 90, 140, 220, 400, 1000] GeV, selected before integrating
the reference. Axis 0 is the leading-jet momentum and axis 1 the second jet.
The plot normalizes each distribution over this displayed region, with no extra
normalization after masking sparse cells. Hatched cells have fewer than ten
effective posterior events, `(sum w)^2 / sum w^2`.

`fo_per_run_fb[run, leading_bin, second_bin, scale]` contains **cross sections per
cell in fb**, not densities. The reference consists of 16 independent one-million
point NNLOJET v1.0.2 integrations, seeds 901..916, performed on 13 November 2026.
It is the real-emission contribution to the paper's NLO Z+jet input. With two jets
above 60 GeV, only this contribution survives, so the reference is LO Z+2 jets.
It is independent of the moment integrations and was not used to fit the weights.

The physics inputs are those of `moments/zj_nlo_moments.json`: 13 TeV,
CT18NNLO, Gmu scheme, 66 < mll < 116 GeV, lepton pT > 27 GeV and |eta| < 2.5,
jet |y| < 4.4 and lepton–jet separation > 0.4. Scale choices (muF,muR)/HT are
(0.5,0.5), (1,1), (0.25,0.25), (0.5,0.25), (0.25,0.5), (0.5,1), (1,0.5).
The stored `fo_internal_errors_fb` are individual-run integration errors. Use the
scatter of independent runs for a combined statistical error, including the
covariance with normalization for a shape. Colors in the plot show central ratios.
