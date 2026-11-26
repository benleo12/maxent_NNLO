# maxent_upgrade

Reweight your existing LO+PS events with Chebyshev moments from a fixed-order calculation.
You supply the positive event weights, the observables you measured and their NLO or NNLO
moments. The package fits one replacement weight per original event. Supply scale choices
or numerical replicas to get a set of event weights for each variation as well.

## Fit your own sample

Install from a checkout with `pip install .`. The fit requires only NumPy.

```sh
maxent-upgrade fit prior.npz targets.json upgraded.npz
```

**The prior** contains one array per observable and a strictly positive `weight` per event.
Keep event IDs and any other quantities you will use for analysis in the same file.

**The targets** identify each Chebyshev moment by its observable and degree. Here is a
complete illustrative table for the absolute rapidity of a boson. Replace the rates,
moments and errors with your fixed-order inputs for the same fiducial cuts.

```json
{
  "config": {"born": {"y_abs": {"range": [0, 2.4], "map": "lin"}}},
  "cross_section_pb": 12.4,
  "moments": [
    {"observable": "y_abs", "degree": 1, "value": -0.12, "error": 0.004},
    {"observable": "y_abs", "degree": 2, "value": -0.31, "error": 0.006}
  ]
}
```

Here `y_abs` names the prior column. Both calculations use the same map
`u = 2(y_abs − lo)/(hi − lo) − 1`, clip it to
`[-1,1]`, and evaluate `T_n(cos θ) = cos(nθ)`. Each value is the normalized
fixed-order average of that polynomial. A window and its fixed-order fraction can be
supplied for an observable such as boson recoil. A pair of degrees identifies a joint
moment of two named observables. Supply the degrees your integration
measured, for example only 2 and 4 for an even angular correction. Automatic
selection compares those inputs with the prior using their errors. To fit all supplied degrees with
no numerical errors available, set `"moment_selection": false` in `config` and
omit `error`. [Maps, windows and joint moments](docs/NEW_PROCESSES.md)

**The result** preserves the event order and original arrays. Use `weight_upgraded`
directly in your analysis. It already includes the event's prior weight and sums to
`cross_section_pb`. Omit the cross section for unit-area weights. The fit prints the
resolved moment orders, agreement with their targets and retained effective statistics.

**Uncertainties.** Include varied targets under `variations`, with their labels and
corresponding rates. The same command refits them at the central moment orders.
For example, `weight_scale[i]` contains one scale choice on the entire event sample,
identified by `labels_scale[i]`. Apply the same observable or cut to every member.

## Compute the targets from fixed-order records

If your integrator already exports the table above, go straight to the fit. If it
exports weighted phase-space records, put the map definitions (the contents of
`config` above) in `observables.json`. Use the same column names and cuts on the two
samples. Their event counts and row order can differ.

```sh
maxent-upgrade moments observables.json fixed_order.npz targets.json --orders 6 --cross-section 12.4
maxent-upgrade fit prior.npz targets.json upgraded.npz
```

Fixed-order `weight` may be signed. Optional `weight_scales` rows (central first)
and `scale_labels` produce all scale targets and rates together. The pointwise
numerical errors assume independent records. For correlated subtraction terms,
supply errors or replicas from independent integration runs. Reuse a target file
for another generator with the same observable definitions and fiducial cuts.
[Input conventions and worked examples](docs/NEW_PROCESSES.md)

## Try it on collider events

From the checkout, this fits the first two absolute-rapidity moments to 112,978
Drell–Yan events, including all six noncentral scale choices. You can open the
[supplied target file](examples/own_prior/targets.json) to see the measured degrees,
the fiducial selection and the rate attached to each scale.

```sh
maxent-upgrade fit examples/data/dy_prior.npz examples/own_prior/targets.json dy_weights.npz
```

For the full released Drell–Yan constraint set, use `maxent-upgrade inspect dy` to see
its inputs and `maxent-upgrade reweight dy prior.npz upgraded.npz --uncertainties`.
The other released tables cover charged-current Drell–Yan (`wp`, `wm`), gluon-fusion
Higgs (`ggh`), diphotons (`aa`) and Z+jet (`zj`).

[All paper examples](examples/README.md) · [Event records and HepMC](docs/HEPMC.md) ·
[API reference](docs/REFERENCE.md) · [Fixed-order booking](MOMENT_CONSTRUCTION.md) ·
[Correlated integration inputs](docs/STUDY_API.md) · [Validation](docs/VALIDATION.md) ·
[Paper reproduction](REPRODUCE.md)

## Citation

B. Assi, *Maximum-entropy reweighting of color-singlet samples* (2026).
The manuscript describes the method and the five applications.

For the letter's NNLO Z+jet recoil moments (`load_moments("dy", nnlo_zj=True)`),
also cite B. Assi, R. Poncelet and J. Whitehead, *Positive-weight moment matching
at NNLO and beyond* (2026), IFJPAN-IV-2026-23, MCNET-26-35.

MIT license. See [LICENSE](LICENSE).
