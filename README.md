# maxent_upgrade

Use fixed-order QCD moments to reweight an existing positive-weight shower sample. The program returns one replacement weight per event and a set of weights for each supplied variation. Event IDs, kinematics and any detector simulation can be reused.

## Reweight your events

Install the package from a checkout with `pip install .`. Put the prior weights in the `weight` column of an NPZ file and give each observable its own column. Keep any event IDs and quantities needed for later analysis in the same file. Use identical observable definitions, units and fiducial cuts for the prior and fixed order.

```sh
maxent-upgrade match prior.npz calculation.json upgraded.npz
maxent-upgrade check upgraded.npz --selection
maxent-upgrade check upgraded.npz
```

**For a new fixed-order calculation**, the calculation file pairs each event observable with its NNLOJET name and specifies the map, interval and polynomial degrees. A product moment identifies both observables and both degrees. The run card supplies the process, cuts, PDFs and scales. Install the NNLOJET moment patch once. The command then integrates the cross section and the requested moments together, saves a reusable table and fits the event weights. [Run card and working calculation file](examples/nnlojet_moments/README.md)

**For a calculation supplied with the code**, replace `calculation.json` with its identifier. This route needs only NumPy. For example, try the released Drell–Yan inputs on the accompanying events:

```sh
maxent-upgrade match examples/data/dy_prior.npz dy upgraded_dy.npz
```

Use `maxent-upgrade inspect dy` to see the cuts and required event columns before using your own generator. Available calculations cover neutral- and charged-current Drell–Yan (`dy`, `wp`, `wm`), gluon-fusion Higgs (`ggh`), diphoton (`aa`) and Z+jet (`zj`) production. [All paper examples](examples/README.md)

**If you have your own moments**, use `maxent-upgrade fit prior.npz targets.json upgraded.npz`. Each table entry gives the observable ID, Chebyshev degree, normalized fixed-order average and numerical or combined error. Include the fiducial cross section for weights in pb. You can also accumulate these inputs in another integrator, including correlations between subtraction terms. [Own inputs, joint moments and correlated errors](docs/NEW_PROCESSES.md)

## Use the weights

For each individual observable, the program compares the prior and fixed-order moments with their quoted errors to choose the polynomial degrees. The first check above shows that comparison for all supplied degrees. The second shows the target and achieved values for the imposed constraints. The fit also checks support and reports the effective number of events. Explicitly requested joint moments enter together.

`weight_upgraded` already includes the prior weight. Sum it over a bin or cut. With a fiducial rate supplied, the result is in pb. The corresponding sum of each row of `weight_scale` gives the same prediction for the scale choice in `labels_scale`. For a new integration use `--replicas N` to propagate correlated numerical errors in additional fits. Scale and integration errors are reported separately. The fit can change predictions for other observables through their correlations in the prior.

The [HepMC guide](docs/HEPMC.md) explains how to measure your own observables and attach the fitted weights to the original records. A [Higgs example with Rivet](examples/own_analysis/README.md) compares fiducial cross sections and differential bins with direct sums after photon cuts.

[API reference](docs/REFERENCE.md) · [Moment definitions](MOMENT_CONSTRUCTION.md) · [Validation](docs/VALIDATION.md) · [Paper reproduction](REPRODUCE.md)

## Citation

B. Assi, *Maximum-entropy reweighting of color-singlet samples* (2026).

For the letter's NNLO Z+jet recoil moments (`load_moments("dy", nnlo_zj=True)`), also cite B. Assi, R. Poncelet and J. Whitehead, *Positive-weight moment matching at NNLO and beyond* (2026), IFJPAN-IV-2026-23, MCNET-26-35.

MIT license. See [LICENSE](LICENSE).
