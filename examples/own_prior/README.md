# Use the same recoil moments with two generators

The input table [dy_recoil.json](dy_recoil.json) applies to 13 TeV neutral-current Drell–Yan production with Born-level leptons. Both leptons must satisfy `pT > 27 GeV` and `|eta| < 2.5`, with a dilepton mass between 66 and 116 GeV. Each shower file supplies `weight` and the dilepton recoil `pT_ll` in GeV. The other event columns are available for subsequent cuts.

The target values come from the degree-one `pT_ll` recoil tower and its window fraction in the [released Drell–Yan integration](../../moments/dy_nnlo_moments.json). The table uses a window that rises between 30 and 60 GeV and ends at 500 GeV. It contains the window share and the first Chebyshev moment averaged inside the window. The inclusive NNLO calculation supplies the fiducial rate. The nonzero recoil region starts one order later. All six noncentral scale choices have their own moments and fiducial rates.

Run these commands from the repository root:

```sh
maxent-upgrade fit examples/data/dy_prior.npz examples/own_prior/dy_recoil.json pythia.npz
maxent-upgrade fit examples/data/herwig_public_prior.npz examples/own_prior/dy_recoil.json herwig.npz
maxent-upgrade check herwig.npz
python examples/own_prior/recoil_check.py pythia.npz herwig.npz
```

The two constraints are the window share and the window-weighted polynomial averaged over all events. Divide the latter by the share to recover the shape moment specified in the target file. The target window share is 0.10301. The script also sums event weights for two selections that do not enter either fit. Central cross sections are in pb. Both original priors use the fitted fiducial normalization for this comparison.

| Selection | MG5+Pythia prior | Reweighted | Herwig prior | Reweighted | Fixed order |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dilepton recoil in the six bins from 60.6155 to 500 GeV | 35.694 | 39.139 | 34.992 | 39.071 | 38.618 |
| Absolute Collins–Soper cosine from 0.9 to 1 | 5.098 | 5.586 | 4.333 | 4.829 | 4.431 |

The recoil cross check shares integration inputs with the target moments. The angle reference comes from a separate integration. The recoil constraint increases the population passing the high-recoil cut in both generators, while their angular predictions retain different decay correlations. The script prints the seven-scale envelope and the integration error on each reference. The envelopes quantify scale variation only.

For a new process, provide its positive-weight sample and the corresponding moments in the [user target format](../../docs/NEW_PROCESSES.md). For an example without a recoil window, [targets.json](targets.json) gives the first two dilepton rapidity moments.
